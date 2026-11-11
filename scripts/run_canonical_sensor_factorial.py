"""Replay the four registered-camera conditions on the original full-size GPU model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluation_stability import dump, sha


ROOT = Path(__file__).resolve().parents[1]
STUDY = "canonical_sensor_factorial"
PLAN = ROOT / "manifests" / (STUDY + ".json")
DESIGN = ROOT / "manifests/sensor_tco_v17.json"
RAW = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
CACHE = ROOT / "environment/development_dataset_cache_v7.npy"
MODEL_SHA = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"


def read(path):
    return json.loads(path.read_text())


def protocol():
    parent = read(DESIGN)
    paths = [DESIGN, Path(__file__), ROOT / "scripts/evaluate_canonical_sensor_factorial.py",
             ROOT / "scripts/evaluate_sensor_tco.py", ROOT / "scripts/diagnose_sensor_components.py",
             ROOT / "scripts/repeatability_capture.py", ROOT / "scripts/prior_ablation_config.py",
             ROOT / "configs/7scenes_baseline.yaml", ROOT / "patches/0001-bfloat16-numpy-export.patch",
             ROOT / "manifests/sensor_tco_readout_replication_evaluation.json"]
    return dict(study=STUDY, upstream_commit=parent["upstream_commit"],
                jobs=[dict(j, run_id=j["run_id"].replace("sensor17_", "sensor17c_", 1)) for j in parent["jobs"]],
                **{k: parent[k] for k in ("frames","seed","size","steps","learning_rate","analysis","dataset_cache_sha256")},
                sources={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                model_sha256=MODEL_SHA,
                method="Original TCO model and GPU path. Cross the supplied sensor-to-RGB pose transform and the registered focal calibration. Fresh process for every cell. Same three offset-50 development windows and frozen analysis as the local experiment. Preserve both original image-based and external camera objectives.",
                recovery="Stop the batch on any technical failure. Keep its partial trace. A repeat requires a distinct attempt after a diagnosed technical failure, within the same budget.",
                capture="40 optimizer updates per run; compare 33 frozen point maps on both dense references and one independent source of RGB camera poses. Scoring is local.",
                stage_cap_usd=3, additional_session_cap_usd=1, later_allocations_released=False)


def source_check(spec):
    if read(PLAN) != spec or sha(CACHE) != spec["dataset_cache_sha256"]:
        raise ValueError("Protocol or frozen dataset cache changed")
    for name, expected in read(DESIGN)["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed parent source: " + name)
    upstream = ["git", "-c", "core.fileMode=false", "-c", "core.autocrlf=input", "-C", str(ROOT / "third_party/TCO")]
    if subprocess.check_output(upstream + ["rev-parse","HEAD"], text=True).strip() != spec["upstream_commit"]:
        raise ValueError("Unexpected upstream version")
    changed = subprocess.check_output(upstream + ["diff","--numstat","--ignore-cr-at-eol","HEAD"],text=True)
    if [line.split("\t",2)[-1] for line in changed.splitlines()] != ["utils/interfaces.py"]:
        raise ValueError("Expected only the bfloat16 NumPy export patch upstream")
    subprocess.run(upstream + ["apply","--reverse","--check",str(ROOT / "patches/0001-bfloat16-numpy-export.patch")], check=True)


def worker(spec, job, attempt, assets):
    import run_memory_probe
    import run_repeat_calibration
    import run_sensor_tco
    # Only the prior intervention from the parent factorial is used. Its
    # memory-probe entry point is replaced BEFORE it can change the model.
    def full_gpu():
        sys.argv.extend(["--data-root",str(assets / "data/7scenes"),
                         "--checkpoint",str(assets / "checkpoints/VGGT-1B")])
        run_repeat_calibration.main()

    run_sensor_tco.plan = lambda: spec
    run_sensor_tco.NAME, run_sensor_tco.MANIFEST = STUDY, PLAN
    run_memory_probe.main = full_gpu
    run_sensor_tco.run_one(job, attempt)


def check_record(folder, spec, job):
    record = read(folder / "run.json")
    losses, gradients = (read(folder / name) for name in ("losses.json", "gradients.json"))
    if (record["status"] != "complete" or record["sensor_condition"] != job["condition"]
            or record["arguments"]["scene"] != job["scene"] or record["base_parameters_modified"]
            or record.get("memory_optimization") or record["manifest_sha256"] != sha(PLAN)
            or record["research_dirty"] or record["checkpoint_download"]["sha256"] != MODEL_SHA
            or len(losses) != spec["steps"] or len(gradients) != spec["steps"]
            or not all(g["finite"] and g["tensors_with_gradient"] == 384 for g in gradients)):
        raise ValueError("Incomplete or noncanonical trajectory in " + folder.name)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "dry-run", "run", "worker"))
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--job", type=int, choices=range(12))
    p.add_argument("--attempt", type=int, choices=(1,2), default=1)
    args = p.parse_args()
    spec = protocol()
    files = {f"data/7scenes/{s}/frame-{f:06d}{ext}" for s in {j["scene"] for j in spec["jobs"]}
             for f in spec["frames"] for ext in (".color.png",".depth.proj.png",".pose.txt")}
    if args.action == "freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN, dict(spec, frame_sha256={k: sha(args.assets / k) for k in sorted(files)}))
        return
    hashes = read(PLAN)["frame_sha256"]
    if set(hashes) != files or any(sha(args.assets / k) != v for k,v in hashes.items()):
        raise ValueError("One or more of the 45 RGB, depth and pose files changed")
    spec.update(frame_sha256=hashes)
    source_check(spec)
    if args.action == "worker":
        if args.job is None:
            p.error("worker requires --job")
        worker(spec, spec["jobs"][args.job], args.attempt, args.assets)
        return
    import hashlib
    with (args.assets / "checkpoints/VGGT-1B/model.safetensors").open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda:stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA:
            raise ValueError("Checkpoint bytes changed")
    if args.action == "dry-run":
        print("Verified checkpoint, 45 input files, cache, original TCO patch and 12 frozen jobs.")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    for i,job in enumerate(spec["jobs"]):
        if args.job is not None and i != args.job:
            continue
        tag = job["run_id"] + f"_a{args.attempt}"
        result = RAW / (tag + ".json")
        folder = ROOT / "runs" / tag
        if result.exists() and read(result)["status"] == "complete":
            if read(result)["record_sha256"] != sha(folder / "run.json"):
                raise ValueError("Completed record changed")
            continue
        if args.attempt == 2 and read(RAW / (job["run_id"] + "_a1.json"))["status"] != "interrupted":
            raise ValueError("Only interrupted trajectories can be retried")
        with (RAW / (tag + ".log")).open("x") as log:
            print("Starting", tag, flush=True)
            try:
                subprocess.run([sys.executable,"-u",__file__,"worker","--assets",str(args.assets),
                                "--job",str(i),"--attempt",str(args.attempt)], stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ,PYTHONHASHSEED=str(spec["seed"]),OMP_NUM_THREADS="4",
                                        OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4"),
                               check=True,timeout=360)
                record = check_record(folder, spec, job)
                dump(result,dict(**job,attempt=args.attempt,status="complete",gpu=record["gpu"],
                                 record_sha256=sha(folder / "run.json")))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                dump(result,dict(**job,attempt=args.attempt,status="interrupted",reason=str(exc)))
                raise


if __name__ == "__main__":
    main()
