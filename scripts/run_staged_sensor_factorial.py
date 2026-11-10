"""Run the complete original calibration design with the validated RTX 3080 storage path."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import probe_staged_initial_gpu as staged
from evaluation_stability import dump, sha
from score_saved_geometry import read


ROOT = staged.ROOT
STUDY = "staged_sensor_factorial"
PLAN = ROOT / "manifests" / (STUDY + ".json")
RAW = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
ORIGINAL = ROOT / "manifests/sensor_tco_v17.json"
PROBE = ROOT / "results/staged_initial_gpu_probe/evidence.json"


def protocol():
    design = read(ORIGINAL)
    paths = [Path(__file__).resolve(), ROOT / "scripts/evaluate_staged_sensor_factorial.py",
             ROOT / "scripts/run_sensor_readout_replication.py", ROOT / "scripts/probe_staged_initial_gpu.py",
             ROOT / "scripts/probe_cached_operands_gpu.py", ROOT / "scripts/stage_initial_forward.py",
             ROOT / "scripts/frozen_linear_storage.py", ROOT / "scripts/evaluate_sensor_tco.py",
             ROOT / "scripts/diagnose_sensor_components.py", ROOT / "manifests/sensor_component_metrics.json",
             ROOT / "manifests/sensor_tco_readout_replication_evaluation.json",
             ROOT / "manifests/sensor_tco_readout_replication.json", ROOT / "manifests/staged_initial_gpu_probe.json",
             ROOT / "manifests/cached_operand_gpu_probe.json", ROOT / "patches/tco-selected-readouts.patch",
             ROOT / "scripts/selected_readouts.py", ORIGINAL, PROBE]
    return dict(study=STUDY, jobs=[dict(job, run_id=job["run_id"].replace("sensor17_", "sensor17p_", 1))
                                   for job in design["jobs"]], seed=design["seed"], frames=design["frames"],
                steps=design["steps"], size=design["size"], learning_rate=design["learning_rate"],
                upstream_commit=design["upstream_commit"], dataset_cache_sha256=design["dataset_cache_sha256"],
                sources={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}, limit_seconds=1800, cloud_usd=0,
                method=design["method"], engineering="Use the same upstream code, staged initial inference, selected four readouts and host-backed BF16 projection operands as the successful eight-update probe in every cell. Run each optimizer trajectory in a fresh process. Include original initialization comparisons wherever original evidence exists.",
                analysis=design["analysis"], dense_evaluations=66, secondary_metrics="Apply the registered depth and published RGB SfM component diagnostics to all 33 point maps.")


def check_inputs(spec):
    if spec != protocol():
        raise ValueError("Staged study changed after freeze")
    if read(PROBE)["container_exit_code"] != 0 or not read(PROBE)["supervisor"]["concordant"]:
        raise ValueError("Eight-update CUDA engineering control did not pass")
    if sha(ROOT / "environment/development_dataset_cache_v7.npy") != spec["dataset_cache_sha256"]:
        raise ValueError("Dataset cache mismatch")
    for rel, digest in read(ORIGINAL)["sources"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("Parent source mismatch in " + rel)
    revision = subprocess.check_output(["git", "-C", str(ROOT / "third_party/TCO"), "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["upstream_commit"]:
        raise ValueError("Unexpected upstream TCO revision")


def worker(job, steps):
    # The frozen probe worker resolves its experiment settings in a fresh Python process.
    # Use precisely that path for every condition with the original forty-step budget.
    staged.FOLDER = ROOT / "runs" / (job["run_id"] + "_a1")
    staged.JOB, staged.NAME, staged.PLAN = job, STUDY, PLAN
    staged.previous.protocol = lambda: {"steps": steps}
    staged.worker()


def verify_run(job, spec):
    folder = ROOT / "runs" / (job["run_id"] + "_a1")
    record = read(folder / "run.json")
    trace = read(folder / "gradients.json")
    losses = read(folder / "losses.json")
    if (record["status"] != "complete" or record["manifest_sha256"] != sha(PLAN)
            or record["arguments"]["steps"] != spec["steps"] or record["base_parameters_modified"]
            or len(trace) != spec["steps"] or len(losses) != spec["steps"]
            or not all(g["finite"] and g["tensors_with_gradient"] == 384 for g in trace)):
        raise ValueError("A full adaptation trajectory is missing or invalid")
    original = folder.with_name(folder.name.replace("sensor17p_", "sensor17_", 1))
    bridge = None
    if (original / "capture.json").exists():
        left, right = (read(f / "capture.json")["records"] for f in (original, folder))
        previous = read(original / "run.json")
        bridge = {key: left[key][field] == right[key][field] for key, field in
                  (("initial_predictions", "tensor_sha256"), ("initial_adapters", "tensor_sha256"),
                   ("after_adapter_initialization_rng", "sha256")) if key in left}
        bridge["inputs"] = previous["evaluation_tensor_sha256"] == record["evaluation_tensor_sha256"]
        bridge["supplied_prior"] = previous["prior_metadata"] == record["prior_metadata"]
        if not all(bridge.values()):
            raise ValueError("A matched original starting state changed")
    storage = read(folder / "storage.json")
    phases = read(folder / "initial_phases.json")
    if storage["projections"] != 192 or "ready_for_adapters" not in phases:
        raise ValueError("Unexpected storage or initialization path")
    return dict(**job, attempt=1, status="complete", record_sha256=sha(folder / "run.json"),
                initial_bridge=bridge, initial_total_loss=losses[0]["total"], storage=storage, phases=phases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "worker"))
    parser.add_argument("--job", type=int)
    args = parser.parse_args()
    if args.action == "freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN, protocol())
        return
    spec = read(PLAN)
    check_inputs(spec)
    if args.action == "worker":
        worker(spec["jobs"][args.job], spec["steps"])
        return
    RAW.mkdir(parents=True, exist_ok=True)
    for i, job in enumerate(spec["jobs"]):
        path = RAW / (job["run_id"] + "_a1.json")
        if path.is_file():
            if read(path) != verify_run(job, spec):
                raise ValueError("Previously recorded attempt requires examination")
            continue
        print("Starting", job["run_id"], flush=True)
        with path.with_suffix(".log").open("x") as log:
            try:
                subprocess.run([sys.executable, "-u", __file__, "worker", "--job", str(i)], check=True,
                               timeout=spec["limit_seconds"], stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                        OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONHASHSEED="42"))
                result = verify_run(job, spec)
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                dump(path, dict(**job, status="interrupted", attempt=1, reason=str(exc)))
                raise
            dump(path, result)


if __name__ == "__main__":
    main()
