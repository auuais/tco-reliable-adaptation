"""Frozen 39-job learning-rate / visual-only screen with bounded runtime."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = "baselines_v4"
SOURCES = ["scripts/run_experiment.py", "scripts/prior_corruptions.py",
           "scripts/run_baseline_screen.py", "configs/7scenes_baseline.yaml", "configs/diagnostic_profiles.json"]
SCENES = ["heads/seq-01", "chess/seq-03", "fire/seq-03"]


def plan():
    jobs = []
    def job(scene, condition, seed, steps, lr, weight):
        return dict(run_id=f"base4_{scene.replace('/', '_')}_{condition}_seed{seed}",
                    scene=scene, condition=condition, seed=seed, steps=steps,
                    learning_rate=lr, weight=weight, profile="clean", prior_seed=1729,
                    size=518, frames=[0, 200, 400, 600, 800])
    jobs.extend(job(s, "zero", 42, 0, .001, 1.) for s in SCENES)
    for seed in [42, 123, 2026]:
        block = [job(s, c, seed, 40, lr, w) for s in SCENES for c, lr, w in
                 [("released", .001, 1.), ("lr3e4", .0003, 1.),
                  ("lr1e4", .0001, 1.), ("visual", .0003, 0.)]]
        random.Random(8400 + seed).shuffle(block)
        jobs.extend(block)
    return dict(study_id=STUDY, jobs=jobs, max_minutes=70, per_job_seconds=360,
        purpose="Test essential simple baselines before developing reliability control",
        scope="Exploratory; three previously inspected sequences; not held-out evaluation",
        seeds="Three optimizer seeds; numerical execution variance remains unresolved",
        visual_only="lambda_pose=lambda_intrinsics=0; lambda_mv_consistency remains 0.2",
        selection="Inspect all three metrics per scene; no selection by best seed or total loss",
        primary_score="Per-run (accuracy/base_accuracy + completeness/base_completeness)/2; equal weight per scene",
        harm="accuracy OR completeness >1% worse than fresh same-scene zero-step control",
        reporting="All 39 jobs, seed-level results, means and observed ranges; no significance claim",
        sources={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES})


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="Write manifest before code commit; never run models")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assets", type=pathlib.Path, default=pathlib.Path("/workspace/tco-assets"))
    args = parser.parse_args()
    p = plan()
    manifest = ROOT / f"manifests/{STUDY}.json"
    if args.freeze:
        if manifest.exists():
            assert json.loads(manifest.read_text()) == p, "Use a new study ID after freezing"
        else:
            dump(manifest, p)
    assert manifest.exists(), "Freeze and commit protocol before running"
    assert json.loads(manifest.read_text()) == p, "Frozen source or plan mismatch"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(p["jobs"]) == len({j["run_id"] for j in p["jobs"]}) == 39
    if args.freeze or args.dry_run:
        print(json.dumps(dict(jobs=39, manifest_sha256=digest)))
        return
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit all code before running"
    report = ROOT / "reports" / STUDY
    assert not (report/"batch_status.json").exists(), "Preserve all attempts; do not overwrite a batch"
    (ROOT/"runs/logs").mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + p["max_minutes"] * 60
    status = []
    for j in p["jobs"]:
        assert not (ROOT/"runs"/j["run_id"]).exists(), "Never overwrite evidence"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batch deadline reached")
        cmd = [sys.executable, "-u", "scripts/run_experiment.py", "--run-id", j["run_id"],
            "--study-id", STUDY, "--manifest-sha256", digest, "--scene", j["scene"],
            "--size", str(j["size"]), "--steps", str(j["steps"]), "--seed", str(j["seed"]),
            "--learning-rate", str(j["learning_rate"]), "--prior-weight", str(j["weight"]),
            "--frames", *map(str, j["frames"]), "--data-root", str(args.assets/"data/7scenes"),
            "--dataset-cache", str(args.assets/"data/7scenes/full_sequence_metadata.npy"),
            "--checkpoint", str(args.assets/"checkpoints/VGGT-1B")]
        entry = dict(run_id=j["run_id"], status="running", command=cmd,
                     utc_start=datetime.datetime.now(datetime.timezone.utc).isoformat())
        status.append(entry)
        dump(report/"batch_status.json", status)
        print("Starting", len(status), "/39", j["run_id"], flush=True)
        try:
            with (ROOT/"runs/logs"/(j["run_id"]+".log")).open("x") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(remaining, p["per_job_seconds"]))
            assert result.returncode == 0, f"Job failed: {j['run_id']}"
            record = json.loads((ROOT/"runs"/j["run_id"]/"run.json").read_text())
            assert record["status"] == "complete" and not record["research_dirty"]
            assert record["runner_sha256"] == p["sources"]["scripts/run_experiment.py"]
            entry["status"] = "complete"
        except BaseException as exc:
            entry.update(status="failed_or_timed_out", error=repr(exc))
            raise
        finally:
            dump(report/"batch_status.json", status)
        print("Completed", len(status), "/39", j["run_id"], flush=True)


if __name__ == "__main__":
    main()
