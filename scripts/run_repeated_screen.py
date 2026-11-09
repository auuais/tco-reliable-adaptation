"""Frozen follow-up to the aborted v2 bridge: descriptive execution repeats."""
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
STUDY = "diagnostics_v3_repeats"


def plan():
    conditions = [
        ("clean40", "clean", 40, 1.0),
        ("clean10", "clean", 10, 1.0),
        ("severe40", "mixed_severe", 40, 1.0),
        ("severe_weak40", "mixed_severe", 40, 0.1),
    ]
    jobs = []
    for repeat in [1, 2, 3]:
        block = []
        for scene in ["heads/seq-01", "chess/seq-03"]:
            for condition, profile, steps, weight in conditions:
                block.append(dict(run_id=f"diag3_{scene.replace('/', '_')}_{condition}_r{repeat}",
                    scene=scene, condition=condition, profile=profile, steps=steps,
                    weight=weight, repeat=repeat, seed=42, prior_seed=1729,
                    frames=[0, 200, 400, 600, 800], size=518))
        random.Random(7200 + repeat).shuffle(block)
        jobs.extend(block)
    return dict(study_id=STUDY, jobs=jobs, created_after="v2 clean bridge mismatch; original-runner and numerical probes",
        scope="exploratory; two previously inspected sequences; not a benchmark or novelty claim",
        repeats="Three same-seed fresh-process executions per condition quantify numerical variation; not three optimization seeds",
        execution_mode="Original upstream numerical mode; no deterministic override, precision or loss repair",
        inference_inputs="Images and supplied priors only; reference depth/clean hidden priors are evaluation-only",
        reporting="All individual runs plus per-scene mean/min/max; no confidence intervals or statistical superiority claim",
        advancement_gate="Large effects that survive observed within-condition ranges; otherwise increase repeats or abandon small-effect claims",
        max_minutes=55, per_job_seconds=240,
        stop_conditions=["first failed/nonfinite run", "per-job timeout", "batch deadline", "24 jobs complete"],
        original_v2_status="Aborted; frozen manifest and partial outputs preserved, tolerance not loosened",
        source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ["scripts/run_diagnostic.py", "scripts/prior_corruptions.py", "configs/7scenes_baseline.yaml", "configs/diagnostic_profiles.json"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    p = plan()
    manifest = ROOT / f"manifests/{STUDY}.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == p, "Frozen plan differs; use a new study ID"
    else:
        manifest.write_text(json.dumps(p, indent=2) + "\n")
    assert len(p['jobs']) == len({j['run_id'] for j in p['jobs']}) == 24
    if args.dry_run:
        print(json.dumps(dict(jobs=24, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
        return
    report = ROOT / "reports" / STUDY
    report.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + p['max_minutes'] * 60
    status = []
    for j in p['jobs']:
        assert not (ROOT / "runs" / j['run_id']).exists(), "Never overwrite evidence"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batch deadline reached")
        cmd = [sys.executable, "-u", "scripts/run_diagnostic.py", "--run-id", j['run_id'],
            "--study-id", STUDY, "--scene", j['scene'], "--size", str(j['size']),
            "--steps", str(j['steps']), "--seed", str(j['seed']),
            "--prior-profile", j['profile'], "--prior-seed", str(j['prior_seed']),
            "--prior-weight", str(j['weight']), "--frames", *map(str, j['frames']),
            "--data-root", "/workspace/tco-assets/data/7scenes",
            "--dataset-cache", "/workspace/tco-assets/data/7scenes/full_sequence_metadata.npy",
            "--checkpoint", "/workspace/tco-assets/checkpoints/VGGT-1B"]
        entry = dict(run_id=j['run_id'], status="running", command=cmd,
                     utc_start=datetime.datetime.now(datetime.timezone.utc).isoformat())
        status.append(entry)
        def save():
            (report / "batch_status.json").write_text(json.dumps(status, indent=2) + "\n")
        save()
        print("Starting", len(status), "/24", j['run_id'], flush=True)
        try:
            with (ROOT / "runs/logs" / (j['run_id'] + ".log")).open("w") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(remaining, p['per_job_seconds']))
            assert result.returncode == 0, f"Job failed: {j['run_id']}"
            record = json.loads((ROOT / "runs" / j['run_id'] / "run.json").read_text())
            assert record['status'] == 'complete'
            assert record['runner_sha256'] == p['source_sha256']['scripts/run_diagnostic.py']
            entry['status'] = 'complete'
        except BaseException as exc:
            entry.update(status='failed_or_timed_out', error=repr(exc))
            raise
        finally:
            save()
        print("Completed", len(status), "/24", j['run_id'], flush=True)


if __name__ == "__main__":
    main()
