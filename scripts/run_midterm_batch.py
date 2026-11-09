"""Bounded reproduction batch for a 48GB GPU, with fresh models per run."""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default="/workspace/tco-assets")
parser.add_argument("--max-hours", type=float, default=4.0)
parser.add_argument("--max-sequence-minutes", type=float, default=20.0)
parser.add_argument("--run-prefix", default="cloud")
parser.add_argument("--dry-run", action="store_true", help="Print commands and validate the protocol without executing any model.")
args = parser.parse_args()
assert re.fullmatch(r"[A-Za-z0-9_-]+", args.run_prefix)
assert args.max_hours > 0 and args.max_sequence_minutes > 0
root = pathlib.Path(__file__).resolve().parents[1]
(root / "reports").mkdir(exist_ok=True)
data_root = pathlib.Path(args.data_root)
manifest = json.loads((root / "manifests/midterm_3scene_kf200.json").read_text())
deadline = time.monotonic() + args.max_hours * 3600
jobs = []
for seq, frames in manifest.items():
    for steps in [0, 40]:
        run_id = f"{args.run_prefix}_{seq.replace('/', '_')}_518_s{steps}_seed42"
        existing = root / "runs" / run_id / "run.json"
        if not args.dry_run and existing.exists() and json.loads(existing.read_text())["status"] == "complete":
            saved = json.loads(existing.read_text())["arguments"]
            assert all(saved[k] == v for k, v in {"scene": seq, "frames": frames, "steps": steps, "size": 518, "seed": 42}.items()), "Completed run protocol mismatch"
            jobs.append({"run_id": run_id, "status": "previously_complete"})
            continue
        if not args.dry_run and existing.parent.exists():
            raise RuntimeError(f"Existing incomplete job {run_id}; preserve it and choose an explicit rerun ID.")
        command = [sys.executable, "-u", str(root / "scripts/run_baseline.py"),
                   "--run-id", run_id, "--scene", seq, "--size", "518", "--steps", str(steps),
                   "--data-root", str(data_root / "data/7scenes"),
                   "--dataset-cache", str(data_root / "data/7scenes/full_sequence_metadata.npy"),
                   "--checkpoint", str(data_root / "checkpoints/VGGT-1B"),
                   "--frames", *map(str, frames)]
        if args.dry_run:
            jobs.append({"run_id": run_id, "command": command})
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batch time budget reached. Stop the rented pod after saving evidence.")
        logdir = root / "runs/logs"
        logdir.mkdir(parents=True, exist_ok=True)
        print("Starting", run_id, flush=True)
        with (logdir / f"{run_id}.log").open("w") as log:
            try:
                completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=min(args.max_sequence_minutes * 60, remaining))
            except subprocess.TimeoutExpired:
                if existing.exists():
                    record = json.loads(existing.read_text())
                    record.update(status="aborted_hardware_timeout", termination_reason="Cloud per-job or batch timeout")
                    existing.write_text(json.dumps(record, indent=2) + "\n")
                jobs.append({"run_id": run_id, "status": "timeout"})
                (root / "reports/batch_status.json").write_text(json.dumps(jobs, indent=2))
                raise
        jobs.append({"run_id": run_id, "exit_code": completed.returncode})
        (root / "reports/batch_status.json").write_text(json.dumps(jobs, indent=2))
        if completed.returncode:
            raise RuntimeError(f"Stopping batch after failed job: {run_id}")
        if not existing.exists() or json.loads(existing.read_text())["status"] != "complete":
            raise RuntimeError(f"Job exited without a completed evidence record: {run_id}")
        print("Completed", run_id, flush=True)
if args.dry_run:
    assert len(jobs) == len(manifest) * 2
    print(json.dumps({"status": "dry_run_only", "jobs": jobs}, indent=2))
else:
    (root / "reports/batch_status.json").write_text(json.dumps(jobs, indent=2))
    print("Batch finished. Save runs/ and stop the rented pod; finishing this script does not stop billing.", flush=True)
