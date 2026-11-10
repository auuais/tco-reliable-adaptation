"""Collect untouched independent cells after a documented staged supervisor interruption."""
import argparse
import os
import subprocess
import sys

import run_staged_sensor_factorial as study
from evaluation_stability import dump
from score_saved_geometry import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = read(study.PLAN)
    study.check_inputs(plan)
    jobs = [(i, job, study.RAW / (job["run_id"] + "_a1.json")) for i,job in enumerate(plan["jobs"])]
    states = {i:read(path)["status"] for i,_,path in jobs if path.is_file()}
    if any(value not in ("complete", "interrupted") for value in states.values()):
        raise ValueError("Unrecognized previous outcome")
    if args.run and "interrupted" not in states.values():
        raise ValueError("A supervisor interruption is required before continuing")
    for i, job, record in jobs:
        if i in states:
            continue
        folder = study.ROOT / "runs" / (job["run_id"] + "_a1")
        log = record.with_suffix(".log")
        if log.exists() or folder.exists():
            if args.run:
                raise RuntimeError("An existing attempt lacks a terminal record: " + job["run_id"])
            continue
        print("Unstarted", job["run_id"], flush=True)
        if not args.run:
            continue
        # Each original child has a fresh model and RNG. No dependency on the preceding cell.
        with log.open("x") as stream:
            try:
                subprocess.run([sys.executable, "-u", study.__file__, "worker", "--job", str(i)],
                               check=True, timeout=plan["limit_seconds"], stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                        OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONHASHSEED="42"))
                result = study.verify_run(job, plan)
            except subprocess.TimeoutExpired as exc:
                result = dict(**job, status="interrupted", attempt=1, reason=str(exc))
            except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
                dump(record, dict(**job, status="interrupted", attempt=1, reason=str(exc)))
                raise
            dump(record, result)
    if args.run:
        final = {j["run_id"]:read(path)["status"] for _,j,path in jobs}
        dump(study.RAW / "continuation_outcomes.json", final)
        print(final, flush=True)


if __name__ == "__main__":
    main()
