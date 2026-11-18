"""Run a serial local tranche from the immutable 36-state redesign inventory."""
import argparse
import json

from profiled_compensation_protocol import EXECUTION, ROOT, require
from register_profiled_compensation import verify_release
from run_profiled_batch import batch_lock, verify_completed
from run_profiled_compensation_local import check_local_engine, host_lock, launch
from score_profiled_compensation import ATTEMPT


def drive(registry, limit, *, dry_run):
    available = len(registry["jobs"])
    manifest = ROOT / EXECUTION
    previously_complete, launched = 0, 0
    for job in registry["jobs"]:
        complete = verify_completed(ROOT, manifest, registry, job, ATTEMPT)
        if complete is not None:
            previously_complete += 1
            if not dry_run:
                print(f"Verified existing {job['run_id']}", flush=True)
            continue
        if dry_run:
            continue
        if launched >= limit:
            break
        name = "tco-profiled-compensation-" + job["run_id"].lower().replace("_", "-")
        argv = [token if token != "FRESH_ATTEMPT_ID" else ATTEMPT for token in job["planned_argv"][1:]]
        launch(name, argv, registry["local_limits"]["case_seconds"])
        verify_completed(ROOT, manifest, registry, job, ATTEMPT)
        launched += 1
    print(json.dumps(dict(already_complete=previously_complete, newly_complete=launched,
                          inventory=available, dry_run=dry_run, rental_usd=0)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=1, help="Maximum fresh states for this local tranche")
    args = p.parse_args()
    require(1 <= args.limit <= 36, "Tranche limit must be between one and 36")
    registry, _ = verify_release()
    if args.dry_run:
        drive(registry, args.limit, dry_run=True)
    else:
        with host_lock(), batch_lock("compensation-screen01"):
            check_local_engine()
            drive(registry, args.limit, dry_run=False)


if __name__ == "__main__":
    main()
