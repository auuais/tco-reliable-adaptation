"""Retry timed-out TCO controls with earlier CUDA allocator reclamation."""
import argparse
import subprocess
import time

from run_profiled_d0 import read, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from tco_frozen_prefix_protocol import IMAGE, PRIVATE, PUBLIC, collect, decision, verify


def retry(job, source_attempt):
    launch = PRIVATE/"launches"/job/source_attempt
    origin = read(launch/"terminal.json")
    original = read(launch/"start.json")["argv"]
    if origin["job"] != job or origin["stopped"] is not True or "TimeoutExpired" not in (origin["error"] or ""):
        raise ValueError("Diagnostic recovery applies to a recorded, stopped, timed-out launch")
    if original.index(IMAGE) >= len(original)-1 or original[-3:] != [job, "--attempt", source_attempt]:
        raise ValueError("Original launch arguments do not match the registered job")
    attempt = "allocator75"
    output = PRIVATE/"allocator_diagnostics"/job/attempt
    output.mkdir(parents=True, exist_ok=False)
    name = f"tco-frozen-{job.replace('_', '-')}-{attempt}"
    original[original.index("--name")+1] = name
    original[-1] = attempt
    original[original.index(IMAGE):original.index(IMAGE)] = [
        "--env", "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.75"]
    write_new(output/"start.json", dict(job=job, attempt=attempt, source_attempt=source_attempt,
                                      argv=original, rental_compute_usd=0))
    container = None
    code = None
    error = None
    start = time.monotonic()
    try:
        container = subprocess.check_output(original, text=True, timeout=45).strip()
        state = docker_json(["inspect", container])[0]
        if state["Image"] != IMAGE or state["HostConfig"]["NetworkMode"] != "none":
            raise RuntimeError("Unexpected diagnostic runtime")
        with (output/"stdout.log").open("xb") as log:
            code = subprocess.run(DOCKER+["start", "--attach", container], stdout=log,
                                  stderr=subprocess.STDOUT, timeout=360).returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        final, issues = ensure_stopped(container) if container else (None, [])
        stopped = bool(final and not final["State"]["Running"])
        result = PRIVATE/"cases"/job/attempt/"complete.json"
        write_new(output/"terminal.json", dict(job=job, attempt=attempt, exit_code=code, error=error,
                  stopped=stopped, stop_issues=issues, wall_seconds=time.monotonic()-start,
                  result=read(result) if stopped and code == 0 and result.exists() else None,
                  rental_compute_usd=0))
    print(job, attempt, code, error, flush=True)


def summarize(contract):
    baseline = collect(contract)
    runs = [read(p) for p in sorted((PRIVATE/"allocator_diagnostics").glob("*/*/terminal.json"))]
    pending = {j["id"] for j in contract["jobs"]} - {r["id"] for r in baseline["rows"]}
    if not {r["job"] for r in runs} <= pending:
        raise ValueError("Diagnostic receipts must fill gaps in the original experiment")
    recovery = [r["result"] for r in runs if r["result"] is not None]
    changed = dict(baseline, rows=[*baseline["rows"], *recovery])
    document = dict(allocator="native", garbage_collection_threshold=.75,
                    separately_timed=True, attempts=runs, rental_compute_usd=0)
    write_new(PUBLIC/"allocator_diagnostics.json", document)
    if len(changed["rows"]) == len(contract["jobs"]):
        write_new(PUBLIC/"allocator_sensitivity_decision.json", decision(contract, changed))
    print(f"Recovered {len(recovery)} missing controls with a different allocator setting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", nargs="?")
    parser.add_argument("source_attempt", nargs="?")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    contract = verify()
    if args.summarize and args.job is None and args.source_attempt is None:
        summarize(contract)
    elif (args.summarize or args.job not in {j["id"] for j in contract["jobs"]} or
          not args.source_attempt or not args.source_attempt.isalnum()):
        parser.error("Unknown registered job or invalid attempt")
    else:
        with host_lock():
            retry(args.job, args.source_attempt)
