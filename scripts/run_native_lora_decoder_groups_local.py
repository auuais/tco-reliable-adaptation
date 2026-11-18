"""Launch registered development cases on the local offline RTX 3080."""
import argparse
import re
import subprocess
import time
from uuid import uuid4

from native_lora_decoder_groups import RAW, registered
from run_profiled_d0 import IMAGE, read, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import LABEL, check_local_engine, create_command, host_lock, inspect_matches


def finished(directory):
    launches = [read(path) for path in sorted(directory.glob("*/terminal.json"))]
    return any(row["stopped"] and row["exit_code"] == 0 and not row["error"] for row in launches)


def launch(job, attempt):
    base = RAW / "launches" / job
    if finished(base):
        print(job, "completed in a prior attempt", flush=True)
        return
    check_local_engine()
    receipt = base / attempt / "terminal.json"
    receipt.parent.mkdir(parents=True, exist_ok=False)
    label = uuid4().hex
    name = f"tco-profiled-decoder-{job.replace('_', '-')}-{attempt}"
    command = create_command("model", name, label, None)
    command = command[:command.index(IMAGE)+1] + [
        "python", "-u", "scripts/run_native_lora_decoder_groups.py", job, attempt]
    write_new(receipt.with_name("start.json"), dict(container=name, argv=command, rental_compute_usd=0))
    start = time.monotonic()
    container, code, error = None, None, None
    try:
        container = subprocess.check_output(command, text=True, timeout=30).strip()
        if not inspect_matches(docker_json(["inspect", container])[0], label, "model"):
            raise RuntimeError("Local container has wrong ownership, resource limits, or offline policy")
        with receipt.with_name("stdout.log").open("xb") as out:
            code = subprocess.run(DOCKER + ["start", "--attach", container], stdout=out,
                                  stderr=subprocess.STDOUT, timeout=1800).returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if container is None:
            try:
                item = docker_json(["inspect", name])[0]
                if item["Config"]["Labels"].get(LABEL) == label:
                    container = item["Id"]
            except subprocess.SubprocessError:
                pass
    finally:
        final, cleanup = ensure_stopped(container) if container else (None, [])
        state = final["State"] if final else None
        stopped = bool(state and not state["Running"])
        elapsed = time.monotonic()-start
        write_new(receipt, dict(job=job, attempt=attempt, exit_code=code, stopped=stopped,
                                error=error, cleanup=cleanup, state=state, wall_seconds=elapsed,
                                rental_compute_usd=0))
    if code != 0 or error or not stopped or state["ExitCode"] != 0:
        raise RuntimeError(f"Local native endpoint failed. See {receipt.parent}")
    print(job, attempt, round(elapsed, 1), "seconds, stopped", flush=True)


def main():
    jobs = [case["id"] for case in registered()["development_cases"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=jobs)
    parser.add_argument("--attempt", default="attempt01")
    args = parser.parse_args()
    if not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        parser.error("Use an attempt ID such as attempt01 or attempt02")
    with host_lock():
        if args.job and args.job != jobs[0] and not finished(RAW / "launches" / jobs[0]):
            raise RuntimeError("Run the historical bridge before any other case")
        for job in jobs:
            if args.job is None or args.job == job:
                launch(job, args.attempt)


if __name__ == "__main__":
    main()
