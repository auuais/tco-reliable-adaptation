"""Launch TCO frozen-prefix jobs serially in the pinned offline Docker image."""
import argparse
import subprocess
import time
from pathlib import Path

from run_profiled_d0 import read, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from tco_frozen_prefix_protocol import PRIVATE, ROOT, IMAGE, verify

DATA = Path("E:/ResearchData/TCO")


def launch(job, attempt, engineering=None, steps=1, fixed_gradient=False):
    kind = "engineering_launches" if engineering else "launches"
    directory = PRIVATE/kind/job
    for prior in directory.glob("*/terminal.json"):
        record = read(prior)
        if record["exit_code"] == 0 and record["stopped"] and not record["error"]:
            output = PRIVATE/("engineering" if engineering else "cases")/job/prior.parent.name/"complete.json"
            if read(output)["id"] != job:
                raise ValueError("Previous receipt has a different identity")
            return
    dest = directory/attempt
    dest.mkdir(parents=True, exist_ok=False)
    name = f"tco-frozen-{job.replace('_', '-')}-{attempt}"
    command = DOCKER + ["create", "--pull", "never", "--network", "none", "--name", name,
                        "--gpus", "all", "--shm-size", "8g",
                        "--mount", f"type=bind,source={ROOT.as_posix()},target=/workspace",
                        "--mount", f"type=bind,source={DATA.as_posix()},target=/data,readonly",
                        "--mount", "type=volume,source=tco-torch-extensions,target=/root/.cache/torch_extensions",
                        "--env", "MKL_NUM_THREADS=4", "--env", "OPENBLAS_NUM_THREADS=4",
                        IMAGE, "python", "-u", "scripts/run_tco_frozen_prefix.py", job, "--attempt", attempt]
    if engineering:
        command += ["--engineering", engineering, "--steps", str(steps)]
        if fixed_gradient:
            command.append("--fixed-gradient")
    write_new(dest/"start.json", dict(job=job, argv=command, rental_compute_usd=0))
    start = time.monotonic()
    container, exit_code, error = None, None, None
    try:
        running = subprocess.check_output(DOCKER+["ps", "-q"], text=True, timeout=20).splitlines()
        if running:
            if any(item["HostConfig"].get("DeviceRequests") for item in docker_json(["inspect", *running])):
                raise RuntimeError("The local GPU already has a Docker job")
        container = subprocess.check_output(command, text=True, timeout=45).strip()
        item = docker_json(["inspect", container])[0]
        if item["Image"] != IMAGE or item["HostConfig"]["NetworkMode"] != "none":
            raise RuntimeError("Unexpected local runtime settings")
        with (dest/"stdout.log").open("xb") as log:
            exit_code = subprocess.run(DOCKER+["start", "--attach", container], stdout=log,
                                       stderr=subprocess.STDOUT, timeout=1200).returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        final, issues = ensure_stopped(container) if container else (None, [])
        stopped = bool(final and not final["State"]["Running"])
        write_new(dest/"terminal.json", dict(job=job, attempt=attempt, exit_code=exit_code,
                  stopped=stopped, error=error, stop_issues=issues, wall_seconds=time.monotonic()-start,
                  rental_compute_usd=0))
    if exit_code or error or not stopped or final["State"]["ExitCode"] != 0:
        raise RuntimeError(f"Local run failed, see {dest}")
    print(job, attempt, f"{time.monotonic()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="attempt01")
    parser.add_argument("--job")
    parser.add_argument("--engineering")
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--fixed-gradient", action="store_true")
    args = parser.parse_args()
    if not args.attempt.isalnum() or args.job and not args.job.replace("_", "").isalnum():
        parser.error("Invalid attempt or job")
    with host_lock():
        if args.engineering:
            if not args.job:
                parser.error("Engineering runs need a unique --job")
            launch(args.job, args.attempt, args.engineering, args.steps, args.fixed_gradient)
        else:
            if args.fixed_gradient:
                parser.error("Fixed gradients are an engineering diagnostic only")
            plan = verify()
            jobs = plan["jobs"]
            if args.job:
                jobs = [next(row for row in jobs if row["id"] == args.job)]
            for job in jobs:
                launch(job["id"], args.attempt)


if __name__ == "__main__":
    main()
