"""Run registered cases serially in the offline local GPU container."""

import argparse
import re
import subprocess
import time

from run_profiled_d0 import now, read, sha, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from tco_one_percent_protocol import ALLOCATOR, CONFIRM_MANIFEST, DATA, IMAGE, MANIFEST, PRIVATE, PUBLIC, ROOT
from tco_one_percent_protocol import TIME_MANIFEST, case_receipts, read_committed, verify


def launch(job, registration, attempt, timeout):
    case_id = job["id"]
    if not re.fullmatch(r"[a-z0-9_]+", case_id):
        raise ValueError("Invalid registered job ID")
    name = f"tco-one-{case_id.replace('_', '-')}-{attempt}"
    target = PRIVATE / "launches" / case_id / attempt
    target.mkdir(parents=True, exist_ok=False)
    command = DOCKER + ["create", "--pull", "never", "--network", "none", "--name", name,
                        "--gpus", "all", "--shm-size", "8g", "--workdir", "/workspace",
                        "--mount", f"type=bind,source={ROOT.as_posix()},target=/workspace",
                        "--mount", f"type=bind,source={DATA.as_posix()},target=/data,readonly",
                        "--mount", "type=volume,source=tco-torch-extensions,target=/root/.cache/torch_extensions",
                        "--env", "MKL_NUM_THREADS=4", "--env", "OPENBLAS_NUM_THREADS=4", "--env", "HF_HUB_OFFLINE=1",
                        "--env", f"PYTORCH_CUDA_ALLOC_CONF={ALLOCATOR}",
                        IMAGE, "python", "-u", "scripts/run_tco_one_percent.py",
                        case_id, attempt, "--registration", registration]
    write_new(target / "start.json", dict(job=case_id, attempt=attempt, utc=now(), argv=command,
                                        rental_compute_usd=0))
    started = time.monotonic()
    container = None
    code = None
    error = None
    finished = None
    stop_issues = []
    print(f"Starting {case_id} ({attempt})", flush=True)
    try:
        running = subprocess.check_output(DOCKER + ["ps", "-q"], text=True, timeout=20).splitlines()
        if running and any(info["HostConfig"].get("DeviceRequests") for info in docker_json(["inspect", *running])):
            raise RuntimeError("The local GPU already has a Docker job")
        container = subprocess.check_output(command, text=True, timeout=45).strip()
        spec = docker_json(["inspect", container])[0]
        if (spec["Image"] != IMAGE or spec["HostConfig"]["NetworkMode"] != "none"
                or f"PYTORCH_CUDA_ALLOC_CONF={ALLOCATOR}" not in spec["Config"]["Env"]):
            raise RuntimeError("The container differs from the registered environment")
        with (target / "stdout.log").open("xb") as log:
            code = subprocess.run(DOCKER + ["start", "--attach", container], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if container:
            finished, stop_issues = ensure_stopped(container)
        stopped = bool(finished and not finished["State"]["Running"])
        log_path = target / "stdout.log"
        log_sha = sha(log_path) if log_path.is_file() else None
        if error is None and (not stopped or code != 0 or finished["State"]["ExitCode"] != 0):
            error = "Container exited unsuccessfully; see private log"
        write_new(target / "terminal.json", dict(job=case_id, attempt=attempt, utc=now(), exit_code=code,
                                                error=error, stopped=stopped, stop_issues=stop_issues,
                                                wall_seconds=time.monotonic() - started, log_sha256=log_sha,
                                                rental_compute_usd=0))
    if container and not stopped:
        raise RuntimeError(f"Unable to establish that {name} stopped")
    print(f"{case_id} {'finished' if error is None else 'failed'} in {time.monotonic() - started:.1f}s",
          flush=True)
    if error:
        print(error, flush=True)
    return error is None


def jobs_for(phase):
    plan = verify()
    if phase == "qualification":
        return plan["probes"], "development", plan["launch_timeout_seconds"]
    quality = read_committed(PUBLIC / "qualification.json")
    if not quality["qualified"]:
        raise ValueError("Finish qualification before timing and scoring")
    if phase in ("fixed", "boundary"):
        jobs = [job for job in plan["jobs"] if job["phase"] == phase
                and (job["arm"] != "cached14" or quality["last14_screen"]["eligible"])]
        return jobs, "development", plan["launch_timeout_seconds"]
    registration = "confirmation" if phase == "confirmation" else "time"
    extension = read_committed(CONFIRM_MANIFEST if phase == "confirmation" else TIME_MANIFEST)
    if extension["development_sha256"] != sha(MANIFEST, text=True):
        raise ValueError("Extension does not bind the development registration")
    jobs = [job for job in extension["jobs"] if job["phase"] == phase]
    return jobs, registration, plan["launch_timeout_seconds"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("qualification", "fixed", "boundary", "timing", "equal_time", "confirmation"))
    cli.add_argument("--job", help="Run just this registered case")
    cli.add_argument("--retry", action="store_true", help="Retry an unsuccessful case once, with identical settings")
    args = cli.parse_args()
    with host_lock():
        jobs, registration, timeout = jobs_for(args.phase)
        if args.job:
            jobs = [next(job for job in jobs if job["id"] == args.job)]
        failed = []
        for job in jobs:
            rows, attempts = case_receipts(job)
            if rows or len(attempts) >= 2 or (attempts and not args.retry):
                if not rows:
                    failed.append(job["id"])
                continue
            if not launch(job, registration, f"attempt0{len(attempts) + 1}", timeout):
                failed.append(job["id"])
        print(f"{args.phase}: {len(jobs) - len(failed)}/{len(jobs)} registered cases complete")
        if failed:
            print("Unsuccessful:", " ".join(failed))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
