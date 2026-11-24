"""Run one registered series of full TCO cases in fresh offline Docker containers."""

import argparse
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4

from run_profiled_d0 import now, read, sha, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from tco_full40_recompute_protocol import DATA, IMAGE, PRIVATE, PUBLIC, REGISTRATIONS, ROOT, STUDY
from tco_full40_recompute_protocol import verify
from tco_one_percent_protocol import ALLOCATOR


LABEL = "tco.reconstruction.study"
MAX_ATTEMPTS = 2


def case_path(job, attempt):
    return PRIVATE / "cases" / job["id"] / attempt / "case"


def launch_path(job, attempt):
    return PRIVATE / "launches" / job["id"] / attempt


def check_device_available():
    running = subprocess.check_output(DOCKER + ["ps", "-q"], text=True, timeout=25).splitlines()
    if running and any(item["HostConfig"].get("DeviceRequests") for item in docker_json(["inspect", *running])):
        raise RuntimeError("A Docker job already owns the local GPU")


def create_argv(job, registration, attempt, name):
    cpu_fixture = job["profile"] == "fixture"
    command = DOCKER + ["create", "--name", name, "--pull", "never", "--network", "none",
                        "--label", f"{LABEL}={STUDY}", "--shm-size", "8g", "--workdir", "/workspace",
                        "--mount", f"type=bind,source={ROOT.as_posix()},target=/workspace",
                        "--mount", f"type=bind,source={DATA.as_posix()},target=/data,readonly",
                        "--mount", "type=volume,source=tco-torch-extensions,target=/root/.cache/torch_extensions",
                        "--env", "MKL_NUM_THREADS=4", "--env", "OPENBLAS_NUM_THREADS=4",
                        "--env", "OMP_NUM_THREADS=4", "--env", "HF_HUB_OFFLINE=1",
                        "--env", f"PYTORCH_CUDA_ALLOC_CONF={ALLOCATOR}"]
    command += ["--env", "CUDA_VISIBLE_DEVICES="] if cpu_fixture else ["--gpus", "all"]
    return command + [IMAGE, "python", "-u", "scripts/run_tco_full40_recompute.py", job["id"],
                      attempt, "--registration", registration]


def validate_complete(job, result, plan):
    for field, value in job.items():
        if result.get(field) != value:
            raise ValueError(f"{job['id']} result does not match its registered {field}")
    if result.get("rental_compute_usd") != 0:
        raise ValueError(f"{job['id']} result does not attest zero rental compute")
    if job["cohort"] == "confirmation":
        expected, = (item for item in plan["expected_confirmation_signatures"]
                     if (item["dataset"], item["scene"]) == (job["dataset"], job["scene"]))
        for key in ("input_sha256", "reference_sha256"):
            if result.get(key) != expected[key]:
                raise ValueError(f"{job['id']} used different decoded benchmark inputs")


def case_status(job, plan):
    attempts = []
    successes = []
    folder = PRIVATE / "launches" / job["id"]
    folders = sorted(path for path in folder.glob("attempt*") if path.is_dir())
    for path in folders:
        terminal = path / "terminal.json"
        if not terminal.is_file():
            raise RuntimeError(f"Unresolved launcher receipt in {path}; inspect that exact attempt")
        attempt = read(terminal)
        if attempt["job"] != job["id"] or attempt["attempt"] != path.name:
            raise ValueError(f"Mismatched launcher receipt in {terminal}")
        attempts.append(attempt)
        if attempt["verdict"] == "completed":
            complete = case_path(job, path.name) / "complete.json"
            if sha(complete) != attempt["complete_sha256"]:
                raise ValueError(f"Mismatched result checksum for {job['id']}")
            result = read(complete)
            validate_complete(job, result, plan)
            successes.append(result)
    if len(successes) > 1:
        raise ValueError(f"Duplicate successful attempts for {job['id']}")
    verdict = "completed" if successes else attempts[-1]["verdict"] if attempts else "unattempted"
    return dict(id=job["id"], verdict=verdict, attempts=attempts, result=successes[0] if successes else None)


def launch(job, registration, plan, attempt, timeout, timeout_reason):
    identifier = job["id"]
    if not re.fullmatch(r"[a-z0-9_]+", identifier) or not re.fullmatch(r"attempt[0-9]{2}", attempt):
        raise ValueError("Unknown case or attempt name")
    if job["profile"] != "fixture":
        check_device_available()
    name = f"tco-full40-{identifier.replace('_', '-')}-{attempt}"
    target = launch_path(job, attempt)
    target.mkdir(parents=True, exist_ok=False)
    command = create_argv(job, registration, attempt, name)
    write_new(target / "start.json", dict(job=identifier, attempt=attempt, utc=now(), argv=command,
                                         outer_seconds=timeout, timeout_reason=timeout_reason,
                                         rental_compute_usd=0))
    started = time.monotonic()
    container = None
    finished = None
    stop_issues = []
    exit_code = None
    error = None
    provisioned = False
    status = "error"
    log = target / "stdout.log"
    complete = case_path(job, attempt) / "complete.json"
    print(f"Starting {identifier} ({attempt})", flush=True)
    try:
        container = subprocess.check_output(command, text=True, timeout=45).strip()
        actual, = docker_json(["inspect", container])
        if (actual["Image"] != IMAGE or actual["HostConfig"]["NetworkMode"] != "none"
                or actual["Config"]["Labels"].get(LABEL) != STUDY
                or not any(m["Destination"] == "/data" and m["RW"] is False for m in actual["Mounts"])
                or bool(actual["HostConfig"].get("DeviceRequests")) != (job["profile"] != "fixture")
                or f"PYTORCH_CUDA_ALLOC_CONF={ALLOCATOR}" not in actual["Config"]["Env"]):
            raise RuntimeError("The created container differs from the registered environment")
        provisioned = True
        with log.open("xb") as stream:
            exit_code = subprocess.run(DOCKER + ["start", "--attach", container],
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=timeout).returncode
        if exit_code == 0:
            validate_complete(job, read(complete), plan)
            status = "completed"
        elif "CUDA out of memory" in log.read_text(encoding="utf-8", errors="replace"):
            status = "oom"
        else:
            error = "Nonzero exit; see private log"
    except subprocess.TimeoutExpired as exc:
        status = "timeout" if provisioned else "error"
        error = f"{type(exc).__name__}: exceeded {exc.timeout}s"
    except KeyboardInterrupt:
        status, error = "interrupted", "Launcher interrupted"
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if not container:
            # A timeout in docker create can leave an owned, stopped container without returning its ID.
            try:
                possible, = docker_json(["inspect", name])
                if possible["Config"].get("Labels", {}).get(LABEL) == STUDY:
                    container = possible["Id"]
            except (subprocess.SubprocessError, ValueError):
                pass
        if container:
            finished, stop_issues = ensure_stopped(container)
        stopped = bool(finished and not finished["State"]["Running"])
        if status == "completed" and (not stopped or finished["State"]["ExitCode"] != 0):
            status, error = "error", "Container did not exit successfully"
        write_new(target / "terminal.json", dict(
            job=identifier, attempt=attempt, utc=now(), verdict=status, exit_code=exit_code, error=error,
            stopped=stopped, stop_issues=stop_issues, wall_seconds=time.monotonic() - started,
            outer_seconds=timeout, complete_sha256=sha(complete) if status == "completed" else None,
            log_sha256=sha(log) if log.is_file() else None, rental_compute_usd=0))
    if container and not stopped:
        raise RuntimeError(f"Unable to verify that the owned container {name} stopped")
    print(f"{identifier} {status} after {time.monotonic() - started:.1f}s", flush=True)
    return status


def status_rows(plan):
    rows = {job["id"]: case_status(job, plan) for job in plan["jobs"]}
    for job in plan["jobs"]:
        row = rows[job["id"]]
        if row["verdict"] != "unattempted":
            continue
        if job["cohort"] == "ten" and plan.get("five_only"):
            row["reason"] = "five_view_only_lock"
        elif job.get("first_gate") and rows[job["first_gate"]]["verdict"] != "completed":
            row["reason"] = "first_seed_not_completed"
    return rows


def export(registration, plan):
    rows = status_rows(plan)
    path = PUBLIC / ("evidence.json" if registration == "locked" else "initial_evidence.json")
    temp = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    write_new(temp, dict(study=STUDY, registration_sha256=sha(REGISTRATIONS[registration], text=True),
                         runs=list(rows.values()), rental_compute_usd=0))
    temp.replace(path)
    print(f"Exported {len(rows)} case receipts to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", choices=REGISTRATIONS)
    parser.add_argument("--cohort", action="append", help="Select one or more cohorts")
    parser.add_argument("--job", action="append", help="Select one or more exact registered IDs")
    parser.add_argument("--case-limit", type=int, help="Maximum number of fresh attempts in this invocation")
    parser.add_argument("--retry", action="store_true", help="Allow one fresh retry of an unsuccessful job")
    parser.add_argument("--timeout", type=int, help="Override outer deadline in seconds after diagnosing a slow phase")
    parser.add_argument("--timeout-reason", help="Recorded reason for changing the outer deadline")
    parser.add_argument("--export", action="store_true", help="Write compact case evidence without launching")
    parser.add_argument("--status", action="store_true", help="Print current case evidence without launching")
    args = parser.parse_args()
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    if args.timeout is not None and (args.timeout < 1200 or not args.timeout_reason):
        parser.error("A new timeout must be at least 1200 seconds and include --timeout-reason")
    plan = verify(args.registration)
    selected = [job for job in plan["jobs"] if (not args.cohort or job["cohort"] in args.cohort)
                and (not args.job or job["id"] in args.job)]
    if not selected or (args.job and set(args.job) != {job["id"] for job in selected}):
        parser.error("The filters must match registered case IDs")
    if args.status or args.export:
        if args.status:
            rows = status_rows(plan)
            for job in selected:
                row = rows[job["id"]]
                print(f"{job['id']:<39} {row['verdict']:<14} {len(row['attempts'])} attempt(s)"
                      f" {row.get('reason', '')}")
        if args.export:
            export(args.registration, plan)
        return
    if not args.cohort and not args.job:
        parser.error("Select a cohort or specific jobs to launch")
    timeout = args.timeout or plan["launch_timeout_seconds"]
    with host_lock():
        count = 0
        for job in selected:
            row = case_status(job, plan)
            if job["cohort"] == "ten" and plan.get("five_only"):
                continue
            if job.get("first_gate"):
                first = next(other for other in plan["jobs"] if other["id"] == job["first_gate"])
                if case_status(first, plan)["verdict"] != "completed":
                    continue
            attempts = len(row["attempts"])
            if row["verdict"] == "completed" or attempts >= MAX_ATTEMPTS or (attempts and not args.retry):
                continue
            launch(job, args.registration, plan, f"attempt{attempts + 1:02d}", timeout, args.timeout_reason)
            count += 1
            if count == args.case_limit:
                break
        print(f"Launched {count} case(s); previous receipts were retained")


if __name__ == "__main__":
    main()
