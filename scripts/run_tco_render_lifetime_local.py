"""Launch only registered render-lifetime cases in the pinned offline environment."""

import argparse
from collections import defaultdict
import json
import re
from unittest.mock import patch
from uuid import uuid4

from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock
import run_tco_courtyard_camera_local as previous
from tco_render_lifetime_protocol import (
    CPU_PROFILES, IMAGE, INITIAL, LOCKED, PRIVATE, PUBLIC, REGISTRATIONS, ROOT, STUDY,
    hashes_for, verify, window_key,
)


def case_path(job, attempt):
    return PRIVATE / "cases" / job["id"] / attempt / "case"


def launch_path(job, attempt):
    return PRIVATE / "launches" / job["id"] / attempt


def create_command(job, registration, attempt, name):
    command = previous.create_argv(job, registration, attempt, name)
    script = "scripts/run_tco_render_lifetime.py"
    if job.get("profile") == "input_audit":
        script = "scripts/run_tco_render_lifetime_local.py"
        return command[:command.index("scripts/run_tco_courtyard_camera.py")] + [script, "--audit-inside"]
    command[command.index("scripts/run_tco_courtyard_camera.py")] = script
    return command


def validated(job, result, plan):
    if any(result.get(key) != value for key, value in job.items()) or result.get("rental_compute_usd") != 0:
        raise ValueError(f"{job['id']} result differs from registration")
    if job.get("profile") in CPU_PROFILES or job.get("probe") == "renderer":
        return
    chunks = [[5]] if len(job["frames"]) == 5 else [[8, 2]]
    if (result["native_depth_batches"] != chunks or result["camera_checkpoint"] != (job["policy"] != "native")
            or result["updates"] != job["steps"] or (job["steps"] and result["lora_tensors"] != 384)):
        raise ValueError(f"{job['id']} used the wrong adaptation configuration")
    if job.get("score"):
        actual = result.get("metrics")
        if not actual or actual["valid_points"] <= 0:
            raise ValueError(f"{job['id']} produced no valid scoring geometry")
        if job["cohort"] == "confirmation":
            expected, = (row for row in plan["expected_confirmation_signatures"]
                         if window_key(row) == window_key(job))
            if (any(result[key] != expected[key] for key in ("input_sha256", "reference_sha256", "raster_shape"))
                    or actual["valid_points"] != expected["valid_points"]):
                raise ValueError(f"{job['id']} used different decoded input bytes")


def timeline(path):
    if not path.is_file():
        return None
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    fields = ("phase", "edge", "updates_done", "allocated", "reserved", "peak_allocated",
              "peak_reserved", "device_free", "host_peak_bytes", "host_rss_bytes", "inventory")
    current_by_phase = defaultdict(dict)
    for row in entries:
        if row["phase"] == "tco_model" or row["updates_done"] not in (0, 1, 2):
            continue
        label = f"{row['updates_done']}:{row['edge']}"
        filtered = {field: row[field] for field in fields if field in row}
        previous_row = current_by_phase[row["phase"]].get(label)
        if not previous_row or filtered.get("allocated", 0) >= previous_row.get("allocated", 0):
            current_by_phase[row["phase"]][label] = filtered
    first_oom = next((row for row in entries if row["edge"] == "oom"), None)
    return dict(events=len(entries), timeline_sha256=sha(path), phase_events=dict(current_by_phase),
                first_oom=first_oom)


def status(job, plan, *, details=False):
    attempts = []
    result = None
    for path in sorted(launch_path(job, "").glob("attempt*")):
        terminal = path / "terminal.json"
        if not terminal.is_file():
            raise RuntimeError(f"An owned launch at {path} lacks a terminal receipt")
        receipt = read(terminal)
        if (receipt["job"], receipt["attempt"]) != (job["id"], path.name):
            raise ValueError(f"Mismatched attempt receipt for {job['id']}")
        out = case_path(job, path.name)
        if receipt["verdict"] == "completed":
            complete = out / "complete.json"
            if result is not None or sha(complete) != receipt["complete_sha256"]:
                raise ValueError(f"Missing, duplicate or changed completion for {job['id']}")
            result = read(complete)
            validated(job, result, plan)
        if details:
            log = path / "stdout.log"
            if receipt["log_sha256"] is not None and sha(log) != receipt["log_sha256"]:
                raise ValueError(f"Log altered for {job['id']}")
            if receipt["verdict"] == "oom":
                errors = log.read_text(encoding="utf-8", errors="replace").splitlines()
                receipt["oom_message"] = next(line.strip() for line in reversed(errors) if "CUDA out of memory" in line)
                receipt["failure_phase"] = next((line.strip() for line in reversed(errors)
                                                 if "in depth_multiview_consistency_energy" in line), None)
            profile = timeline(out / "profile.jsonl")
            if profile:
                receipt["timeline"] = profile
            if receipt.get("error"):
                receipt["error"] = re.sub(r"(?i)[A-Z]:[/\\][^'\"]+", "<local path>", receipt["error"])
        attempts.append(receipt)
    return dict(id=job["id"], verdict="completed" if result is not None else
                attempts[-1]["verdict"] if attempts else "unattempted", attempts=attempts, result=result)


def prerequisite(job):
    for key in ("first_gate", "source_job", "capture_job", "matched_state_from"):
        if key in job:
            return job[key]
    return None


def check_launch_gate(job, plan):
    cohort = job["cohort"]
    if cohort in ("confirmation", "pilot", "fallback") or (cohort in ("screen", "trace") and job["policy"] in ("n", "g", "ng")):
        evidence = PUBLIC / "qualification.json"
        if not evidence.is_file():
            raise ValueError("Complete real-render qualification first")
        qualified = read(evidence)["qualified"]
        if cohort != "confirmation" and not qualified.get(job["policy"], False):
            return False
    if cohort == "confirmation":
        evidence = PUBLIC / "capacity.json"
        if not evidence.is_file() or not read(evidence).get("complete"):
            return False
        panel_plan = read(INITIAL)
        for case in panel_plan["jobs"]:
            if case["cohort"] != "repeat":
                continue
            row = status(case, panel_plan)
            if row["verdict"] not in ("completed", "nonfinite") and len(row["attempts"]) < panel_plan["max_attempts"]:
                raise ValueError("Finish the scheduled native repeatability attempts before confirmation")
    required = prerequisite(job)
    if required:
        pool = plan["jobs"]
        match = next((case for case in pool if case["id"] == required), None)
        if match is None or status(match, plan)["verdict"] != "completed":
            return False
    return True


def export(registration, plan):
    rows = [status(job, plan, details=True) for job in plan["jobs"]]
    path = PUBLIC / ("evidence.json" if registration == "locked" else "initial_evidence.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    write_new(temp, dict(study=STUDY, registration_sha256=sha(REGISTRATIONS[registration], text=True),
                         runs=rows, rental_compute_usd=0))
    temp.replace(path)
    print(f"Exported {len(rows)} registered {registration} case rows")
    return rows


def audit_inside():
    from pathlib import Path
    import tempfile

    import numpy as np
    import torch

    from run_tco_one_percent import DATA_ROOT, load_case, signature

    if torch.cuda.is_available():
        raise RuntimeError("The input audit must run without a GPU")
    plan = read(INITIAL)
    rows = []
    windows = plan["planned_windows"]
    for job in windows:
        with tempfile.TemporaryDirectory(prefix="tco-render-input-audit-") as folder:
            supplied, reference = load_case(job, Path(folder), job["width"])
        arrays = {"rgb": supplied["images"].numpy(), "poses": supplied["extrs"],
                  "intrinsics": supplied["intrs"]}
        surface = {"mask": reference["valid_mask"], "surface": reference["pointclouds"]}
        valid = np.count_nonzero(surface["mask"])
        if (valid <= 0 or any(not np.isfinite(array).all() for array in arrays.values())
                or not np.isfinite(surface["surface"][surface["mask"]]).all()):
            raise ValueError(f"Invalid decoded geometry in {job['scene']}")
        row = {key: job[key] for key in ("dataset", "scene", "frames", "width")}
        rows.append(dict(**row, input_sha256=signature(arrays), reference_sha256=signature(surface),
                         raster_shape=list(arrays["rgb"].shape), valid_points=int(valid),
                         valid_points_by_view=np.count_nonzero(surface["mask"], axis=(1, 2)).tolist()))
        print(f"{job['scene']}: {int(valid)} registered evaluation points", flush=True)
    write_new(PRIVATE / "decoded_inputs.json", dict(study=STUDY, registration_sha256=sha(INITIAL, text=True),
              windows=rows, input_source_sha256=hashes_for(windows, DATA_ROOT), rental_compute_usd=0))


def audit_host():
    import subprocess
    from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped

    verify("initial")
    if (PRIVATE / "decoded_inputs.json").exists():
        raise ValueError("Decoded input receipt already exists")
    config = {"STUDY": STUDY, "CPU_PROFILES": CPU_PROFILES}
    with host_lock(), patch.multiple(previous, **config):
        case = dict(id="input_audit", profile="input_audit")
        argv = create_command(case, "initial", "attempt01", "tco-render-lifetime-input-audit")
        container = subprocess.check_output(argv, text=True, timeout=45).strip()
        try:
            actual, = docker_json(["inspect", container])
            if (actual["Image"] != IMAGE or actual["HostConfig"]["NetworkMode"] != "none"
                    or actual["HostConfig"].get("DeviceRequests")):
                raise ValueError("Input audit has the wrong environment")
            subprocess.run(DOCKER + ["start", "--attach", container], check=True, timeout=1200)
        finally:
            final, errors = ensure_stopped(container)
            if final is None or final["State"]["Running"]:
                raise RuntimeError(f"Unable to verify that the input audit stopped: {errors}")
            subprocess.run(DOCKER + ["rm", container], check=True, timeout=30)


def execute(args, jobs, plan):
    old_create = previous.create_argv

    def argv(job, registration, attempt, name):
        command = old_create(job, registration, attempt, name)
        command[command.index("scripts/run_tco_courtyard_camera.py")] = "scripts/run_tco_render_lifetime.py"
        return command

    config = dict(STUDY=STUDY, CPU_PROFILES=CPU_PROFILES, PRIVATE=PRIVATE, PUBLIC=PUBLIC,
                  validate_complete=validated, case_path=case_path, launch_path=launch_path, create_argv=argv)
    with host_lock(), patch.multiple(previous, **config):
        launched = 0
        for case in jobs:
            if not check_launch_gate(case, plan):
                print(f"{case['id']} gated", flush=True)
                continue
            row = status(case, plan)
            count = len(row["attempts"])
            if row["verdict"] in ("completed", "nonfinite") or count >= plan["max_attempts"] or (count and not args.retry):
                continue
            previous.launch(case, args.registration, plan, f"attempt{count + 1:02d}",
                            args.timeout or plan["launch_timeout_seconds"], args.timeout_reason)
            launched += 1
            if launched == args.case_limit:
                break
        print(f"Launched {launched} new cases", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("registration", choices=REGISTRATIONS, nargs="?", default="initial")
    cli.add_argument("--cohort", action="append")
    cli.add_argument("--job", action="append")
    cli.add_argument("--retry", action="store_true")
    cli.add_argument("--status", action="store_true")
    cli.add_argument("--export", action="store_true")
    cli.add_argument("--case-limit", type=int)
    cli.add_argument("--timeout", type=int)
    cli.add_argument("--timeout-reason")
    cli.add_argument("--audit", action="store_true")
    cli.add_argument("--audit-inside", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.audit or args.audit_inside:
        audit_inside() if args.audit_inside else audit_host()
        return
    if args.timeout and (args.timeout < 1200 or not args.timeout_reason):
        cli.error("A timeout override requires at least 1200 seconds and a documented reason")
    plan = verify(args.registration)
    jobs = [case for case in plan["jobs"] if (not args.cohort or case["cohort"] in args.cohort)
            and (not args.job or case["id"] in args.job)]
    if args.job and set(args.job) != {case["id"] for case in jobs}:
        cli.error("Unknown case ID")
    if args.export:
        export(args.registration, plan)
    if args.status:
        for case in jobs:
            row = status(case, plan)
            print(f"{case['id']:<32} {row['verdict']:<12} {len(row['attempts'])} attempt(s)")
    if args.export or args.status:
        return
    if not args.job and not args.cohort:
        cli.error("Choose explicit cases or cohorts")
    execute(args, jobs, plan)


if __name__ == "__main__":
    main()
