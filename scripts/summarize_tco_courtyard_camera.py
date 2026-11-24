"""Seal the phase-A diagnosis and describe the registered ten-view gate."""

import argparse
import json
import re

from run_profiled_d0 import read, sha, write_new
from run_tco_courtyard_camera_local import case_path, case_status, launch_path
from tco_courtyard_camera_protocol import INITIAL, PUBLIC, verify


def completed(jobs, plan, name, field):
    row = case_status(jobs[name], plan)
    if row["verdict"] != "completed":
        raise ValueError(f"Required case has not completed: {name}")
    return row["result"][field], row["attempts"][-1]["complete_sha256"]


def diagnosis(jobs, plan):
    rescore, rescore_hash = completed(jobs, plan, "archived_rescore", "archived")
    trace, trace_hash = completed(jobs, plan, "diagnose_courtyard", "diagnosis")
    return dict(archived=rescore, traces=trace, rental_compute_usd=0,
                initial_sha256=sha(INITIAL, text=True),
                receipt_sha256={"archived_rescore": rescore_hash, "diagnose_courtyard": trace_hash})


def profile_summary(path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    maxima = {}
    for event in events:
        maxima[event["phase"]] = max(maxima.get(event["phase"], 0), event["allocated"])
    failed = [event for event in events if event["edge"] == "oom"]
    return dict(events=len(events), timeline_sha256=sha(path),
                first_oom=failed[0] if failed else None, phase_max_allocated=maxima,
                peak_allocated=max(event["peak_allocated"] for event in events),
                peak_reserved=max(event["peak_reserved"] for event in events),
                first_update_camera=next((event for event in events if event["phase"] == "camera_head"
                                          and event["edge"] == "exit" and event["updates_done"] == 0
                                          and event["allocated"] > 5 * 2**30), None))


def gate_case(job, plan):
    evidence = case_status(job, plan)
    attempts = []
    for row in evidence["attempts"]:
        folder = launch_path(job, row["attempt"])
        result = {key: row.get(key) for key in ("attempt", "verdict", "exit_code", "log_sha256", "complete_sha256")}
        if row["verdict"] == "oom":
            text = (folder / "stdout.log").read_text(encoding="utf-8", errors="replace")
            result["oom_message"] = next(line.strip() for line in reversed(text.splitlines())
                                         if re.search("CUDA out of memory", line, flags=re.I))
        profile = case_path(job, row["attempt"]) / "profile.jsonl"
        if profile.is_file() and profile.stat().st_size:
            result["timeline"] = profile_summary(profile)
        attempts.append(result)
    return dict(verdict=evidence["verdict"], attempts=attempts,
                resources=None if evidence["result"] is None else
                {key: evidence["result"][key] for key in ("updates", "scene_seconds", "peak_allocated",
                                                         "peak_reserved", "native_depth_batches")})


def screen(jobs, plan):
    names = ("gate_c8_off", "gate_camera", "profile10_c8_off", "profile10_camera",
             "screen5_c8_off", "screen5_camera", "promote_camera")
    cases = {name: gate_case(jobs[name], plan) for name in names}
    two = cases["gate_camera"]
    full = cases["promote_camera"]
    if two["verdict"] == full["verdict"] == "completed":
        verdict = "promote"
    elif two["verdict"] == "oom" and sum(row["verdict"] == "oom" for row in two["attempts"]) >= 2:
        verdict = "repeatable_two_update_oom"
    elif full["verdict"] == "oom" and sum(row["verdict"] == "oom" for row in full["attempts"]) >= 2:
        verdict = "repeatable_full_length_oom"
    else:
        verdict = "ten_view_fit_unverified"
    if any(cases[name]["verdict"] != "completed" for name in ("screen5_c8_off", "screen5_camera")):
        raise ValueError("Five-view feasibility requires both clean screens")
    return dict(gate=verdict, ten_view_promoted=verdict == "promote", cases=cases,
                initial_sha256=sha(INITIAL, text=True), rental_compute_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("diagnosis", "screen"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    plan = verify("initial")
    jobs = {job["id"]: job for job in plan["jobs"]}
    value = (diagnosis if args.action == "diagnosis" else screen)(jobs, plan)
    if args.publish:
        write_new(PUBLIC / f"{args.action}.json", value)
    else:
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
