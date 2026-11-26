"""Execute registered five-view cases using the existing offline host launcher."""

import argparse
from contextlib import contextmanager
import math
from unittest.mock import patch
from uuid import uuid4

from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock
import run_tco_courtyard_camera_local as host
from tco_five_view_replay_cache_protocol import (
    CPU_PROFILES, IMAGE, PRIVATE, PUBLIC, REGISTRATIONS, STUDY, verify, window_key,
)


def validated(job, result, plan):
    if any(result.get(key) != value for key, value in job.items()) or result["rental_compute_usd"] != 0:
        raise ValueError(f"{job['id']} differs from its registration")
    if job.get("probe") in {"camera_fixture", "camera_fixture_gpu", "rescore"} or job.get("gradient_mode") == "regenerate":
        return
    if (result["native_depth_batches"] != [[5]] or result["updates"] != job["steps"]
            or (job["steps"] and result["lora_tensors"] != 384)):
        raise ValueError(f"{job['id']} changed the registered model configuration")
    if job.get("score"):
        metrics = result["metrics"]
        if metrics["valid_points"] <= 0 or any(
            not math.isfinite(metrics[name]) or metrics[name] < 0
            for name in ("accuracy_mean_m", "completeness_mean_m")
        ):
            raise ValueError(f"{job['id']} did not produce finite scores")
        if job["cohort"] == "confirmation":
            expected, = (row for row in plan["expected_confirmation_signatures"]
                         if window_key(row) == window_key(job))
            if any(result.get(key) != expected[key] for key in ("input_sha256", "reference_sha256", "raster_shape")):
                raise ValueError(f"{job['id']} used different decoded input bytes")
            if metrics["valid_points"] != expected["valid_points"]:
                raise ValueError(f"{job['id']} changed the evaluation mask")


@contextmanager
def launcher():
    original = host.create_argv

    def create(job, registration, attempt, unused_name):
        name = f"tco-five-view-{job['id'].replace('_', '-')}-{attempt}"
        command = original(job, registration, attempt, name)
        command[command.index("scripts/run_tco_courtyard_camera.py")] = "scripts/run_tco_five_view_replay_cache.py"
        return command

    with patch.multiple(host, STUDY=STUDY, CPU_PROFILES=CPU_PROFILES, IMAGE=IMAGE,
                        PRIVATE=PRIVATE, PUBLIC=PUBLIC, create_argv=create, validate_complete=validated):
        yield


def read_evidence(plan):
    with launcher():
        return [host.case_status(job, plan) for job in plan["jobs"]]


def export(registration, plan):
    rows = read_evidence(plan)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    output = PUBLIC / f"{registration}_evidence.json"
    draft = output.with_name(f".{output.stem}.{uuid4().hex}.tmp")
    write_new(draft, dict(study=STUDY, registration_sha256=sha(REGISTRATIONS[registration], text=True),
                         runs=rows, rental_compute_usd=0))
    draft.replace(output)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", choices=REGISTRATIONS)
    parser.add_argument("--cohort", action="append")
    parser.add_argument("--job", action="append")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--activate-deterministic-fallback", action="store_true")
    args = parser.parse_args()
    plan = verify(args.registration)
    jobs = [job for job in plan["jobs"] if (not args.cohort or job["cohort"] in args.cohort)
            and (not args.job or job["id"] in args.job)]
    if (args.job and {job["id"] for job in jobs} != set(args.job)) or not jobs:
        parser.error("The filters must match registered job IDs")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    if args.export:
        export(args.registration, plan)
    if args.status:
        with launcher():
            for job in jobs:
                row = host.case_status(job, plan)
                print(f"{job['id']:<52} {row['verdict']:<12} {len(row['attempts'])} attempt(s)")
    if args.export or args.status:
        return
    if not args.job and not args.cohort:
        parser.error("Select registered cases or cohorts explicitly")
    launched = 0
    with host_lock(), launcher():
        dependencies = {job["id"]: (job, plan) for job in plan["jobs"]}
        if args.registration != "initial":
            initial = read(REGISTRATIONS["initial"])
            dependencies.update({job["id"]: (job, initial) for job in initial["jobs"]})
        for job in jobs:
            if job.get("conditional") and not args.activate_deterministic_fallback:
                continue
            state = host.case_status(job, plan)
            count = len(state["attempts"])
            if state["verdict"] in {"completed", "nonfinite"} or count >= plan["max_attempts"]:
                continue
            if count and not args.retry:
                continue
            parents = [job[field] for field in ("source_job", "q_source_job") if job.get(field)]
            if any(host.case_status(*dependencies[parent])["verdict"] != "completed" for parent in parents):
                print(f"{job['id']} waiting for source cases", flush=True)
                continue
            host.launch(job, args.registration, plan, f"attempt{count + 1:02d}",
                        plan["launch_timeout_seconds"], None)
            launched += 1
            if launched == args.case_limit:
                break
    print(f"Launched {launched} fresh case(s)", flush=True)


if __name__ == "__main__":
    main()
