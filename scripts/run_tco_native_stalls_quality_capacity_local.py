"""Run the registered TCO study serially on the local RTX 3080."""

import argparse
from contextlib import contextmanager, nullcontext
from itertools import groupby
import math
from uuid import uuid4
from unittest.mock import patch

import run_tco_courtyard_camera_local as host
from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock
from tco_native_stalls_quality_capacity_protocol import (
    CAPACITY, CPU_PROFILES, IMAGE, INITIAL, LOCKED, PRIVATE, PUBLIC, SELECTION, STUDY,
    receipt_correction, verify,
)


def validated(job, result, plan):
    if any(result.get(key) != value for key, value in job.items()) or result["rental_compute_usd"] != 0:
        raise ValueError("Result differs from the registered case")
    if job["profile"] in CPU_PROFILES or job.get("probe") == "renderer":
        return
    policy = job["policy"]
    views = len(job["frames"])
    runtime = result["runtime"]
    settings = dict(torch=plan["torch"], cuda="12.8", gpu=plan["gpu"],
                    gsplat_sha256=plan["gsplat_sha256"], allocator=plan["allocator"],
                    gpu_memory_fraction=plan["gpu_memory_fraction"])
    if any(runtime.get(key) != value for key, value in settings.items()):
        raise ValueError("Case ran with different runtime pins")
    if not 0 < result["peak_allocated"] <= result["cuda_device_total"] * plan["gpu_memory_fraction"]:
        raise ValueError("Case has invalid peak allocated memory")
    expected_batches = [[5]] if views == 5 else [[8, 2]]
    if (result["native_depth_batches"] != expected_batches or result["camera_checkpoint"] != (policy != "native")
            or result["updates"] != job["steps"] or (job["steps"] and result["lora_tensors"] != 384)
            or result["head_taps"] != [4, 11, 17, 23]):
        raise ValueError("Case used non-native depth grouping, wrong update count or wrong LoRA configuration")
    if job["score"]:
        metrics = result["metrics"]
        if metrics["valid_points"] <= 0 or any(not math.isfinite(metrics[key]) or metrics[key] <= 0
                                               for key in ("accuracy_mean_m", "completeness_mean_m")):
            raise ValueError("Nonfinite or unusable scoring geometry")
    if job["cohort"] == "confirmation":
        expected, = (row for row in plan["expected_confirmation_signatures"]
                     if all(row[key] == job[key] for key in ("dataset", "scene", "frames", "width")))
        if (any(result[key] != expected[key] for key in ("input_sha256", "reference_sha256", "raster_shape"))
                or result["metrics"]["valid_points"] != expected["valid_points"]):
            raise ValueError("Confirmation input or scoring geometry differs from registration")


@contextmanager
def launcher():
    original = host.create_argv

    def create(job, registration, attempt, unused_name):
        name = f"tco-native-stalls-{job['id'].replace('_', '-')}-{attempt}"
        argv = original(job, registration, attempt, name)
        argv[argv.index("scripts/run_tco_courtyard_camera.py")] = "scripts/run_tco_native_stalls_quality_capacity.py"
        return argv

    with patch.multiple(host, STUDY=STUDY, CPU_PROFILES=CPU_PROFILES, PRIVATE=PRIVATE, PUBLIC=PUBLIC,
                        IMAGE=IMAGE, create_argv=create, validate_complete=validated):
        yield


def telemetry_path(job, attempt):
    return PRIVATE / "host_telemetry" / job["id"] / f"{attempt}.jsonl"
def state(job, plan, *, details=False):
    row = host.case_status(job, plan)
    if details:
        from tco_native_stall_trace import summarize_trace
        for attempt in row["attempts"]:
            folder = host.launch_path(job, attempt["attempt"])
            logfile = folder / "stdout.log"
            if attempt["log_sha256"] is not None and sha(logfile) != attempt["log_sha256"]:
                raise ValueError(f"Unexpected log hash in {job['id']}")
            if attempt["verdict"] == "oom":
                log = logfile.read_text(encoding="utf-8", errors="replace").splitlines()
                attempt["oom_message"] = next(line.strip() for line in reversed(log) if "CUDA out of memory" in line)
            if job["profile"] == "timeline":
                directory = host.case_path(job, attempt["attempt"])
                attempt["trace"] = summarize_trace(directory)
                telemetry = telemetry_path(job, attempt["attempt"])
                if telemetry.is_file():
                    attempt["host_telemetry_sha256"] = sha(telemetry)
    gate = PRIVATE / "gates" / f"{job['id']}.json"
    if not row["attempts"] and gate.is_file():
        row.update(verdict="gated", gate_reason=read(gate)["reason"])
    return row


def gate(job, reason):
    path = PRIVATE / "gates" / f"{job['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if read(path)["reason"] != reason:
            raise ValueError(f"Gate for {job['id']} already recorded")
    else:
        write_new(path, dict(reason=reason))


def publish_evidence(plan):
    PUBLIC.mkdir(parents=True, exist_ok=True)
    target = PUBLIC / "evidence.json"
    draft = target.with_name(f".evidence-{uuid4().hex}.tmp")
    rows = [state(job, plan, details=True) for job in plan["jobs"]]
    write_new(draft, dict(study=STUDY, registration_sha256=sha(INITIAL, text=True), runs=rows,
                          rental_compute_usd=0))
    draft.replace(target)


def run(job, plan, retry=True):
    row = state(job, plan)
    if row["verdict"] in {"completed", "gated", "nonfinite"}:
        return row
    limit = receipt_correction(plan).get("retry_limits", {}).get(job["id"], 2) if retry else 1
    while len(row["attempts"]) < limit:
        attempt = f"attempt{len(row['attempts']) + 1:02d}"
        trace = nullcontext()
        if job["profile"] == "timeline":
            from tco_native_stall_trace import host_telemetry
            trace = host_telemetry(telemetry_path(job, attempt))
        with trace:
            host.launch(job, "initial", plan, attempt, plan["launch_timeout_seconds"], None)
        row = state(job, plan)
        if row["verdict"] in {"completed", "nonfinite"}:
            break
    return row


def cohort(plan, name):
    return [job for job in plan["jobs"] if job["cohort"] == name]


def rescore_jobs(plan, original_cohort):
    originals = {row["id"]: row for row in cohort(plan, original_cohort)}
    for job in cohort(plan, "rescore"):
        if job["source_job"] not in originals:
            continue
        if state(originals[job["source_job"]], plan)["verdict"] == "completed":
            run(job, plan)
        else:
            gate(job, "source_has_no_valid_reconstruction")


def development(plan):
    for job in cohort(plan, "repeat"):
        run(job, plan)
    traces = {job["id"]: job for job in cohort(plan, "trace")}
    for key, job in traces.items():
        if not job.get("conditional"):
            run(job, plan, retry=False)
    chess = state(traces["trace_chess_native"], plan)
    heads = traces["trace_heads_native"]
    if chess["verdict"] == "completed":
        run(heads, plan, retry=False)
    else:
        gate(heads, "chess_native_trace_did_not_complete")
    focused = traces["trace_chess_focused"]
    if chess["verdict"] == "timeout":
        run(focused, plan, retry=False)
    else:
        gate(focused, "no_timed_out_chess_native_trace")
    native = [state(job, plan) for job in cohort(plan, "repeat")
              if job["policy"] == "native" and job["scene"].startswith("chess/")]
    for job in cohort(plan, "small"):
        if sum(row["verdict"] == "timeout" for row in native) >= 2:
            run(job, plan)
        else:
            gate(job, "full_raster_native_timeout_did_not_repeat")
    rescore_jobs(plan, "repeat")
def make_lock(path, candidate, ids, qualification):
    payload = dict(study=STUDY, initial_sha256=sha(INITIAL, text=True), candidate=candidate,
                   ids=ids, qualification_sha256=sha(PUBLIC / qualification, text=True))
    if path.is_file():
        if read(path) != payload:
            raise ValueError(f"Existing lock differs from {path.name}")
    else:
        write_new(path, payload)


def confirm_quality(plan):
    jobs = cohort(plan, "confirmation")
    fixture = read(PUBLIC / "quality_qualification.json")
    reason = "camera_qualification_unresolved"
    if not fixture["qualified"]["camera"]:
        make_lock(LOCKED, None, [], "quality_qualification.json")
        for job in jobs:
            gate(job, reason)
        rescore_jobs(plan, "confirmation")
        return

    make_lock(LOCKED, "camera", [job["id"] for job in jobs], "quality_qualification.json")
    for dataset, selection in groupby(jobs, key=lambda row: row["dataset"]):
        stop_reason = None
        for pair_id, cases in groupby(selection, key=lambda row: row["pair_id"]):
            pair = list(cases)
            if stop_reason:
                for job in pair:
                    gate(job, stop_reason)
                continue
            rows = [run(job, plan) for job in pair]
            if any(row["verdict"] != "completed" for row in rows):
                stop_reason = f"{dataset}_missing_pair_{pair_id}"
        publish_evidence(plan)
    rescore_jobs(plan, "confirmation")


def capacity(plan):
    if not (PUBLIC / "quality_decision.json").is_file():
        raise ValueError("Publish the five-view quality decision before the capacity stage")
    qualification = read(PUBLIC / "qualification.json")["qualified"]
    screen = cohort(plan, "screen")
    for job in [*screen, *cohort(plan, "capacity_trace")]:
        if job["policy"] not in {"native"} and not qualification.get(job["policy"], False):
            gate(job, "policy_not_qualified")
        else:
            run(job, plan, retry=False)

    promoted = None
    for job in cohort(plan, "pilot"):
        policy = job["policy"]
        completed = [state(case, plan)["verdict"] == "completed" for case in screen if case["policy"] == policy]
        if promoted or not all(completed):
            gate(job, "earlier_policy_promoted" if promoted else "screen_did_not_finish_twice")
            continue
        row = run(job, plan)
        if row["verdict"] == "completed" and all(item["verdict"] != "oom" for item in row["attempts"]):
            promoted = policy
    cases = [job for job in plan["jobs"] if job["cohort"] in {"capacity", "capacity_control"}]
    ids = [job["id"] for job in cases if job["policy"] == promoted]
    make_lock(CAPACITY, promoted, ids, "qualification.json")
    first = {(job["scene"], job["policy"]): job for job in cases if job["seed"] == job["first_seed"]}
    for job in cases:
        policy = job["policy"]
        if not promoted or policy not in {promoted, "camera", "native"}:
            gate(job, "no_eligible_pilot" if not promoted else "not_locked_policy_or_control")
            continue
        if policy != promoted and job["seed"] != job["first_seed"]:
            control = first[job["scene"], policy]
            if state(control, plan)["verdict"] != "completed":
                gate(job, "first_seed_control_incomplete")
                continue
        run(job, plan)


def run_group(plan, name):
    by_id = {job["id"]: job for job in plan["jobs"]}
    for job in cohort(plan, name):
        source = job.get("source_job")
        if source and state(by_id[source], plan)["verdict"] != "completed":
            gate(job, "source_has_no_valid_completion")
        else:
            run(job, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "quality_fixture", "confirmation",
                                         "capacity_fixture", "capacity", "export", "status"))
    args = parser.parse_args()
    plan = verify()
    with host_lock(), launcher():
        actions = {"development": development, "confirmation": confirm_quality, "capacity": capacity}
        if args.stage in actions:
            actions[args.stage](plan)
        elif args.stage.endswith("_fixture"):
            run_group(plan, args.stage)
        elif args.stage == "status":
            for job in plan["jobs"]:
                row = state(job, plan)
                if row["attempts"] or row["verdict"] == "gated":
                    print(job["id"], row["verdict"], len(row["attempts"]))
            return
        publish_evidence(plan)


if __name__ == "__main__":
    main()
