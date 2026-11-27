"""Replay the registered five-view geometry and ten-view capacity decisions on CPU."""

import argparse
from collections import Counter, defaultdict
import json
import math
from statistics import median

from replay_tco_courtyard_camera import payload_digest
from replay_tco_five_view_replay_cache import (
    ENDPOINTS, finite_positive, percentile, quality, require, window,
)
from replay_tco_full40_recompute import valid_sha
from run_profiled_d0 import read, sha, write_new
from tco_native_stalls_quality_capacity_protocol import (
    LOCKED, CAPACITY, INITIAL, POLICIES, PUBLIC, ROOT, SELECTION, STUDY, job_table,
)


CAPACITY_SEEDS = {
    "heads/seq-01": (42, 43, 44),
    "office/seq-02": (7, 77, 777),
    "pumpkin/seq-01": (7, 77, 777),
}
TERMINAL = {"completed", "oom", "timeout", "error", "nonfinite", "interrupted"}


def verify_attempts(job, record, max_attempts):
    identifier, status, attempts = record["id"], record["verdict"], record["attempts"]
    require(identifier == job["id"] and len(attempts) <= max_attempts, f"Invalid roster or retries for {identifier}")
    if not attempts:
        require(status in {"unattempted", "gated"} and record["result"] is None,
                f"Missing terminal receipt for {identifier}")
        if status == "gated":
            require(bool(record.get("gate_reason")), f"Missing gate reason for {identifier}")
        return
    require(status == attempts[-1]["verdict"]
            and sum(row["verdict"] == "completed" for row in attempts) == (status == "completed"),
            f"Inconsistent completion records for {identifier}")
    for index, attempt in enumerate(attempts, 1):
        verdict = attempt["verdict"]
        require(attempt["job"] == identifier and attempt["attempt"] == f"attempt{index:02d}"
                and verdict in TERMINAL and attempt["rental_compute_usd"] == 0,
                f"Wrong attempt identity or spend for {identifier}")
        require(finite_positive(attempt["wall_seconds"]) and attempt["outer_seconds"] == 1200,
                f"Invalid timing record for {identifier}")
        proof = attempt.get("complete_sha256")
        if verdict == "completed":
            require(attempt["stopped"] and attempt["exit_code"] == 0 and valid_sha(attempt["log_sha256"])
                    and isinstance(record["result"], dict) and proof == payload_digest(record["result"]),
                    f"Changed completion receipt for {identifier}")
        else:
            require(proof is None, f"Failure claims a completion receipt for {identifier}")
        if verdict == "oom":
            require(attempt["stopped"] and type(attempt["exit_code"]) is int and attempt["exit_code"] != 0
                    and valid_sha(attempt["log_sha256"]) and "TimeoutExpired" not in (attempt["error"] or "")
                    and "CUDA out of memory" in attempt.get("oom_message", ""),
                    f"Missing OOM evidence for {identifier}")
        if verdict == "timeout":
            require(attempt["exit_code"] is None and "TimeoutExpired" in (attempt["error"] or ""),
                    f"Missing deadline evidence for {identifier}")
    require((record["result"] is not None) == (status == "completed"), f"Inconsistent result for {identifier}")


def verify_model_result(job, result):
    identifier = job["id"]
    steps = job["steps"]
    views = len(job["frames"])
    require(views in (5, 10) and result["native_depth_batches"] == ([[5]] if views == 5 else [[8, 2]]),
            f"Wrong depth batches for {identifier}")
    require(result["updates"] == steps and result["lora_tensors"] == (384 if steps else 0)
            and result["head_taps"] == [4, 11, 17, 23] and not result["depth_recompute_spans"]
            and result["camera_checkpoint"] is (job["policy"] != "native"),
            f"Changed optimizer work or model layout for {identifier}")
    optimizer = result["optimizer"]
    require(optimizer["step_calls"] == steps and optimizer["factor_states"] == (384 if steps else 0)
            and optimizer["final_gradients"] == (384 if steps else 0)
            and optimizer["counter_min"] == optimizer["counter_max"] == (steps if steps else None),
            f"Changed Adam work for {identifier}")
    require(len(result["losses"]) == steps and all(type(x) in (int, float) and math.isfinite(x)
                                                   for x in result["losses"]),
            f"Missing or nonfinite losses for {identifier}")
    shape = result["raster_shape"]
    require(len(shape) == 4 and shape[:2] == [views, 3] and shape[-1] == job["width"] and shape[2] > 0,
            f"Changed raster for {identifier}")
    require(result["final_readout"] == dict(views=views, height=shape[2], width=shape[3], geometry_dtype="torch.float32"),
            f"Invalid final reconstruction for {identifier}")
    require(all(valid_sha(result[field]) for field in ("input_sha256", "reference_sha256", "output_sha256")),
            f"Missing data provenance for {identifier}")
    runtime = result["runtime"]
    require(runtime["gpu"] == "NVIDIA GeForce RTX 3080" and runtime["gpu_memory_fraction"] == 0.875
            and runtime["allocator"] == "garbage_collection_threshold:0.75,expandable_segments:True"
            and runtime["torch"] == "2.7.1+cu128" and runtime["cuda"] == "12.8"
            and runtime["gsplat_sha256"] == "339c0272e544d6a166814cf49e2d21f500eaf9018e69585f1fcaadb5ed3c940e",
            f"Changed hardware or runtime for {identifier}")
    require(finite_positive(result["scene_seconds"]) and
            0 < result["peak_allocated"] <= min(result["peak_reserved"], 0.875 * result["cuda_device_total"])
            and result["peak_reserved"] <= result["cuda_device_total"],
            f"Implausible allocator counters for {identifier}")
    if job.get("score"):
        metrics = result["metrics"]
        require(metrics["valid_points"] > 0 and all(finite_positive(metrics[key]) for key in ENDPOINTS),
                f"Unusable geometry for {identifier}")


def verify(plan, ledger, initial_sha, expected_windows=None):
    require(plan["study"] == ledger["study"] == STUDY and ledger["registration_sha256"] == initial_sha
            and ledger["rental_compute_usd"] == 0, "Evidence study, ancestry or spend differs")
    require(plan["margin"] == 1.01 and plan["bootstrap_seed"] == 20261126 and plan["bootstrap_samples"] == 10000,
            "Changed quality protocol")
    require(plan["jobs"] == job_table(plan["planned_windows"]), "Changed prospective case roster")
    for name, digest in plan["source_sha256_lf"].items():
        path = ROOT / name
        if path.is_file() or not name.startswith("third_party/"):
            require(path.is_file() and sha(path, text=True) == digest, f"Changed source {name}")
    jobs = {job["id"]: job for job in plan["jobs"]}
    by_id = {row["id"]: row for row in ledger["runs"]}
    require(jobs.keys() == by_id.keys() and len(jobs) == len(plan["jobs"]) == len(ledger["runs"]),
            "Missing or duplicate case identity")
    signatures = {window(row): row for row in (expected_windows or [])}
    for identifier, job in jobs.items():
        record = by_id[identifier]
        verify_attempts(job, record, plan["max_attempts"])
        result = record["result"]
        if result is None:
            continue
        require(all(result.get(key) == value for key, value in job.items()) and result["rental_compute_usd"] == 0,
                f"Changed job or spend for {identifier}")
        if "runtime" in result:
            verify_model_result(job, result)
        elif job["cohort"] in ("repeat", "confirmation", "screen", "pilot", "capacity", "capacity_control", "small"):
            raise ValueError(f"Missing model runtime for {identifier}")
        if job["cohort"] == "confirmation" and signatures:
            expected = signatures[window(job)]
            require(all(result[name] == expected[name] for name in
                        ("input_sha256", "reference_sha256", "raster_shape"))
                    and result["metrics"]["valid_points"] == expected["valid_points"],
                    f"Changed held-out input or ground truth for {identifier}")
    return jobs, by_id


def verify_rescores(jobs, records):
    rescored = {}
    for identifier, job in jobs.items():
        if job["cohort"] != "rescore":
            continue
        source = job["source_job"]
        result, original = records[identifier]["result"], records[source]["result"]
        differences = None
        if result is not None:
            require(original is not None and result["output_sha256"] == original["output_sha256"]
                    and result["reference_sha256"] == original["reference_sha256"]
                    and result["input_sha256"] == original["input_sha256"]
                    and result["source_completion_sha256"] == records[source]["attempts"][-1]["complete_sha256"]
                    and result["source_prediction_sha256"] == original["artifact_sha256"]["prediction.npz"]
                    and result["metrics"]["valid_points"] == original["metrics"]["valid_points"],
                    f"Changed input geometry for rescore {identifier}")
            differences = {name: 1000 * (result["metrics"][name] - original["metrics"][name])
                           for name in ENDPOINTS}
            require(all(math.isfinite(delta) and abs(delta) <= 0.01 for delta in differences.values()),
                    f"Independent score differs by more than 0.01 mm for {identifier}")
        rescored[identifier] = dict(source_job=source, verdict=records[identifier]["verdict"],
                                    differences_mm=differences)
    return rescored


def summarize_resources(jobs, records, cohorts):
    buckets = defaultdict(list)
    for identifier, job in jobs.items():
        if job["cohort"] in cohorts:
            buckets[(job["dataset"], job["scene"], job["policy"])].append(records[identifier])
    rows = []
    for (dataset, scene, policy), cases in sorted(buckets.items()):
        results = [case["result"] for case in cases if case["result"] is not None]
        elapsed = [item["scene_seconds"] for item in results]
        attempts = Counter(attempt["verdict"] for case in cases for attempt in case["attempts"])
        first = sum(bool(case["attempts"]) and case["attempts"][0]["verdict"] == "completed" for case in cases)
        rows.append(dict(dataset=dataset, scene=scene, policy=policy, planned=len(cases),
                         completed=len(results), first_attempt=first, recovered=len(results)-first,
                         attempts=dict(sorted(attempts.items())),
                         median_scene_s=median(elapsed) if elapsed else None,
                         p90_scene_s=percentile(elapsed, 0.9) if elapsed else None,
                         median_peak_allocated_gib=median(x["peak_allocated"] for x in results)/2**30 if results else None,
                         peak_allocated_gib=max(x["peak_allocated"] for x in results)/2**30 if results else None,
                         peak_reserved_gib=max(x["peak_reserved"] for x in results)/2**30 if results else None))
    return rows


def native_repeatability(records):
    pairs = []
    for scene in ("chess", "courtyard"):
        for seed in (42, 44):
            key = f"repeat_{scene}_{seed}_native"
            parents = [records[f"{key}_{replica}"] for replica in ("a", "b")]
            a, b = (row["result"] for row in parents)
            pair = dict(scene=scene, seed=seed, verdicts=[row["verdict"] for row in parents], errors=None)
            if a is not None and b is not None:
                fields = ("input_sha256", "reference_sha256", "raster_shape")
                require(all(a[field] == b[field] for field in fields), f"Mismatched native replicas for {key}")
                pair["errors"] = {}
                for name in ENDPOINTS:
                    first, second = a["metrics"][name], b["metrics"][name]
                    pair["errors"][name] = dict(replica_a_mm=first*1000, replica_b_mm=second*1000,
                                                absolute_difference_mm=abs(second-first)*1000,
                                                b_over_a=second/first)
            pairs.append(pair)
    return pairs


def clean_completion(record):
    """Separate recovered operational failures from completion mixed with an explicit OOM."""
    return record["result"] is not None and not any(row["verdict"] == "oom" for row in record["attempts"])


def capacity_verdict(jobs, records, qualified, lock=None):
    screen = {}
    for policy in POLICIES:
        rows = [records[f"screen_{policy}_{index}"] for index in (1, 2)]
        screen[policy] = dict(completed=sum(clean_completion(row) for row in rows),
                              slot_verdicts=[row["verdict"] for row in rows])
    candidate = None
    eligible = []
    for policy in SELECTION:
        if not qualified.get(policy) or screen[policy]["completed"] != 2:
            continue
        eligible.append(policy)
        if clean_completion(records[f"pilot_{policy}"]):
            candidate = policy
            break
    selected = [job for job in jobs.values() if job["cohort"] == "capacity" and job["policy"] == candidate]
    if lock is not None:
        require(lock["candidate"] == candidate, "Locked candidate differs from registered resource-only pilot rule")
        locked_ids = lock.get("selected_ids", lock.get("ids"))
        if locked_ids is not None:
            require(set(locked_ids) == {job["id"] for job in selected}, "Locked capacity roster differs")
    completion = []
    if candidate is not None:
        actual = {(job["scene"], job["seed"]) for job in selected}
        require(actual == {(scene, seed) for scene, seeds in CAPACITY_SEEDS.items() for seed in seeds}
                and len(selected) == 9 and all(job["steps"] == 40 for job in selected),
                "Changed nine-run capacity roster")
        completion = [records[job["id"]] for job in selected]
    n = sum(row["result"] is not None for row in completion)
    clean = sum(clean_completion(row) for row in completion)
    first = sum(bool(row["attempts"]) and row["attempts"][0]["verdict"] == "completed" for row in completion)
    verdict = "replicated_capacity" if clean == 9 else "intermittent_fit" if n else "unestablished"
    return dict(candidate=candidate, eligible_pilots=eligible, screen=screen, verdict=verdict,
                planned=len(completion), completed=n, first_attempt=first, recovered=n-first,
                slots={row["id"]: row["verdict"] for row in completion})


def quality_verdict(jobs, records, planned_windows, candidate):
    rows = [(job, records[key]) for key, job in jobs.items() if job["cohort"] == "confirmation"]
    expected = Counter(row["dataset"] for row in planned_windows)
    require(expected == {"7scenes": 7, "eth3d": 8} and len(rows) == 90 and
            {window(job) for job, _ in rows} == {window(row) for row in planned_windows},
            "Changed held-out study roster")
    table = quality([job for job, _ in rows], [row for _, row in rows], "camera", planned_windows)
    return dict(candidate=candidate, datasets=table, joint_1pct_supported=candidate == "camera"
                and all(row["verdict"] == "pass" for row in table.values()),
                missing={job["id"]: row["verdict"] for job, row in rows if row["result"] is None},
                resources=summarize_resources(jobs, records, {"confirmation"}))


def verify_qualification(path, stage, records):
    from qualify_tco_native_stalls_quality_capacity import common_gates

    report = read(path)
    require(report["study"] == STUDY and report["stage"] == stage and report["rental_compute_usd"] == 0,
            f"Invalid {stage} qualification identity")
    require(report["qualified"] == common_gates(report, stage), f"Inconsistent {stage} qualification")
    for identifier, digest in report["receipt_sha256"].items():
        row = records[identifier]
        require(row["verdict"] == "completed" and digest == row["attempts"][-1]["complete_sha256"],
                f"Detached qualification fixture {identifier}")
    return report["qualified"]


def replay(stage):
    path = INITIAL
    plan, ledger = read(path), read(PUBLIC / "evidence.json")
    quality_lock = read(LOCKED)
    planned_windows = quality_lock.get("planned_windows", plan["planned_windows"])
    signatures = quality_lock.get("expected_confirmation_signatures", plan["expected_confirmation_signatures"])
    initial_sha = sha(path, text=True)
    qualification_path = PUBLIC / "quality_qualification.json"
    candidate = quality_lock["candidate"]
    jobs, records = verify(plan, ledger, initial_sha, signatures)
    quality_qualified = verify_qualification(qualification_path, "quality", records)
    locked_ids = [identifier for identifier, job in jobs.items() if job["cohort"] == "confirmation"]
    require(quality_lock["study"] == STUDY and quality_lock["initial_sha256"] == initial_sha and
            quality_lock["qualification_sha256"] == sha(qualification_path, text=True)
            and candidate == ("camera" if quality_qualified["camera"] else None)
            and quality_lock["ids"] == (locked_ids if candidate else []),
            "Five-view lock differs from qualification or registration")
    rescores = verify_rescores(jobs, records)
    if stage == "quality":
        return quality_verdict(jobs, records, planned_windows, candidate)
    qualification_path = PUBLIC / "qualification.json"
    capacity_lock = read(CAPACITY)
    require(capacity_lock["study"] == STUDY and capacity_lock["initial_sha256"] == initial_sha
            and capacity_lock["qualification_sha256"] == sha(qualification_path, text=True),
            "Ten-view lock differs from qualification or registration")
    capacity = capacity_verdict(jobs, records, verify_qualification(qualification_path, "capacity", records),
                                capacity_lock)
    selected_cohorts = {"screen", "pilot", "capacity", "capacity_control"}
    capacity["resources"] = summarize_resources(jobs, records, selected_cohorts)
    score = quality_verdict(jobs, records, planned_windows, candidate)
    require(score == read(PUBLIC / "quality_decision.json"), "Final quality result differs from sealed result")
    attempts = [attempt for record in records.values() for attempt in record["attempts"]]
    return dict(study=STUDY, rental_compute_usd=0, quality=score, capacity=capacity, rescores=rescores,
                native_repeatability=native_repeatability(records),
                development_resources=summarize_resources(jobs, records, {"repeat", "small"}),
                cases={key: dict(verdict=row["verdict"], gate_reason=row.get("gate_reason"))
                       for key, row in records.items()},
                terminal_attempt_counts=dict(sorted(Counter(row["verdict"] for row in attempts).items())),
                total_attempt_wall_seconds=sum(row["wall_seconds"] for row in attempts))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stage", choices=("quality", "full"), default="full")
    cli.add_argument("--publish", action="store_true")
    args = cli.parse_args()
    result = replay(args.stage)
    output = PUBLIC / ("quality_decision.json" if args.stage == "quality" else "decision.json")
    if output.is_file():
        require(read(output) == result, f"{output.name} differs from CPU replay")
    elif args.publish:
        write_new(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
