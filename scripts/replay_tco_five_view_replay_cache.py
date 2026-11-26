"""Recompute selection and reconstruction decisions from compact CPU evidence."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
from statistics import median

from run_profiled_d0 import read, sha, write_new
from replay_tco_courtyard_camera import payload_digest


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_five_view_replay_cache_v1"
PUBLIC = ROOT / "results" / STUDY
PHASES = ("initial", "screen", "combination", "development", "locked")
ENDPOINTS = ("accuracy_mean_m", "completeness_mean_m")
MARGIN = 1.01
SEEDS = (7, 77, 777)
RESOURCES = ("scene_seconds", "peak_allocated", "peak_reserved")


def authority(phase):
    stem = STUDY if phase == "initial" else STUDY.removesuffix("_v1") + f"_{phase}_v1"
    return ROOT / "manifests" / f"{stem}.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified(phase):
    registration = authority(phase)
    plan = read(registration)
    ledger = read(PUBLIC / f"{phase}_evidence.json")
    require(ledger["registration_sha256"] == sha(registration, text=True), f"Detached evidence for {phase}")
    require(plan["study"] == ledger["study"] == STUDY and ledger["rental_compute_usd"] == 0,
            f"Wrong study or spend for {phase}")
    require(plan["margin"] == MARGIN and plan["bootstrap_seed"] == 20261126
            and plan["bootstrap_samples"] == 10_000, "Statistical registration differs")
    if phase != "initial":
        previous = PHASES[PHASES.index(phase) - 1]
        require(plan["previous_registration_sha256"] == sha(authority(previous), text=True)
                and plan["initial_sha256"] == sha(authority("initial"), text=True), "Registration ancestry differs")
    for name, digest in plan["source_sha256_lf"].items():
        source = ROOT / name
        if source.is_file() or not name.startswith("third_party/"):
            require(source.is_file() and sha(source, text=True) == digest, f"Source changed: {name}")
    for name, digest in plan.get("authority_sha256_lf", {}).items():
        require(sha(ROOT / name, text=True) == digest, f"Authority changed: {name}")
    jobs, rows = plan["jobs"], ledger["runs"]
    require([job["id"] for job in jobs] == [row["id"] for row in rows], f"Roster differs for {phase}")
    require(len({row["id"] for row in rows}) == len(rows), f"Duplicate IDs for {phase}")
    for job, row in zip(jobs, rows):
        attempts = row["attempts"]
        require(len(attempts) <= plan["max_attempts"], f"Excess attempts for {row['id']}")
        for sequence, attempt in enumerate(attempts, 1):
            require(attempt["job"] == row["id"] and attempt["attempt"] == f"attempt{sequence:02d}"
                    and attempt["rental_compute_usd"] == 0, f"Attempt identity differs for {row['id']}")
        result = row["result"]
        expected = attempts[-1]["verdict"] if attempts else "unattempted"
        require(row["verdict"] == expected and (result is not None) == (expected == "completed"),
                f"Result state differs for {row['id']}")
        if result is None:
            continue
        require(attempts[-1]["stopped"] and attempts[-1]["exit_code"] == 0
                and payload_digest(result) == attempts[-1]["complete_sha256"], f"Changed receipt for {row['id']}")
        require(all(result.get(key) == value for key, value in job.items())
                and result["rental_compute_usd"] == 0, f"Changed arguments for {row['id']}")
        if "runtime" in result:
            require(result["native_depth_batches"] == [[5]] and result["updates"] == job["steps"]
                    and (job["steps"] == 0 or result["lora_tensors"] == 384)
                    and result["head_taps"] == [4, 11, 17, 23], f"Changed TCO workload for {row['id']}")
            runtime = result["runtime"]
            require(runtime["gpu"] == "NVIDIA GeForce RTX 3080" and runtime["gpu_memory_fraction"] == 0.875
                    and runtime["torch"] == "2.7.1+cu128"
                    and runtime["gsplat_sha256"] == "339c0272e544d6a166814cf49e2d21f500eaf9018e69585f1fcaadb5ed3c940e"
                    and 0 < result["peak_allocated"] <= result["peak_reserved"]
                    and result["peak_reserved"] <= result["cuda_device_total"] * 0.875 + 2**20,
                    f"Changed GPU configuration for {row['id']}")
    return plan, rows


def window(job):
    return job["dataset"], job["scene"], tuple(job["frames"]), job["width"]


def finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def resources_by_policy(jobs, rows):
    """Only complete matched windows enter the resource ordering."""
    bucket = defaultdict(dict)
    for job, row in zip(jobs, rows):
        if job["cohort"] != "development":
            continue
        key = (window(job), job["seed"])
        require(key not in bucket[job["policy"]], f"Duplicate development case for {job['id']}")
        bucket[job["policy"]][key] = row["result"]
    native = bucket["native"]
    by_policy = {}
    for policy, cases in sorted(bucket.items()):
        complete = sorted(key for key, result in cases.items() if result is not None)
        record = dict(completed=len(complete), planned=len(cases), eligible=False)
        by_policy[policy] = record
        if set(cases) != set(native) or len(complete) != len(native) or not native or None in native.values():
            continue
        grouped = defaultdict(list)
        for key in complete:
            grouped[key[0]].append((cases[key], native[key]))
        ratios = {}
        for key, pairs in sorted(grouped.items()):
            ratios[key[1]] = {
                metric: median(row[0][metric] for row in pairs) / median(row[1][metric] for row in pairs)
                for metric in RESOURCES
            }
        means = {metric: sum(scene[metric] for scene in ratios.values()) / len(ratios)
                 for metric in RESOURCES}
        sites = cases[complete[0]].get("sites", [])
        record.update(eligible=True, per_scene_ratios=ratios, mean_ratios=means, sites=len(sites),
                      peak_at_most_native=all(scene["peak_allocated"] <= 1 for scene in ratios.values()))
    return by_policy


def choose(resources, qualified):
    candidates = [(name, row) for name, row in resources.items() if name in qualified and row["eligible"]]
    if not candidates:
        return None, "no_complete_valid_candidate"

    def time_key(pair):
        name, row = pair
        ratios = row["mean_ratios"]
        return ratios["scene_seconds"], ratios["peak_allocated"], row["sites"], name

    faster = [pair for pair in candidates if pair[1]["peak_at_most_native"] and time_key(pair)[0] < 1]
    if faster:
        return min(faster, key=time_key)[0], "faster_at_native_peak_or_below"
    memory = [pair for pair in candidates if time_key(pair)[0] <= 1.05]
    if memory:
        return min(memory, key=lambda pair: (time_key(pair)[1], *time_key(pair)))[0], "within_time_budget"
    return min(candidates, key=time_key)[0], "descriptive_tradeoff"


def percentile(values, probability):
    """Linear interpolation using zero-based rank (N - 1) * p."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def grade(estimate, lower, upper, margin=MARGIN):
    if estimate <= margin and upper <= margin:
        return "pass"
    if lower > margin:
        return "degradation"
    return "inconclusive"


def paired_bootstrap(scenes, nested=True, samples=10_000):
    """Resample paired seeds inside each sampled physical scene."""
    rng = random.Random(20261126)

    def ratios(pairs):
        return tuple(median(pair[1][i] for pair in pairs) / median(pair[0][i] for pair in pairs)
                     for i in range(len(ENDPOINTS)))

    observed = [ratios(scene) for scene in scenes]
    by_scene = [[ratios([scene[i] for i in (a, b, c)])
                 for a in range(3) for b in range(3) for c in range(3)] if nested else [true]
                for scene, true in zip(scenes, observed)]
    draws = [[], []]
    for _ in range(samples):
        values = [rng.choice(by_scene[rng.randrange(len(scenes))]) for _ in scenes]
        for i in range(len(ENDPOINTS)):
            draws[i].append(sum(item[i] for item in values) / len(scenes))
    result = {}
    for i, name in enumerate(ENDPOINTS):
        ratio = sum(item[i] for item in observed) / len(scenes)
        low, high = (percentile(draws[i], q) for q in (0.05, 0.95))
        omit = [(sum(row[i] for row in observed) - item[i]) / (len(scenes) - 1)
                for item in observed] if len(scenes) > 1 else []
        result[name] = dict(ratio=ratio, lower=low, upper=high, verdict=grade(ratio, low, high),
                            sensitivity={str(margin): grade(ratio, low, high, margin) for margin in (1.03, 1.05)},
                            scene_ratios=[item[i] for item in observed], leave_one_out=omit)
    return result


def quality(jobs, rows, candidate, planned_windows):
    grouped = defaultdict(dict)
    for job, row in zip(jobs, rows):
        if job["cohort"] != "confirmation":
            continue
        require(job["seed"] in SEEDS and job["policy"] in {"native", candidate}, "Invalid confirmation arm or seed")
        key = (job["seed"], job["policy"])
        require(key not in grouped[window(job)], f"Duplicate quality pair for {job['id']}")
        grouped[window(job)][key] = row["result"]
    report = {}
    for dataset in ("7scenes", "eth3d"):
        expected = sorted(window(item) for item in planned_windows if item["dataset"] == dataset)
        cases = {key: grouped.get(key, {}) for key in expected}
        if any(len(value) != 6 or None in value.values() for value in cases.values()) or not cases:
            report[dataset] = dict(verdict="inconclusive", reason="missing_matched_pairs",
                                    completed=sum(sum(item is not None for item in case.values())
                                                  for case in cases.values()), expected=6 * len(expected))
            continue
        scenes = []
        names = []
        resources = {}
        raw_mm = {}
        for key, by_arm in cases.items():
            names.append(key[1])
            pairs = [[by_arm[seed, policy] for policy in ("native", candidate)] for seed in SEEDS]
            for first, second in pairs:
                require(all(first[field] == second[field] for field in ("input_sha256", "reference_sha256", "raster_shape")),
                        f"Unmatched confirmation bytes for {key}")
            values = [[tuple(row["metrics"][ep] for ep in ENDPOINTS) for row in pair] for pair in pairs]
            require(all(finite_positive(v) for pair in values for arm in pair for v in arm),
                    f"Unusable geometry ratio for {key}")
            scenes.append(values)
            raw_mm[key[1]] = [[[1000 * v for v in arm] for arm in pair] for pair in values]
            resources[key[1]] = {metric: [[row[metric] for row in pair] for pair in pairs] for metric in RESOURCES}
        endpoints = paired_bootstrap(scenes)
        report[dataset] = dict(
            verdict="pass" if all(result["verdict"] == "pass" for result in endpoints.values()) else
                    "degradation" if any(result["verdict"] == "degradation" for result in endpoints.values()) else
                    "inconclusive",
            names=names, seeds=list(SEEDS), endpoints=endpoints, scene_only=paired_bootstrap(scenes, nested=False),
            millimetres=raw_mm, resources=resources,
        )
    return report


def replay():
    if not authority("locked").is_file():
        raise ValueError("The study has not recorded its locked phase")
    ledgers = {phase: verified(phase) for phase in PHASES}
    development, dev_rows = ledgers["development"]
    plan, rows = ledgers["locked"]
    lock = read(PUBLIC / "locked_plan.json")
    resource_table = resources_by_policy(development["jobs"], dev_rows)
    candidate, reason = choose(resource_table, set(lock["qualified"]))
    require(candidate == lock["candidate"] == plan["candidate"] and reason == lock["reason"],
            "Recorded selection differs from resources")
    quality_table = quality(plan["jobs"], rows, candidate, plan["planned_windows"]) if candidate else {}
    latency = {}
    for dataset in ("7scenes", "eth3d"):
        for arm in ("native", candidate):
            if arm is None:
                continue
            subset = [row["result"] for job, row in zip(plan["jobs"], rows)
                      if job["cohort"] == "confirmation" and job["dataset"] == dataset and job["policy"] == arm]
            clean = [item for item in subset if item is not None]
            if clean:
                times = [item["scene_seconds"] for item in clean]
                latency[f"{dataset}/{arm}"] = dict(completed=len(clean), planned=len(subset), median_s=median(times),
                                                  p90_s=percentile(times, 0.9),
                                                  peak_allocated_gib=max(item["peak_allocated"] for item in clean)/2**30,
                                                  peak_reserved_gib=max(item["peak_reserved"] for item in clean)/2**30)
    attempts = Counter(row["verdict"] for _, stage in ledgers.values() for case in stage for row in case["attempts"])
    rescores = []
    scored = {case["id"]: case["result"] for case in rows if case["result"] is not None}
    for job in plan["jobs"]:
        if job.get("probe") != "rescore":
            continue
        result, original = scored.get(job["id"]), scored.get(job["source_job"])
        differences = None
        if result is not None:
            require(original is not None and result["output_sha256"] == original["output_sha256"]
                    and result["reference_sha256"] == original["reference_sha256"], "Rescore provenance differs")
            differences = {key: 1000 * (result["metrics"][key] - original["metrics"][key]) for key in ENDPOINTS}
        rescores.append(dict(id=job["id"], differences_mm=differences,
                             pass_all=differences is not None and max(map(abs, differences.values())) <= 0.01))
    return dict(study=STUDY, candidate=candidate, selection_reason=reason,
                development=resource_table, confirmation=quality_table, confirmation_resources=latency, rescores=rescores,
                joint_1pct_supported=bool(quality_table) and all(row["verdict"] == "pass" for row in quality_table.values()),
                terminal_attempt_counts=dict(sorted(attempts.items())), rental_compute_usd=0)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = replay()
    if args.publish:
        write_new(PUBLIC / "decision.json", result)
    else:
        recorded = PUBLIC / "decision.json"
        if recorded.is_file():
            require(read(recorded) == result, "Published decision differs from CPU replay")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
