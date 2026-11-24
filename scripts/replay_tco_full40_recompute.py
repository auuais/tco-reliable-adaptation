"""Independently replay the full-40 TCO memory and noninferiority decisions on CPU."""

import argparse
from collections import Counter, defaultdict
import json
import math
import random
import re
from pathlib import Path
from statistics import mean, median

from replay_tco_one_percent import paired_bootstrap, quantile
from run_profiled_d0 import read, sha, write_new


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_full40_recompute_v1"
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
LOCKED = ROOT / "manifests/tco_full40_recompute_locked_v1.json"
PUBLIC = ROOT / "results" / STUDY
QUALITY = ("accuracy_mean_m", "completeness_mean_m")
RESOURCES = ("scene_seconds", "peak_allocated", "peak_reserved", "cold_seconds", "host_peak_bytes")
MARGIN = 1.01
TERMINAL = {"completed", "oom", "timeout", "error", "interrupted", "nonfinite"}
TAPS = [4, 11, 17, 23]
LORA_TENSORS = 24 * 2 * 4 * 2


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def number(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def require(condition, note):
    if not condition:
        raise ValueError(note)


def check_jobs(plan):
    jobs = plan["jobs"]
    expect = {}
    seen = set()
    for job in jobs:
        key = job["id"]
        cohort, dataset, scene, seed, policy = (job[name] for name in ("cohort", "dataset", "scene", "seed", "policy"))
        require(cohort in ("ten", "development", "confirmation"), f"Invalid cohort for {key}")
        frames = job["frames"]
        views = 10 if cohort == "ten" else 5
        require(len(frames) == len(set(frames)) == views and job["steps"] == 40, f"Invalid work for {key}")
        block = cohort, dataset, scene, seed, policy
        require(key not in expect and block not in seen, f"Duplicate registered case {key}")
        expect[key] = job
        seen.add(block)
    return expect


def check_attempts(run, job):
    key, verdict = run["id"], run["verdict"]
    attempts = run["attempts"]
    if verdict == "unattempted":
        require(not attempts, f"Unattempted case has launch receipts for {key}")
        return
    require(verdict in TERMINAL and attempts, f"Missing or invalid terminal status for {key}")
    require(attempts[-1]["verdict"] == verdict, f"Terminal status disagrees with attempt for {key}")
    require(sum(item["verdict"] == "completed" for item in attempts) == (verdict == "completed"),
            f"Inconsistent successful attempts for {key}")
    names = [item["attempt"] for item in attempts]
    require(len(names) == len(set(names)), f"Duplicate launch receipt for {key}")
    for attempt in attempts:
        status = attempt["verdict"]
        receipt_key = attempt.get("job", attempt.get("id", key))
        require(status in TERMINAL and receipt_key == job["id"] and attempt["rental_compute_usd"] == 0,
                f"Unexpected launch receipt for {key}")
        if status in ("completed", "oom"):
            require(attempt["stopped"] and valid_sha(attempt["log_sha256"])
                    and type(attempt["exit_code"]) is int
                    and (attempt["exit_code"] == 0) == (status == "completed"),
                    f"Invalid {status} proof for {key}")
        if "wall_seconds" in attempt:
            require(number(attempt["wall_seconds"]), f"Invalid elapsed time for {key}")


def check_result(row, job, scored=True):
    key = job["id"]
    require(isinstance(row, dict) and all(row.get(name) == value for name, value in job.items()),
            f"Completed result differs from registration for {key}")
    steps = job["steps"]
    require(row["updates"] == steps and row["rental_compute_usd"] == 0, f"Wrong work or budget for {key}")
    require(row["lora_tensors"] == (LORA_TENSORS if steps else 0) and row["head_taps"] == TAPS,
            f"Wrong adaptation inventory for {key}")
    dataset = job["dataset"]
    width = job.get("width", 518)
    height = {("7scenes", 518): 392, ("eth3d", 518): 350, ("7scenes", 294): 224}[dataset, width]
    shape = [len(job["frames"]), 3, height, width]
    require(row["raster_shape"] == shape, f"Invalid raster shape for {key}")
    require(all(valid_sha(row[field]) for field in ("input_sha256", "reference_sha256", "output_sha256")),
            f"Invalid evidence hash for {key}")
    require(all(number(row[field]) and row[field] > 0 for field in RESOURCES)
            and row["peak_allocated"] <= row["peak_reserved"], f"Invalid resource evidence for {key}")
    if scored:
        require(all(number(row["metrics"][field]) for field in QUALITY) and row["metrics"]["valid_points"] > 0,
                f"Invalid geometry evidence for {key}")
    require(len(row["losses"]) == steps
            and all(type(loss) in (int, float) and math.isfinite(loss) for loss in row["losses"]),
            f"Missing or nonfinite loss for {key}")


def verify_runs(plan, evidence, initial=False):
    expected = {job["id"]: job for job in plan["jobs"]} if initial else check_jobs(plan)
    require(len(expected) == len(plan["jobs"]), "Duplicate registered case")
    require(evidence["rental_compute_usd"] == 0, "Unexpected rental spend")
    records, rows, anchors = {}, {}, {}
    runtime = None
    signatures = {(item["dataset"], item["scene"]): item for item in plan.get("expected_confirmation_signatures", [])}
    for run in evidence["runs"]:
        key = run["id"]
        require(key in expected and key not in records, f"Unexpected or duplicate run {key}")
        job = expected[key]
        check_attempts(run, job)
        records[key] = run
        row = run["result"]
        if run["verdict"] != "completed":
            require(row is None, f"An unsuccessful case has scientific scores for {key}")
            continue
        if job.get("profile") == "fixture":
            require(all(row.get(field) == item for field, item in job.items())
                    and row["fixture"] and row["rental_compute_usd"] == 0, "Unverified head fixture")
            rows[key] = row
            continue
        check_result(row, job, scored=job.get("score", True))
        require(runtime is None or runtime == row["runtime"], f"Runtime mismatch for {key}")
        runtime = row["runtime"]
        unit = job["dataset"], job["scene"], tuple(job["frames"]), job.get("width", 518)
        identity = row["input_sha256"], row["reference_sha256"]
        if job.get("probe") != "leak":
            require(anchors.setdefault(unit, identity) == identity, f"Input or reference changed for {unit}")
        if job["cohort"] == "confirmation" and signatures:
            allowed = signatures[unit[:2]]
            require(identity == (allowed["input_sha256"], allowed["reference_sha256"])
                    and row["raster_shape"] == allowed["shape"], f"Confirmation input qualification failed for {key}")
        receipt = run["attempts"][-1]
        if "output_sha256" in receipt:
            require(row["output_sha256"] == receipt["output_sha256"], f"Mismatched output receipt for {key}")
        rows[key] = row
    return records, rows


def repeatability(rows):
    by_scene = defaultdict(list)
    for row in rows.values():
        if row["cohort"] == "repeat":
            by_scene[row["scene"]].append(row)
    return {scene: {field: dict(values=[scalar(row, field) for row in pair],
                               absolute_range=max(scalar(row, field) for row in pair)
                               - min(scalar(row, field) for row in pair))
                    for field in QUALITY}
            for scene, pair in sorted(by_scene.items()) if len(pair) == 2}


def summarize_screen(plan, records, rows, cohort):
    jobs = [job for job in plan["jobs"] if job["cohort"] == cohort]
    summary = {}
    ranking = {"on": [], "off": []}
    for job in jobs:
        key, policy = job["id"], job["policy"]
        row = rows.get(key)
        item = {"verdict": records.get(key, {}).get("verdict", "missing")}
        if row is not None:
            seconds = row["mean_step_seconds"]
            require(number(seconds) and seconds > 0, f"Missing synchronized step time for {key}")
            item.update({field: row[field] for field in RESOURCES}, mean_step_seconds=seconds)
            ranking[policy.rsplit("_", 1)[1]].append((seconds, row["peak_allocated"], -int(policy[1:2]), policy))
        summary[policy] = item
    return dict(policies=summary, ranking={name: [value[-1] for value in sorted(entries)]
                                          for name, entries in ranking.items()})


def summarize_resource_repeats(plan, records, rows):
    policies = defaultdict(list)
    for job in plan["jobs"]:
        if job["cohort"] == "resources":
            policies[job["policy"]].append(job["id"])
    result = {}
    for policy, keys in sorted(policies.items()):
        observations = [rows[key] for key in keys if key in rows]
        result[policy] = dict(verdicts={key: records.get(key, {}).get("verdict", "missing") for key in keys},
                              median={name: median(row[name] for row in observations) for name in RESOURCES}
                              if observations else None,
                              range={name: [min(row[name] for row in observations), max(row[name] for row in observations)]
                                     for name in RESOURCES} if observations else None)
    return result


def audit_checks(rows, candidate, reference):
    names = [f"fixed_{policy}" for policy in (candidate, reference)] + [f"leak_{candidate}"]
    if any(name not in rows for name in names):
        return dict(missing=[name for name in names if name not in rows])
    evaluated, control, changed_reference = [rows[name] for name in names]
    candidate_audit = evaluated["audit"]
    control_audit = control["audit"]
    pairs = (evaluated, control, changed_reference)
    return dict(
        reference_isolation=(evaluated["input_sha256"] == changed_reference["input_sha256"]
                             and evaluated["reference_sha256"] != changed_reference["reference_sha256"]
                             and evaluated["output_sha256"] == changed_reference["output_sha256"]),
        initialized_all_factors=(len(candidate_audit["adapter_initial"]) == LORA_TENSORS
                                 and candidate_audit["adapter_initial"] == control_audit["adapter_initial"]),
        matched_render_orders=candidate_audit["render_orders"] == control_audit["render_orders"],
        frozen_weights=all(row["audit"]["frozen_start_sha256"] == row["audit"]["frozen_end_sha256"] for row in pairs),
        all_factor_gradients_present=all(len(row["audit"]["gradients"]) == row["steps"]
                                         and all(len(values) == LORA_TENSORS
                                                 for values in row["audit"]["gradients"].values()) for row in pairs),
        candidate_gradients_match_native_by_hash=candidate_audit["gradients"] == control_audit["gradients"],
    )


def analyze_initial(plan, evidence, candidate, reference):
    records, rows = verify_runs(plan, evidence, initial=True)
    return dict(
        observed=len(rows),
        reference_and_profile_runs={
            job["id"]: records.get(job["id"], {}).get("verdict", "missing")
            for job in plan["jobs"] if job["cohort"] in ("reference", "profile")
        },
        fixture=rows.get("head_fixture", {}).get("fixture"),
        screen_ten=summarize_screen(plan, records, rows, "screen"),
        screen_five=summarize_screen(plan, records, rows, "screen5"),
        resource_repeats=summarize_resource_repeats(plan, records, rows),
        candidate_audit=audit_checks(rows, candidate, reference),
        reference_error_repeatability=repeatability(rows),
    )


def paired_series(jobs, rows, candidate, reference):
    comparison = defaultdict(lambda: defaultdict(dict))
    for job in jobs:
        if job["policy"] in (candidate, reference):
            comparison[job["scene"]][job["policy"]][job["seed"]] = job["id"]
    complete = {}
    missing = []
    for scene, by_policy in sorted(comparison.items()):
        seeds = sorted(set(by_policy[candidate]) | set(by_policy[reference]))
        series = []
        for seed in seeds:
            pair = tuple(by_policy[policy].get(seed) for policy in (candidate, reference))
            if any(key not in rows for key in pair):
                missing.append(dict(scene=scene, seed=seed, candidate=pair[0], reference=pair[1]))
            else:
                series.append(tuple(rows[key] for key in pair))
        if len(series) == len(seeds) == 3:
            complete[scene] = series
        elif len(seeds) != 3:
            missing.append(dict(scene=scene, registered_seed_count=len(seeds)))
    return complete, missing


def scalar(row, field):
    return row["metrics"][field] if field in QUALITY else row[field]


def room_ratio(pairs, field):
    denominator = median(scalar(ref, field) for _, ref in pairs)
    require(denominator > 0, f"Undefined reference denominator for {field}")
    return median(scalar(trial, field) for trial, _ in pairs) / denominator


def seed_and_scene_bootstrap(series, seed, count):
    generator = random.Random(seed)
    names = sorted(series)
    draws = {field: [] for field in QUALITY}
    for _ in range(count):
        scenes = generator.choices(names, k=len(names))
        sampled = [generator.choices(series[name], k=len(series[name])) for name in scenes]
        for field in QUALITY:
            draws[field].append(mean(room_ratio(pairs, field) for pairs in sampled))
    return {f"{side}_{percent:02d}": {field: quantile(values, percent / 100) for field, values in draws.items()}
            for side, percent in (("lower", 5), ("upper", 95))}


def endpoint(point, lower, upper):
    if point <= MARGIN and upper <= MARGIN:
        return "within_margin"
    if lower > MARGIN:
        return "degradation_beyond_margin"
    return "inconclusive_margin"


def pair_summary(pairs):
    return dict(
        ratios={name: room_ratio(pairs, name) for name in (*QUALITY, *RESOURCES)},
        candidate={name: median(scalar(trial, name) for trial, _ in pairs) for name in (*QUALITY, *RESOURCES)},
        reference={name: median(scalar(ref, name) for _, ref in pairs) for name in (*QUALITY, *RESOURCES)},
        seeds=[dict(seed=trial["seed"], candidate=trial["id"], reference=ref["id"]) for trial, ref in pairs],
    )


def analyze_five_view(jobs, rows, candidate, reference, seed, samples, inference):
    series, missing = paired_series(jobs, rows, candidate, reference)
    results = {scene: pair_summary(pairs) for scene, pairs in series.items()}
    report = dict(candidate=candidate, reference=reference, scenes=results, missing_pairs=missing,
                  verdict="inconclusive_missing_cases", endpoints={field: "inconclusive_missing_cases" for field in QUALITY})
    if not results or missing:
        return report
    room_values = {scene: item["ratios"] for scene, item in results.items()}
    ratios = {field: mean(values[field] for values in room_values.values()) for field in (*QUALITY, *RESOURCES)}
    report.update(ratios=ratios, worst_scene={field: max(room_values, key=lambda name: room_values[name][field])
                                             for field in QUALITY},
                  scenes_over_margin={field: sum(item[field] > MARGIN for item in room_values.values())
                                      for field in QUALITY}, point_estimate_gates={field: ratios[field] <= MARGIN
                                                                                   for field in QUALITY})
    if not inference:
        report.update(verdict="development_descriptive")
        return report
    quality_only = {scene: {field: values[field] for field in QUALITY} for scene, values in room_values.items()}
    bounds = paired_bootstrap(quality_only, seed=seed, count=samples)
    endpoints = {field: endpoint(ratios[field], bounds["lower_05"][field], bounds["upper_95"][field])
                 for field in QUALITY}
    states = set(endpoints.values())
    if states == {"within_margin"}:
        verdict = "within_margin"
    elif "degradation_beyond_margin" in states:
        verdict = "degradation_beyond_margin"
    else:
        verdict = "inconclusive_margin"
    report.update(verdict=verdict, endpoints=endpoints, bounds=bounds,
                  paired_scene_and_seed_sensitivity=seed_and_scene_bootstrap(series, seed, samples))
    return report


def reproducible_oom(records, job):
    run = records.get(job["id"])
    return (run is not None and run["verdict"] == "oom"
            and sum(item["verdict"] == "oom" for item in run["attempts"]) >= 2)


def analyze_ten_view(jobs, records, rows, candidate, reference, off_policies):
    by_window = defaultdict(list)
    for job in jobs:
        by_window[job["scene"]].append(job)
    windows = {}
    for scene, registered in sorted(by_window.items()):
        policies = defaultdict(list)
        for job in registered:
            policies[job["policy"]].append(job)
        report = {}
        for policy, policy_jobs in sorted(policies.items()):
            successes = [rows[job["id"]] for job in policy_jobs if job["id"] in rows]
            report[policy] = dict(
                statuses={job["id"]: records.get(job["id"], {}).get("verdict", "missing") for job in policy_jobs},
                completed=len(successes), scheduled=len(policy_jobs),
                reproducible_oom=not successes and any(reproducible_oom(records, job) for job in policy_jobs),
                median={field: median(scalar(row, field) for row in successes)
                        for field in (*QUALITY, *RESOURCES)} if successes else None,
                resource_range={field: [min(scalar(row, field) for row in successes),
                                         max(scalar(row, field) for row in successes)]
                                for field in RESOURCES} if successes else None,
            )
        fitted = report.get(candidate, {}).get("completed", 0) == 3
        off_fit = [policy for policy in off_policies if report.get(policy, {}).get("completed", 0) > 0]
        all_off_oom = all(report.get(policy, {}).get("reproducible_oom", False) for policy in off_policies)
        exact_pairs = {}
        for policy in sorted(off_fit):
            matched, _ = paired_series(registered, rows, candidate, policy)
            if scene in matched:
                exact_pairs[policy] = pair_summary(matched[scene])
        if off_fit:
            attribution = "chunk_only_fits" if reference not in off_fit else "native_reference_fits"
        elif fitted and off_policies and all_off_oom:
            attribution = "recomputation_enabled_fit_among_tested_policies"
        else:
            attribution = "inconclusive_comparator_coverage"
        windows[scene] = dict(complete_candidate_fit=fitted, attribution=attribution, off_policies_with_fit=off_fit,
                              arm_results=report, paired_comparisons=exact_pairs)
    return dict(windows=windows, heads_fit=windows.get("heads/seq-01", {}).get("complete_candidate_fit", False),
                three_window_fit=len(windows) == 3 and all(window["complete_candidate_fit"]
                                                          for window in windows.values()))


def analyze(plan, evidence):
    records, rows = verify_runs(plan, evidence)
    candidate, reference = plan["candidate"], plan["reference"]
    require(candidate != reference, "Candidate must differ from the native reference")
    controls = plan.get("off_policies", [name for name in plan.get("ten_policies", []) if name.endswith("_off")])
    off_policies = sorted({*controls, reference, *([plan["chunk_control"]] if plan.get("chunk_control") else [])})
    sample_seed = plan.get("bootstrap_seed", 20261124)
    samples = plan.get("bootstrap_samples", 10000)
    five = defaultdict(list)
    ten = []
    for job in plan["jobs"]:
        if job["cohort"] == "ten":
            ten.append(job)
        else:
            five[job["cohort"], job["dataset"]].append(job)
    five_results = {"development": {}, "confirmation": {}}
    for (cohort, dataset), jobs in sorted(five.items()):
        infer = cohort == "confirmation"
        summary = analyze_five_view(jobs, rows, candidate, reference, sample_seed, samples, infer)
        other = sorted({job["policy"] for job in jobs} - {candidate, reference})
        summary["other_controls"] = {
            arm: analyze_five_view(jobs, rows, candidate, arm, sample_seed, samples, False) for arm in other
        }
        five_results[cohort][dataset] = summary
    ten_results = analyze_ten_view(ten, records, rows, candidate, reference, off_policies)
    required_datasets = ("7scenes", "eth3d")
    confirmation = five_results["confirmation"]
    quality_states = {dataset: confirmation.get(dataset, {}).get("verdict", "inconclusive_missing_cases")
                      for dataset in required_datasets}
    return dict(
        candidate=candidate, reference=reference, relative_error_margin=MARGIN,
        scheduled=len(plan["jobs"]), terminal=len(records), completed=len(rows),
        outcome_counts=dict(sorted(Counter(run["verdict"] for run in records.values()).items())),
        missing_receipts=[job["id"] for job in plan["jobs"] if job["id"] not in records],
        five_view=five_results, five_view_supported=all(state == "within_margin" for state in quality_states.values()),
        ten_view=ten_results, rental_compute_usd=0,
    )


def replay(lock=LOCKED, evidence=PUBLIC / "evidence.json", decision=PUBLIC / "decision.json", publish=False,
           initial=INITIAL, initial_evidence=PUBLIC / "initial_evidence.json"):
    plan = read(lock)
    observations = read(evidence)
    digest = sha(lock, text=True)
    require(observations["registration_sha256"] == digest, "Evidence refers to a different registration")
    result = dict(registration_sha256=digest, **analyze(plan, observations))
    if plan.get("initial_sha256"):
        require(plan["initial_sha256"] == sha(initial, text=True), "Locked registration refers to a different pilot")
        first_plan, first_observations = read(initial), read(initial_evidence)
        require(first_observations["registration_sha256"] == plan["initial_sha256"],
                "Pilot evidence refers to a different registration")
        result["engineering"] = analyze_initial(first_plan, first_observations, plan["candidate"], plan["reference"])
    if publish and not decision.is_file():
        write_new(decision, result)
    else:
        require(result == read(decision), "Published decision differs from independent replay")
    print(json.dumps(dict(completed=result["completed"], scheduled=result["scheduled"],
                          heads_ten_view=result["ten_view"]["heads_fit"],
                          all_ten_view_windows=result["ten_view"]["three_window_fit"],
                          five_view_supported=result["five_view_supported"]), indent=2))
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--registration", type=Path, default=LOCKED)
    cli.add_argument("--evidence", type=Path, default=PUBLIC / "evidence.json")
    cli.add_argument("--decision", type=Path, default=PUBLIC / "decision.json")
    cli.add_argument("--initial-registration", type=Path, default=INITIAL)
    cli.add_argument("--initial-evidence", type=Path, default=PUBLIC / "initial_evidence.json")
    cli.add_argument("--publish", action="store_true")
    args = cli.parse_args()
    replay(args.registration, args.evidence, args.decision, args.publish,
           args.initial_registration, args.initial_evidence)
