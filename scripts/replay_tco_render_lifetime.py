"""Recompute local TCO render-lifetime findings from registered CPU evidence."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, median

from replay_tco_courtyard_camera import payload_digest
from replay_tco_full40_recompute import (
    MARGIN, QUALITY, RESOURCES, analyze_five_view, check_attempts, check_result,
    number, pair_summary, paired_series, require, scalar, valid_sha,
)
from run_profiled_d0 import read, sha, write_new
from tco_full40_recompute_protocol import runtime_pins


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_render_lifetime_v1"
PUBLIC = ROOT / "results" / STUDY
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
LOCKED = ROOT / "manifests/tco_render_lifetime_locked_v1.json"
POLICIES = ("native", "camera", "n", "g", "ng", "ngd", "camera_d")
PROMOTION_ORDER = ("n", "g", "ng", "ngd")
CAMERA = "camera"
NATIVE = "native"
SEEDS = (7, 77, 777)
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_SAMPLES = 10_000
GSPLAT = "339c0272e544d6a166814cf49e2d21f500eaf9018e69585f1fcaadb5ed3c940e"
TEN_SEEDS = {"heads/seq-01": (42, 43, 44), "office/seq-02": SEEDS, "pumpkin/seq-01": SEEDS}
FIVE_SEVEN = dict.fromkeys(("chess/seq-05", "fire/seq-04", "office/seq-06", "pumpkin/seq-07",
                           "redkitchen/seq-04"), (124, 200))
FIVE_SEVEN.update({"heads/seq-01": (150, 200), "stairs/seq-04": (74, 100)})
FIVE_ETH = {name: (start, 5) for name, start in (
    ("delivery_area", 1), ("electro", 2), ("facade", 27), ("kicker", 2),
    ("office", 2), ("playground", 1), ("terrains", 3),
)}


def window_key(job):
    return job["dataset"], job["scene"], tuple(job["frames"]), job["width"]


def verify_cases(plan, evidence, digest):
    """Check closed receipts and provenance independently of the launcher."""
    require(plan["study"] == evidence["study"] == STUDY and evidence["registration_sha256"] == digest
            and evidence["rental_compute_usd"] == 0, "Detached evidence or rental spend")
    require(all(plan.get(name) == value for name, value in runtime_pins().items())
            and plan["margin"] == MARGIN and plan["bootstrap_seed"] == BOOTSTRAP_SEED
            and plan["bootstrap_samples"] == BOOTSTRAP_SAMPLES and plan["max_attempts"] == 2,
            "The registered runtime or statistical protocol differs")
    jobs = {job["id"]: job for job in plan["jobs"]}
    records = {row["id"]: row for row in evidence["runs"]}
    require(len(jobs) == len(plan["jobs"]) and len(records) == len(evidence["runs"])
            and set(jobs) == set(records), "Evidence roster differs")
    require(all(valid_sha(item) for item in plan["input_source_sha256"].values()), "Invalid input digest")
    for name, expected in plan["source_sha256_lf"].items():
        path = ROOT / name
        if name.startswith("third_party/") and not path.is_file():
            continue
        require(path.is_file() and sha(path, text=True) == expected, f"Pinned study source differs: {name}")
    registered = plan.get("expected_confirmation_signatures", [])
    signatures = {window_key(item): item for item in registered}
    require(len(signatures) == len(registered), "Duplicate confirmation input signature")
    confirmations = {window_key(job) for job in jobs.values() if job["cohort"] == "confirmation"}
    if plan.get("registration") == "locked":
        require(set(signatures) == confirmations, "Confirmation decoded signature roster differs")

    rows, seen, effective_sources = {}, {}, {}
    for key, job in jobs.items():
        record = records[key]
        check_attempts(record, job)
        attempts = record["attempts"]
        require(0 <= len(attempts) <= 2 and [a["attempt"] for a in attempts] ==
                [f"attempt{i + 1:02d}" for i in range(len(attempts))], f"Invalid retry sequence: {key}")
        for entry in attempts:
            if entry["verdict"] == "oom":
                require(entry["complete_sha256"] is None and "cuda out of memory" in
                        entry.get("oom_message", "").lower(), f"Missing explicit CUDA OOM: {key}")
        if record["verdict"] != "completed":
            require(record["result"] is None, f"Failed case published a result: {key}")
            continue
        row = record["result"]
        require(row is not None and payload_digest(row) == attempts[-1]["complete_sha256"]
                and all(row.get(field) == value for field, value in job.items())
                and row["rental_compute_usd"] == 0, f"Completion receipt differs: {key}")
        rows[key] = row
        if "runtime" not in row:
            require(job["cohort"] in ("qualification", "rescore", "input_audit"), f"GPU evidence missing for {key}")
            continue
        check_result(row, job, scored=job["score"])
        runtime = row["runtime"]
        require(all(runtime[name] == plan[name] for name in ("gpu", "gpu_memory_fraction", "allocator"))
                and runtime["torch"] == "2.7.1+cu128" and runtime["cuda"] == "12.8"
                and runtime["gsplat_sha256"] == GSPLAT
                and row["peak_allocated"] <= row["cuda_device_total"] * plan["gpu_memory_fraction"],
                f"Runtime or memory cap differs: {key}")
        check_depth_batches(job, row)
        hashes = runtime.get("objective_callable_sha256"), runtime.get("forward_callable_sha256")
        role = (job["policy"] in {"n", "ng", "ngd"}, job["policy"] in {"g", "ng", "ngd"},
                job.get("profile") == "timeline")
        require(all(valid_sha(value) for value in hashes) and effective_sources.setdefault(role, hashes) == hashes
                and all(valid_sha(value) for value in row["artifact_sha256"].values())
                and (not job["score"] or valid_sha(row["artifact_sha256"].get("prediction.npz"))),
                f"Effective source or artifact digests missing: {key}")
        unit = window_key(job)
        decoded = row["input_sha256"], row["reference_sha256"], row["raster_shape"]
        if job.get("probe") != "leak":
            require(seen.setdefault(unit, decoded) == decoded, f"Mismatched input pair: {key}")
        if unit in signatures:
            signature = signatures[unit]
            require(decoded == tuple(signature[name] for name in ("input_sha256", "reference_sha256", "raster_shape"))
                    and row["metrics"]["valid_points"] == signature["valid_points"], f"Decoded inputs differ: {key}")
    for key, job in jobs.items():
        if job.get("first_gate") and records[key]["verdict"] != "unattempted":
            require(records[job["first_gate"]]["verdict"] == "completed", f"Native seed gate bypassed: {key}")
    return records, rows


def check_depth_batches(job, row):
    views = len(job["frames"])
    expected = [[8, 2]] if views == 10 else [[5]]
    require(row["native_depth_batches"] == expected and row["camera_checkpoint"] == (job["policy"] != NATIVE),
            f"Depth batches or camera assignment differ: {job['id']}")
    recompute = []
    if job["policy"] in {"ngd", "camera_d"} and job.get("steps"):
        recompute = ["[0:8]", "[8:10]"] if views == 10 else ["[0:5]"]
    require(row.get("depth_recompute_spans", []) == recompute, f"Depth recomputation differs: {job['id']}")


def classify(record, attempts_required=2):
    """A fit with an OOM history is unstable. A timeout stays operational."""
    attempts = Counter(item["verdict"] for item in record["attempts"])
    if record["verdict"] == "completed":
        return "unstable" if attempts["oom"] else "completed"
    if attempts["oom"] >= attempts_required:
        return "repeatable_oom"
    if attempts["nonfinite"]:
        return "nonfinite"
    if len(record["attempts"]) < attempts_required:
        return "pending"
    return "inconclusive"


def is_qualified(item):
    """A fixed-adjoint pass alone cannot certify a change to the real loss."""
    return bool(item and item.get("qualified") and item.get("real_renderer_passed")
                and item.get("real_update_passed"))


def arm_screen(jobs, records, policy):
    selected = [job for job in jobs if job["policy"] == policy and job["cohort"] in ("screen", "fallback")
                and job["id"].startswith("screen_") and job["steps"] == 2]
    require(len(selected) == 2, f"Expected two independent gate repeats for {policy}")
    items = [records[job["id"]] for job in selected]
    case_states = {job["id"]: classify(records[job["id"]]) for job in selected}
    oom_count = sum(entry["verdict"] == "oom" for item in items for entry in item["attempts"])
    completions = sum(item["verdict"] == "completed" for item in items)
    if completions and oom_count:
        state = "unstable"
    elif completions == 2:
        state = "completed"
    elif oom_count >= 2:
        state = "repeatable_oom"
    elif "nonfinite" in case_states.values():
        state = "nonfinite"
    elif "pending" in case_states.values():
        state = "pending"
    else:
        state = "inconclusive"
    return dict(state=state, completions=completions, oom_attempts=oom_count, cases=case_states)


def derive_screen(plan, records, qualified):
    """Execute the ordered resource gates without using reconstruction scores."""
    results = {}
    promoted = None
    wait_for_earlier = False
    for policy in PROMOTION_ORDER:
        pilot = records[f"pilot_{policy}"]
        screen = arm_screen(plan["jobs"], records, policy)
        promotion = classify(pilot)
        allowed = is_qualified(qualified.get(policy))
        if policy == "ngd":
            allowed = allowed and all(is_qualified(qualified.get(key)) for key in ("n", "g", "ng"))
        result = dict(qualified=allowed, screen=screen, pilot=promotion)
        results[policy] = result
        if promoted or wait_for_earlier or not allowed:
            require(not pilot["attempts"], f"Pilot bypassed ordered qualification: {policy}")
            result["gate"] = "ineligible"
            continue
        if screen["state"] != "completed":
            require(not pilot["attempts"], f"Pilot bypassed two-update screen: {policy}")
            result["gate"] = screen["state"]
            wait_for_earlier = screen["state"] == "pending"
            continue
        result["gate"] = promotion
        if promotion == "completed":
            promoted = policy
        elif promotion == "pending":
            wait_for_earlier = True
    return dict(promoted_policy=promoted, arms=results, pending=wait_for_earlier,
                selection_final=not wait_for_earlier)


def validate_locked_roster(plan):
    candidate = plan["candidate"]
    required = set()
    if candidate is not None:
        require(candidate in PROMOTION_ORDER, "Unregistered candidate")
        required.update(("ten", "7scenes", scene, tuple(range(0, 1000, 100)), policy, seed)
                        for scene, seeds in TEN_SEEDS.items()
                        for policy in (candidate, CAMERA, NATIVE) for seed in seeds)
        for dataset, windows in (("7scenes", FIVE_SEVEN), ("eth3d", FIVE_ETH)):
            required.update(("confirmation", dataset, scene, tuple(range(start, start + 5 * stride, stride)),
                             policy, seed) for scene, (start, stride) in windows.items()
                            for policy in (candidate, NATIVE) for seed in SEEDS)
    jobs = [job for job in plan["jobs"] if job["cohort"] in ("ten", "confirmation")]
    actual = {(job["cohort"], *window_key(job)[:3], job["policy"], job["seed"]) for job in jobs}
    require(actual == required and len(jobs) == len(actual)
            and all(job["steps"] == 40 and job["width"] == 518 and job["score"] and not job["profile"]
                    for job in jobs), "Locked roster differs from the study handoff")


def complete_row(row, record):
    return row is not None and classify(record) == "completed"


def analyze_capacity(plan, records, rows):
    """Require three independent successful seeds in each registered window."""
    candidate = plan["candidate"]
    stable_rows = {key: row for key, row in rows.items() if complete_row(row, records[key])}
    windows = {}
    for scene, seeds in TEN_SEEDS.items():
        jobs = [job for job in plan["jobs"] if job["cohort"] == "ten" and job["scene"] == scene]
        if not jobs:
            continue
        arms = {}
        for policy in (candidate, NATIVE, CAMERA):
            scheduled = [job for job in jobs if job["policy"] == policy]
            successes = [stable_rows[job["id"]] for job in scheduled if job["id"] in stable_rows]
            first = next(job for job in scheduled if job["seed"] == seeds[0])
            compared = None
            if policy != candidate:
                paired, _ = paired_series(jobs, stable_rows, candidate, policy)
                if scene in paired:
                    compared = pair_summary(paired[scene])
            arms[policy] = dict(
                completed=len(successes), scheduled=len(scheduled),
                repeatable_oom=classify(records[first["id"]]) == "repeatable_oom",
                cases={job["id"]: classify(records[job["id"]]) for job in scheduled},
                median={field: median(scalar(row, field) for row in successes)
                        for field in (*QUALITY, *RESOURCES)} if successes else None,
                paired=compared,
            )
        native = arms[NATIVE]
        ratios = native["paired"]["ratios"] if native["paired"] else None
        resource_advantage = bool(ratios and ratios["peak_allocated"] <= 1 and ratios["peak_reserved"] <= 1
                                  and min(ratios["peak_allocated"], ratios["peak_reserved"]) < 1)
        windows[scene] = dict(arms=arms, native_comparator_supported=native["repeatable_oom"] or resource_advantage)
    fitted = sum(row["arms"][candidate]["completed"] for row in windows.values())
    supported = len(windows) == len(TEN_SEEDS) and all(row["native_comparator_supported"] for row in windows.values())
    return dict(promoted_policy=candidate, windows=windows, valid_completions=fitted, scheduled_completions=9,
                complete=fitted == 9, native_comparator_supported=supported)


def analyze_repeatability(plan, rows):
    groups = defaultdict(lambda: defaultdict(dict))
    rescores = {}
    for job in plan["jobs"]:
        if job["cohort"] == "repeat":
            window = ("eth3d", (0, 5, 10, 15, 20)) if job["scene"] == "courtyard" else (
                "7scenes", (0, 200, 400, 600, 800))
            require((job["dataset"], tuple(job["frames"])) == window and job["policy"] == NATIVE
                    and job["steps"] == 40 and not job["profile"], "Repeat roster changed conditions")
            groups[job["scene"]][job["seed"]][job["replica"]] = job["id"]
        elif job["cohort"] == "rescore" and job["id"] in rows:
            source = rows[job["source_job"]]
            row = rows[job["id"]]
            require(row["reference_sha256"] == source["reference_sha256"]
                    and row["output_sha256"] == source["output_sha256"]
                    and row["source_prediction_sha256"] == source["artifact_sha256"]["prediction.npz"]
                    and row["original_metrics"] == source["metrics"], "Rescore used different saved outputs")
            metrics = row["metrics"]
            original = source["metrics"]
            require(metrics["valid_points"] == original["valid_points"], "Rescore mask size changed")
            rescores[source["id"]] = dict(id=row["id"], delta_mm={field: 1000 * (metrics[field] - original[field])
                                                                 for field in QUALITY})
            rescores[source["id"]]["within_tolerance"] = all(abs(value) <= 0.01
                                                               for value in rescores[source["id"]]["delta_mm"].values())
    require(set(groups) == {"chess/seq-03", "courtyard"} and
            all(set(registered) == set(SEEDS) for registered in groups.values()), "Native repeat roster differs")
    result = {}
    for scene, registered in sorted(groups.items()):
        require(all(set(replicas) == {1, 2} for replicas in registered.values()), f"Wrong repeat roster in {scene}")
        pairs = {seed: (rows.get(registered[seed][1]), rows.get(registered[seed][2])) for seed in SEEDS}
        records = {}
        bounds = {}
        for field in QUALITY:
            values = {}
            for seed, (a, b) in pairs.items():
                if a is None or b is None:
                    continue
                first, second = scalar(a, field), scalar(b, field)
                require(first > 0 and second > 0, f"Cannot compare a nonpositive native score in {scene}")
                values[seed] = (first, second)
            records[field] = {str(seed): dict(first_mm=1000 * a, second_mm=1000 * b,
                                              signed_delta_mm=1000 * (b - a), abs_delta_mm=1000 * abs(b - a),
                                              signed_relative=b / a - 1, absolute_log_ratio=abs(math.log(b / a)))
                              for seed, (a, b) in values.items()}
            if len(values) == len(SEEDS):
                swaps = []
                for swap in product((0, 1), repeat=len(SEEDS)):
                    first = median(values[seed][switch] for seed, switch in zip(SEEDS, swap))
                    second = median(values[seed][1 - switch] for seed, switch in zip(SEEDS, swap))
                    swaps.append(dict(swap=list(swap), second_to_first=second / first))
                bounds[field] = dict(median_label_swaps=swaps,
                                     native_spread_crosses_1pct=any(x["second_to_first"] > MARGIN for x in swaps))
        observed = {key: rescores.get(key) for replicas in registered.values() for key in replicas.values()}
        result[scene] = dict(pair_coverage=len(bounds) == len(QUALITY), endpoints=records, label_sensitivity=bounds,
                             output_hashes={str(seed): [row["output_sha256"] if row else None for row in pair]
                                            for seed, pair in pairs.items()},
                             rescores=observed, scorer_repeatability_supported=all(
                                 value is not None and value["within_tolerance"] for value in observed.values()))
    return result


def analyze_five(plan, records, rows, capacity):
    """Keep dataset-level distance decisions pending until local capacity qualifies."""
    jobs = [job for job in plan["jobs"] if job["cohort"] == "confirmation"]
    if not capacity["complete"]:
        require(all(records[job["id"]]["verdict"] == "unattempted" for job in jobs),
                "Opened confirmation despite the capacity gate")
        return {}
    candidate = plan["candidate"]
    answer = {}
    timed_rows = {}
    for job in jobs:
        row = rows.get(job["id"])
        if row is not None:
            timed_rows[job["id"]] = dict(row, wall_seconds=records[job["id"]]["attempts"][-1]["wall_seconds"])
    for dataset in ("7scenes", "eth3d"):
        selected = [job for job in jobs if job["dataset"] == dataset]
        base = analyze_five_view(selected, timed_rows, candidate, NATIVE,
                                 BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES, inference=True)
        paired, _ = paired_series(selected, timed_rows, candidate, NATIVE)
        base["per_seed"] = {
            scene: [dict(seed=a["seed"], valid_points=b["metrics"]["valid_points"],
                         candidate_mm={name: 1000 * scalar(a, name) for name in QUALITY},
                         native_mm={name: 1000 * scalar(b, name) for name in QUALITY},
                         candidate_wall_seconds=a["wall_seconds"], native_wall_seconds=b["wall_seconds"])
                    for a, b in pairs]
            for scene, pairs in paired.items()
        }
        rooms = base["scenes"]
        if len(rooms) > 1:
            base["leave_one_scene_out"] = {
                removed: {name: mean(row["ratios"][name] for scene, row in rooms.items() if scene != removed)
                          for name in QUALITY}
                for removed in rooms
            }
        answer[dataset] = base
    return answer


def timeline_changes(plan, records):
    """Expose compact phase traces for interpretation without forcing a causal claim."""
    profiles = {}
    for job in plan["jobs"]:
        if job.get("profile") != "timeline":
            continue
        key = job["id"]
        profiles[key] = [
            {name: item[name] for name in ("attempt", "verdict", "oom_message", "timeline") if name in item}
            for item in records[key]["attempts"]
        ]
    return profiles


def tensor_passed(row, rtol):
    if row.get("invalid_values"):
        expected = False
    elif "present" in row:
        expected = row["present"] == row["expected_present"] is False
    else:
        expected = number(row["l2"]) and number(row["ref_l2"]) and row["l2"] <= 1e-6 + rtol * row["ref_l2"]
    require(row["passed"] is expected, "Reported renderer tensor tolerance result differs")
    return expected


def render_comparison(row):
    tolerances = {"values": 1e-5, "mv_gradients": 1e-4, "total_gradients": 1e-4, "outputs": 1e-5}
    require(all(row[name] for name in tolerances), "Missing renderer tensor fields")
    fields = {name: [tensor_passed(item, rtol) for item in row[name].values()]
              for name, rtol in tolerances.items()}
    expected = row["exact_discrete"] and 0 < row["counts"]["visible"] <= row["counts"]["total"] and all(
        all(items) for items in fields.values())
    require(row["passed"] is expected, "Reported renderer comparison differs")
    return expected, fields


def renderer_verdict(receipt, captured):
    real = receipt["real"]
    require([item["step"] for item in real] == [0, 1]
            and [item["capture_sha256"] for item in real] ==
            [captured[name] for name in ("renderer_inputs.pt", "renderer_inputs_1.pt")],
            "Renderer replay used different loss input captures")
    policies = {"native": "camera", "camera": "camera", "n": "n", "g": "g", "ng": "ng",
                "ngd": "ng", "camera_d": "camera"}
    evidence = []
    for section in (receipt["synthetic"], *real):
        repeats = [render_comparison(item)[0] for item in section["native_repeats"]]
        negative, fields = render_comparison(section["negative"])
        detected = (section["negative"]["exact_discrete"] and all(fields["values"])
                    and any(not item for item in fields["mv_gradients"]))
        require(len(repeats) == 2 and section["native_repeats_pass"] == all(repeats)
                and section["negative_detected"] == detected and (not detected or not negative),
                "Renderer native repeatability or detached-alpha negative control differs")
        evidence.append((all(repeats) and detected,
                         {policy: render_comparison(item)[0] for policy, item in section["policies"].items()}))
    passed = {name: all(valid and effects[key] for valid, effects in evidence) for name, key in policies.items()}
    require(receipt["qualified"] == passed, "Renderer qualification differs from measured values")
    return passed


def factor_checks(report, families):
    fields = report["factors"]["fields"]
    expected = {f"{family}/{step}" for family in families for step in (1, 2)}
    require(set(fields) == expected and all(item["count"] == 384 for item in fields.values()),
            "Missing factor comparison values")
    passes = {}
    for key, item in fields.items():
        bound = 0 if key.startswith("adam_step/") else 1e-4
        passed = number(item["max_relative_l2"]) and item["max_relative_l2"] <= bound
        require(item["passed"] == passed, "Reported factor tolerance result differs")
        passes[key] = passed
    grouped = {str(step): all(value for key, value in passes.items() if key.endswith(f"/{step}"))
               for step in (1, 2)}
    require({key: item["passed"] for key, item in report["factors"]["steps"].items()} == grouped,
            "Reported update tolerance differs")
    return grouped


def verify_qualification(qualification, records, rows, digest):
    require(qualification["rental_compute_usd"] == 0, "Invalid qualification budget")
    reader = ROOT / "scripts/qualify_tco_render_lifetime.py"
    require(qualification["qualification_reader_sha256"] == sha(reader, text=True),
            "Qualification reader differs from the recorded gradient index correction")
    declared = qualification["qualified"]
    policies = qualification["policies"]
    require(set(declared) == set(policies), "Qualification policy inventory differs")
    receipts = qualification["jobs"]
    for name, file_sha in receipts.items():
        require(name in records, f"Unregistered qualification result: {name}")
        if file_sha is None:
            require(name not in rows, f"Qualification omitted a successful result: {name}")
        else:
            require(name in rows and records[name]["attempts"][-1]["complete_sha256"] == file_sha,
                    f"Unbound qualification result: {name}")
    if "initial_sha256" in qualification:
        require(qualification["initial_sha256"] == digest, "Detached qualification receipt")
    renderer = rows.get("renderer_replay", {}).get("renderer", {})
    rendering = (renderer_verdict(renderer, rows["renderer_capture"]["artifact_sha256"])
                 if renderer else {})
    require(rendering == qualification["fixtures"]["renderer"]["qualified"], "Qualification fixtures changed")
    for policy, info in policies.items():
        checks = {name: value for name, value in info.items() if isinstance(value, dict) and "passed" in value}
        failed = {name for name, value in checks.items() if not value["passed"]}
        require(type(declared[policy]) is bool and declared[policy] == info["qualified"] == (not failed)
                and set(info["reasons"]) == failed,
                f"Qualification disagrees for {policy}")
        renderer_pass = rendering.get(policy, False)
        update_pass = info["real"]["passed"] and info["matched"]["passed"]
        if declared[policy]:
            common = ("grad", "preclip", "parameter")
            real = (*common, "exp_avg", "exp_avg_sq", "adam_step")
            fixed_steps = factor_checks(info["fixed"], common)
            free_steps = factor_checks(info["real"], real)
            matched_steps = factor_checks(info["matched"], real)
            native_steps = factor_checks(info["native_repeat"], real)
            require("cpu_fixture" in receipts and "renderer_replay" in receipts
                    and rows["cpu_fixture"]["fixture"]["passed"]
                    and renderer_pass and update_pass and f"matched_{policy}" in receipts
                    and "matched_native" in receipts and f"real_{policy}" in receipts
                    and all(fixed_steps.values()) and all(matched_steps.values()) and free_steps["1"] and native_steps["1"]
                    and info["fixed"]["forward_exact"] and info["fixed"]["predictions"]["max_relative_l2"] <= 1e-5
                    and info["matched"]["matched_restore_valid"] and all(info["matched"]["forward_exact_by_update"])
                    and info["real"]["forward_exact_by_update"][0] and info["native_repeat"]["forward_exact_by_update"][0],
                    f"Real-renderer or matched optimizer proof missing for {policy}")
        yield policy, dict(qualified=declared[policy], real_renderer_passed=renderer_pass, real_update_passed=update_pass)


def initial_report():
    plan = read(INITIAL)
    digest = sha(INITIAL, text=True)
    records, rows = verify_cases(plan, read(PUBLIC / "initial_evidence.json"), digest)
    qualification = read(PUBLIC / "qualification.json")
    qualified = dict(verify_qualification(qualification, records, rows, digest))
    selection = derive_screen(plan, records, qualified)
    report = dict(study=STUDY, initial_sha256=digest, qualification_sha256=sha(PUBLIC / "qualification.json", text=True),
                  **selection, profiles=timeline_changes(plan, records),
                  base_controls={name: records[name]["verdict"] for name in ("screen_native_1", "screen_camera_1",
                                                                             "screen_camera_2")},
                  rental_compute_usd=0)
    return plan, records, rows, report


def locked_report(screen):
    plan = read(LOCKED)
    digest = sha(LOCKED, text=True)
    require(plan["initial_sha256"] == screen["initial_sha256"] and plan["candidate"] == screen["promoted_policy"]
            and screen["selection_final"], "The capacity lock differs from selection")
    authorities = plan["authority_sha256_lf"]
    require(set(authorities) >= {"screen.json", "qualification.json"}
            and all(sha(PUBLIC / name, text=True) == digest for name, digest in authorities.items()),
            "The lock's authority files changed")
    validate_locked_roster(plan)
    records, rows = verify_cases(plan, read(PUBLIC / "evidence.json"), digest)
    value = analyze_capacity(plan, records, rows)
    return plan, records, rows, dict(study=STUDY, lock_sha256=digest, **value, rental_compute_usd=0)


def verify_or_publish(name, value, publish):
    path = PUBLIC / f"{name}.json"
    if publish and not path.exists():
        write_new(path, value)
    else:
        require(read(path) == value, f"Published {name} differs from independent replay")


def replay(publish=False, stage="decision"):
    initial, initial_records, initial_rows, screen = initial_report()
    if stage == "screen":
        verify_or_publish(stage, screen, publish)
        return screen
    verify_or_publish("screen", screen, False)
    locked, locked_records, locked_rows, capacity = locked_report(screen)
    if stage == "capacity":
        verify_or_publish(stage, capacity, publish)
        return capacity
    verify_or_publish("capacity", capacity, False)
    quality = analyze_five(locked, locked_records, locked_rows, capacity)
    both_datasets = set(quality) == {"7scenes", "eth3d"}
    quality_supported = both_datasets and all(item["verdict"] == "within_margin" for item in quality.values())
    seed_sensitivity_passed = quality_supported and all(
        all(ratio <= MARGIN for ratio in item["paired_scene_and_seed_sensitivity"]["upper_95"].values())
        for item in quality.values())
    decision = dict(study=STUDY, initial_sha256=screen["initial_sha256"], lock_sha256=capacity["lock_sha256"],
                    promoted_policy=screen["promoted_policy"], capacity=capacity, quality=quality,
                    quality_supported=quality_supported, quality_seed_sensitivity_passed=seed_sensitivity_passed,
                    repeatability=analyze_repeatability(initial, initial_rows), rental_compute_usd=0,
                    further_gpu_evidence_gate=capacity["complete"] and capacity["native_comparator_supported"]
                    and seed_sensitivity_passed)
    all_records = [*initial_records.values(), *locked_records.values()]
    decision["attempt_counts"] = dict(sorted(Counter(item["verdict"]
                                                     for row in all_records for item in row["attempts"]).items()))
    decision["case_counts"] = dict(sorted(Counter(row["verdict"] for row in all_records).items()))
    verify_or_publish(stage, decision, publish)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "capacity", "decision"), nargs="?", default="decision")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(replay(args.publish, args.stage), indent=2))


if __name__ == "__main__":
    main()
