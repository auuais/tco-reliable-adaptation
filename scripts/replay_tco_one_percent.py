"""Replay the TCO resource, quality, mechanism and confirmation decisions on CPU."""

import argparse
from collections import defaultdict
import json
import math
import random
from statistics import mean, median

from run_profiled_d0 import read, sha, write_new
from tco_one_percent_protocol import CONFIRM_MANIFEST, MANIFEST, PUBLIC, TIME_MANIFEST, collect, is_success


QUALITY = ("accuracy_mean_m", "completeness_mean_m")
FIELDS = ("scene_seconds", "peak_allocated", *QUALITY)
ADDITIONAL = ("cold_seconds", "peak_reserved", "host_peak_bytes")
REFERENCE = "four_40"
GEOMETRY = "geometry_10"
PLAIN = "cached12_10"
GEOMETRY_CONTROLS = (PLAIN, "own_10", "shuffle_10")


def arm_key(job):
    return f"{job['arm']}_{job['steps']}"


def scalar(row, field):
    number = row["metrics"][field] if field in QUALITY else row[field]
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Nonfinite or nonpositive {field} in {row['id']}")
    return number


def verify_rows(jobs, evidence):
    expected = {job["id"]: job for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError("Duplicate registered jobs")
    rows = {}
    anchors = {}
    runtime = None
    receipts = {}
    for item in evidence["attempts"]:
        key = item["job"], item["attempt"]
        if key in receipts or item["job"] not in expected or item["rental_compute_usd"] != 0:
            raise ValueError("Unexpected launch receipt")
        receipts[key] = item
    for row in evidence["rows"]:
        key = row["id"]
        if key in rows or key not in expected or any(row[field] != val for field, val in expected[key].items()):
            raise ValueError(f"Unexpected completed job: {key}")
        receipt = receipts.get((key, row["attempt"]))
        if (not receipt or not is_success(receipt) or receipt["wall_seconds"] != row["launch_wall_seconds"]
                or row["updates"] != row["steps"] or row["rental_compute_usd"] != 0):
            raise ValueError(f"Mismatch between case and launch receipt: {key}")
        signature = row["input_sha256"], row["reference_sha256"], row["raster_shape"]
        unit = row["dataset"], row["scene"]
        if anchors.setdefault(unit, signature) != signature:
            raise ValueError(f"Dataset pairing broke on {unit}")
        if runtime is not None and runtime != row["runtime"]:
            raise ValueError("Mismatched runtime within registered evidence")
        runtime = row["runtime"]
        rows[key] = row
    successes = [item["job"] for item in receipts.values() if is_success(item)]
    if sorted(successes) != sorted(rows) or evidence["rental_compute_usd"] != 0:
        raise ValueError("Successful launch receipts and public runs do not correspond")
    return rows


def normalize_scenes(jobs, rows, rooms):
    # Seeds are replicates of a sequence, and sequences belong to physical scenes.
    sequence_jobs = defaultdict(list)
    for job in jobs:
        sequence_jobs[job["scene"], arm_key(job)].append(job)
    medians = defaultdict(dict)
    for (scene, arm), expected in sequence_jobs.items():
        if all(job["id"] in rows for job in expected):
            pairs = [rows[job["id"]] for job in expected]
            medians[arm][scene] = {field: median(scalar(row, field) for row in pairs)
                                  for field in (*FIELDS, *ADDITIONAL)}
    room_sequences = defaultdict(list)
    for sequence, room in rooms.items():
        room_sequences[room].append(sequence)
    summaries = {}
    for arm, by_seq in medians.items():
        references = medians.get(REFERENCE, {})
        per_room = {}
        for room, sequences in room_sequences.items():
            if all(seq in by_seq and seq in references for seq in sequences):
                per_room[room] = {field: mean(by_seq[seq][field] / references[seq][field] for seq in sequences)
                                  for field in FIELDS}
        complete = set(per_room) == set(room_sequences)
        aggregate = ({field: mean(item[field] for item in per_room.values()) for field in FIELDS}
                     if complete else None)
        summaries[arm] = dict(medians_by_sequence=by_seq, ratios_by_room=per_room, ratios=aggregate,
                              worst_room_quality={field: max((result[field] for result in per_room.values()),
                                                              default=None) for field in QUALITY},
                              rooms_over_1percent={field: sum(result[field] > 1.01 for result in per_room.values())
                                                   for field in QUALITY})
    return summaries


def aggregate_errors(ratios):
    return mean(ratios[field] for field in QUALITY)


def passes_quality_challenge(candidate, control):
    return all(candidate[field] <= 1.01 * control[field] for field in QUALITY) and (
        aggregate_errors(candidate) < aggregate_errors(control))


def matched_control(stats, candidate):
    target = stats[candidate]["ratios"]
    eligible = {arm: entry["ratios"] for arm, entry in stats.items()
                if arm.startswith("four_") and entry["ratios"]
                and entry["ratios"]["scene_seconds"] <= target["scene_seconds"]}
    if not eligible and stats.get("zero_0", {}).get("ratios"):
        anchor = stats["zero_0"]["ratios"]
        if anchor["scene_seconds"] <= target["scene_seconds"]:
            eligible["zero_0"] = anchor
    return min(eligible, key=lambda key: (aggregate_errors(eligible[key]), eligible[key]["scene_seconds"], key),
               default=None)


def geometry_test(stats):
    if not all(stats.get(name, {}).get("ratios") for name in (GEOMETRY, *GEOMETRY_CONTROLS)):
        return {"passed": None}
    geometry = stats[GEOMETRY]
    controls = {name: {"passed": passes_quality_challenge(geometry["ratios"], stats[name]["ratios"]),
                       "quality_differences_against_f40": {
                           field: geometry["ratios"][field] - stats[name]["ratios"][field] for field in QUALITY}}
                for name in GEOMETRY_CONTROLS}
    chess = (geometry["ratios_by_room"]["chess"]["completeness_mean_m"]
             < stats[PLAIN]["ratios_by_room"]["chess"]["completeness_mean_m"])
    heads = all(geometry["ratios_by_room"]["heads"][field] < 1 for field in QUALITY)
    return dict(passed=chess and heads and all(test["passed"] for test in controls.values()),
                chess_completeness_improved=chess, heads_below_full40=heads, controls=controls)


def analyze_development(plan, qualification, evidence, time_plan=None):
    clear = qualification["last14_screen"]["eligible"]
    jobs = [job for job in plan["jobs"] if clear or job["arm"] != "cached14"]
    if time_plan:
        jobs.extend(job for job in time_plan["jobs"] if job["phase"] == "equal_time")
    observed = verify_rows(jobs, evidence)
    incomplete = [job["id"] for job in jobs if job["id"] not in observed]
    stats = normalize_scenes(jobs, observed, plan["physical_rooms"])
    mechanism = geometry_test(stats)
    decisions = {}
    arms = sorted({arm_key(job) for job in jobs if job["arm"] not in ("four", "zero", "shuffle")})
    thresholds = plan["thresholds"]
    timing_locked = (time_plan or {}).get("locked_by_candidate", {})
    for arm in arms:
        ratios = stats.get(arm, {}).get("ratios")
        gates = ({field: ratios[field] <= thresholds[field] for field in FIELDS} if ratios else None)
        control = matched_control(stats, arm) if ratios else None
        timed = timing_locked.get(arm)
        comparison_ready = timed is not None and (timed == 0 or bool(stats.get(f"four_{timed}", {}).get("ratios")))
        head_to_head = passes_quality_challenge(ratios, stats[control]["ratios"]) if control else None
        suitable = gates and all(gates.values()) and (arm != GEOMETRY or mechanism["passed"])
        advances = bool(suitable and comparison_ready and head_to_head)
        decisions[arm] = dict(gates=gates, qualifies_for_timing=bool(suitable), matched_control=control,
                              matched_quality_passed=head_to_head, time_matched_budget_locked=timed,
                              advances=advances)
    eligible = [arm for arm in arms if decisions[arm]["advances"]]
    verdict = "inconclusive_missing_development" if incomplete else "no_candidate"
    candidate = None
    if not incomplete:
        if any(d["qualifies_for_timing"] and d["time_matched_budget_locked"] is None for d in decisions.values()):
            verdict = "needs_equal_time"
        elif eligible:
            feasible = sorted(eligible, key=lambda arm: (stats[arm]["ratios"]["scene_seconds"],
                                                         stats[arm]["ratios"]["peak_allocated"],
                                                         int(arm in (GEOMETRY, "own_10")), arm))
            candidate = feasible[0]
            verdict = "ready_for_confirmation"
        elif any(d["qualifies_for_timing"] and (d["matched_control"] is None or
                 d["matched_control"] == "zero_0" and d["time_matched_budget_locked"] != 0)
                 for d in decisions.values()):
            verdict = "inconclusive_time_match"
    return dict(verdict=verdict, nominated=candidate, complete=len(observed), scheduled=len(jobs), pending=incomplete,
                last14_screen=qualification["last14_screen"], comparisons=stats,
                candidates=decisions, geometry_test=mechanism)


def quantile(values, fraction):
    values = sorted(values)
    rank = (len(values) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return (1 - weight) * values[lower] + weight * values[upper]


def paired_bootstrap(room_values, seed, count):
    rooms = sorted(room_values)
    generator = random.Random(seed)
    names = tuple(room_values[rooms[0]])
    distribution = {metric: [] for metric in names}
    for _ in range(count):
        sample = generator.choices(rooms, k=len(rooms))
        for metric in names:
            distribution[metric].append(mean(room_values[room][metric] for room in sample))
    return dict(lower_05={name: quantile(values, .05) for name, values in distribution.items()},
                upper_95={name: quantile(values, .95) for name, values in distribution.items()},
                leave_one_out={room: {name: mean(room_values[other][name] for other in rooms if other != room)
                                      for name in names} for room in rooms} if len(rooms) > 1 else {})


def analyze_confirmation(plan, locked, evidence):
    rows = verify_rows(locked["jobs"], evidence)
    candidate = locked["candidate"]
    results = {}
    for dataset, rooms in locked["physical_rooms"].items():
        jobs = [job for job in locked["jobs"] if job["dataset"] == dataset]
        missing = [job["id"] for job in jobs if job["id"] not in rows]
        stats = normalize_scenes(jobs, rows, rooms)
        output = dict(missing=missing, comparisons=stats, verdict="inconclusive_missing_cases")
        if not missing:
            study = stats[candidate]
            bootstrap = dict(seed=plan["confirmation"]["bootstrap_seed"],
                             count=plan["confirmation"]["bootstrap_samples"])
            uncertainty = paired_bootstrap(
                {room: {field: values[field] for field in QUALITY} for room, values in study["ratios_by_room"].items()},
                **bootstrap)
            thresholds = plan["thresholds"]
            gates = {field: study["ratios"][field] <= thresholds[field] for field in FIELDS}
            upper_pass = {field: uncertainty["upper_95"][field] <= thresholds[field] for field in QUALITY}
            control = matched_control(stats, candidate)
            head_to_head = bool(control and passes_quality_challenge(study["ratios"], stats[control]["ratios"]))
            verdict = "within_margin" if all(gates.values()) and all(upper_pass.values()) and head_to_head else (
                "inconclusive_margin" if all(gates.values()) and head_to_head else "missed_confirmation_targets")
            output.update(quality_uncertainty=uncertainty, point_estimate_gates=gates, margin_gates=upper_pass,
                          matched_control=control, matched_quality_passed=head_to_head, verdict=verdict)
            if candidate == GEOMETRY:
                geometry = {}
                for name in GEOMETRY_CONTROLS:
                    differences = {room: {field: values[field] - stats[name]["ratios_by_room"][room][field]
                                          for field in QUALITY} for room, values in study["ratios_by_room"].items()}
                    geometry[name] = dict(room_differences=differences,
                                          composite_uncertainty=paired_bootstrap(
                                              {room: dict(composite=aggregate_errors(values))
                                               for room, values in differences.items()}, **bootstrap))
                output["geometry_differences"] = geometry
        results[dataset] = output
    return results


def publish_or_replay(publish=False, preview=False):
    plan = read(MANIFEST)
    qualification = read(PUBLIC / "qualification.json")
    if not qualification["qualified"] or qualification["development_sha256"] != sha(MANIFEST, text=True):
        raise ValueError("Qualification does not match the development protocol")
    time_plan = read(TIME_MANIFEST) if TIME_MANIFEST.is_file() else None
    jobs = [job for job in plan["jobs"] if job["arm"] != "cached14" or qualification["last14_screen"]["eligible"]]
    if time_plan:
        jobs.extend(job for job in time_plan["jobs"] if job["phase"] == "equal_time")
    evidence = collect(jobs) if publish or preview else read(PUBLIC / "evidence.json")
    report = analyze_development(plan, qualification, evidence, time_plan)
    if publish:
        write_new(PUBLIC / "evidence.json", evidence)
        write_new(PUBLIC / "decision.json", report)
    elif not preview and report != read(PUBLIC / "decision.json"):
        raise ValueError("Published development decision does not replay")
    confirmation = None
    if CONFIRM_MANIFEST.is_file() and not preview:
        locked = read(CONFIRM_MANIFEST)
        if (report["nominated"] != locked["candidate"] or locked["development_sha256"] != sha(MANIFEST, text=True)):
            raise ValueError("The locked recipe differs from the development result")
        evidence = collect(locked["jobs"]) if publish else read(PUBLIC / "confirmation_evidence.json")
        confirmation = analyze_confirmation(plan, locked, evidence)
        if publish:
            write_new(PUBLIC / "confirmation_evidence.json", evidence)
            write_new(PUBLIC / "confirmation_decision.json", confirmation)
        elif confirmation != read(PUBLIC / "confirmation_decision.json"):
            raise ValueError("Published confirmation decision does not replay")
    print(json.dumps(dict(development=report["verdict"], candidate=report["nominated"],
                          complete=report["complete"], scheduled=report["scheduled"],
                          timing_candidates=[key for key, val in report["candidates"].items()
                                             if val["qualifies_for_timing"]],
                          geometry=report["geometry_test"]["passed"],
                          confirmation={key: value["verdict"] for key, value in (confirmation or {}).items()}),
                     indent=2))
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group()
    action.add_argument("--publish", action="store_true")
    action.add_argument("--preview", action="store_true")
    args = cli.parse_args()
    publish_or_replay(args.publish, args.preview)
