"""Replay five-view TCO LayerNorm comparisons using registered rooms and paired seeds."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
import random
from statistics import mean, median

from replay_tco_five_view_replay_cache import percentile


METRICS = ("accuracy_mm", "completeness_mm", "scene_seconds", "peak_allocated_bytes", "peak_reserved_bytes")
SEEDS = [42, 43, 44]
DRAWS = 10_000
RNG_SEED = 20261128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grade(summary, margin, strict=False):
    ratio, low, high = (summary[k] for k in ("ratio", "lower_95", "upper_95"))
    def inside(x):
        return x < margin if strict else x <= margin
    if inside(ratio) and inside(high):
        return "pass"
    if not inside(low):
        return "supported_violation"
    if not inside(ratio):
        return "observed_violation_uncertain"
    return "inconclusive"


def scalar_summary(observed, draws):
    ratio = mean(observed.values())
    return dict(ratio=ratio, percent_change=100 * (ratio - 1), lower_95=percentile(draws, .05),
                upper_95=percentile(draws, .95), worst_room=max(observed, key=observed.get),
                leave_one_room_out={k: mean(v for room, v in observed.items() if room != k)
                                    for k in observed} if len(observed) > 1 else {})


def matched_ratios(paired, indices):
    numerator, denominator = paired
    return tuple(median(numerator[j][m] for j in indices) / median(denominator[j][m] for j in indices)
                 for m in range(len(METRICS)))


def compare(rooms, rows, candidate, reference, dataset, draws):
    draws_per_metric = [[] for _ in METRICS]
    medians = {}
    ratios = {}
    environments = defaultdict(list)
    resamples = {}
    triples = list(product(range(3), repeat=3))
    for room, environment in rooms.items():
        sides = [[tuple(rows[(room, arm, seed)][m] for m in METRICS) for seed in SEEDS]
                 for arm in (candidate, reference)]
        medians[room] = {arm: {metric: median(values) for metric, values in zip(METRICS, zip(*side))}
                         for arm, side in zip((candidate, reference), sides)}
        ratios[room] = matched_ratios(sides, (0, 1, 2))
        resamples[room] = [matched_ratios(sides, t) for t in triples]
        environments[environment].append(room)

    # The Stanford environment sizes are fixed. Other cohorts resample entire visits/rooms.
    strata = list(environments.values()) if dataset == "stanford12" else [list(rooms)]
    rng = random.Random(RNG_SEED)
    for _ in range(draws):
        sampled = [rng.choice(resamples[rng.choice(block)]) for block in strata for _ in block]
        for i, values in enumerate(zip(*sampled)):
            draws_per_metric[i].append(math.fsum(values) / len(rooms))
    metrics = {}
    for i, name in enumerate(METRICS):
        rr = {room: values[i] for room, values in ratios.items()}
        metrics[name] = scalar_summary(rr, draws_per_metric[i])
        metrics[name]["environment_ratios"] = {env: mean(rr[room] for room in members)
                                               for env, members in environments.items()}
        metrics[name]["rooms"] = {
            room: {"candidate": m[candidate][name], "reference": m[reference][name],
                   "delta": m[candidate][name] - m[reference][name], "ratio": rr[room]}
            for room, m in medians.items()}
    return metrics


def collect(plan, rows, arms):
    require(plan["dataset"] in {"stanford12", "arkitscenes", "development"}, "Unknown cohort")
    rooms = {r["room"]: r["environment"] for r in plan["rooms"]}
    require(len(rooms) == len(plan["rooms"]) and all(rooms) and all(rooms.values()), "Invalid room roster")
    require(plan["seeds"] == SEEDS, "Expected all three registered paired seeds")
    main, anchors, excluded = {}, {}, Counter()
    for row in rows:
        role = row.get("role", "primary")
        if role == "anchor":
            room = row["room"]
            require(room in rooms and room not in anchors, "Duplicate or unknown anchor")
            require(row["arm"] in {"L0", "VGGT", "ZERO", "zero"}, "Unexpected anchor arm")
            anchors[room] = row
            continue
        if role not in {"primary", "development", "followup", "replication"}:
            require(role in {"duplicate", "engineering"}, "Unknown row role")
            excluded[role] += 1
            continue
        require(plan["dataset"] == "development" or role in {"primary","replication"},
                "Development row entered a confirmation cohort")
        room, arm, seed = key = (row["room"], row["arm"], row["seed"])
        require(room in rooms and arm in arms and seed in SEEDS and key not in main,
                f"Duplicate, unpaired or unregistered row {key}")
        require(row.get("environment", rooms[room]) == rooms[room], "Room environment mismatch")
        if row.get("status", "completed") == "completed":
            for m in METRICS:
                require(isinstance(row.get(m), (float, int)) and math.isfinite(row[m]) and row[m] > 0,
                        f"Invalid {m} for {key}")
        main[key] = row
    return rooms, main, anchors, excluded


def analyze(plan, rows, candidate="LN10", short="L10", full="L40", draws=DRAWS):
    """Analyze already scored/verified rows. CLI uses exactly 10,000 paired resamples."""
    arms = (candidate, short, full)
    require(len(set(arms)) == 3 and draws >= 2, "Invalid comparison")
    rooms, cases, anchors, excluded = collect(plan, rows, arms)
    missing, failures, complete_rooms = [], [], {}
    for room, env in sorted(rooms.items()):
        successful = []
        for key in product([room], arms, SEEDS):
            row = cases.get(key)
            if row is None:
                missing.append(list(key))
            elif row.get("status", "completed") != "completed":
                failures.append(dict(room=room, arm=key[1], seed=key[2], status=row["status"]))
            else:
                successful.append(row)
        # A run cannot quietly change its RGB/priors, or the evaluator's points, within a room.
        for k in ("input_sha256", "reference_sha256"):
            require(len({row.get(k) for row in successful}) <= 1, f"Unmatched {k} for {room}")
        if len(successful) == len(arms) * len(SEEDS):
            complete_rooms[room] = env
    complete = len(complete_rooms) == len(rooms) and bool(rooms)
    formal = complete and len(rooms) >= 6 and plan["dataset"] != "development"
    result = dict(dataset=plan["dataset"], seeds=SEEDS, bootstrap_seed=RNG_SEED, draws=draws,
                  arms=list(arms), expected=len(rooms)*3*3, scored=len(cases)-len(failures),
                  complete_rooms=list(complete_rooms), missing=missing, failures=failures,
                  formal_coverage=formal, excluded_roles=dict(excluded), comparisons={},
                  per_seed_rows=[{k: row[k] for k in ("room","arm","seed",*METRICS) if k in row}
                                 for _,row in sorted(cases.items()) if row.get("status","completed") == "completed"])
    if not complete_rooms:
        result["joint_verdict"] = "insufficient_coverage"
        return result
    for ref in (short, full):
        result["comparisons"][ref] = compare(complete_rooms, cases, candidate, ref, plan["dataset"], draws)
    short_cmp, full_cmp = (result["comparisons"][a] for a in (short,full))
    grades = {
        "accuracy_superiority_vs_short": grade(short_cmp["accuracy_mm"], 1, True),
        "completeness_vs_short": grade(short_cmp["completeness_mm"], 1.03),
        "accuracy_vs_full": grade(full_cmp["accuracy_mm"], 1.01),
        "completeness_vs_full": grade(full_cmp["completeness_mm"], 1.03),
        "faster_vs_full": grade(full_cmp["scene_seconds"], 1, True),
    }
    result["decisions_on_complete_rooms"] = grades
    result["historical_completeness_at_1pct"] = grade(full_cmp["completeness_mm"], 1.01)
    result["completeness_superiority_vs_short"] = grade(short_cmp["completeness_mm"], 1, True)
    result["speedup_vs_full"] = 1/full_cmp["scene_seconds"]["ratio"]
    result["point_gates"] = dict(
        better_accuracy_vs_short=short_cmp["accuracy_mm"]["ratio"] < 1.0,
        accuracy_vs_full=full_cmp["accuracy_mm"]["ratio"] <= 1.01,
        completeness_vs_full=full_cmp["completeness_mm"]["ratio"] <= 1.03,
        faster_vs_full=full_cmp["scene_seconds"]["ratio"] < 1.0)
    if not formal:
        verdict = "development" if plan["dataset"] == "development" and complete else "insufficient_coverage"
    elif all(v == "pass" for v in grades.values()):
        verdict = "pass"
    elif any(v == "supported_violation" for v in grades.values()):
        verdict = "supported_violation"
    else:
        verdict = "inconclusive"
    result["joint_verdict"] = verdict
    result["missing_anchors"] = sorted(room for room in rooms
                                       if room not in anchors or anchors[room].get("status", "completed") != "completed")
    result["retained_gain"] = {}
    for room, row in sorted(anchors.items()):
        if room not in complete_rooms or row.get("status", "completed") != "completed":
            continue
        gain = {}
        for m in ("accuracy_mm", "completeness_mm"):
            require(math.isfinite(row[m]) and row[m] >= 0, "Invalid zero-step anchor")
            scores = full_cmp[m]["rooms"][room]
            vg, ref, cand = row[m], scores["reference"], scores["candidate"]
            gain[m] = dict(vggt=vg, full=ref, candidate=cand,
                           fraction=(vg-cand)/(vg-ref) if vg > ref else None)
        result["retained_gain"][room] = gain
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--candidate", default="LN10")
    p.add_argument("--short", default="L10")
    p.add_argument("--full", default="L40")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    result = analyze(plan, rows["rows"] if isinstance(rows,dict) else rows,
                     args.candidate, args.short, args.full)
    text = json.dumps(result, allow_nan=False, indent=2) + "\n"
    if args.out.exists() and args.out.read_text(encoding="utf-8") != text:
        raise ValueError("Different decision already exists at output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(result["joint_verdict"], result["scored"], "/", result["expected"])


if __name__ == "__main__":
    main()
