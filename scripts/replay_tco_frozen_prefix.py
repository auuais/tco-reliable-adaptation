"""Recompute the TCO benchmark decision from the committed measurement rows."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_frozen_prefix_v1"
RESULTS = ROOT/"results"/STUDY
QUALITY = ("accuracy_mean_m", "completeness_mean_m")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equivalent(actual, expected):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], expected[k]) for k in actual)
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)
    return actual == expected


def verify(plan, evidence, published):
    scheduled = {j["id"]: j for j in plan["jobs"]}
    measured = {row["id"]: row for row in evidence["rows"]}
    if (len(evidence["rows"]) != len(measured) or scheduled.keys() != measured.keys() or
            evidence["rental_compute_usd"] != 0 or plan["rental_compute_usd"] != 0):
        raise ValueError("Wrong registered job inventory or spend")
    layouts = defaultdict(set)
    per_scene = defaultdict(list)
    for tag, result in measured.items():
        if (result.get("rental_compute_usd") != 0 or result["metrics"] is None or
                any(result[k] != v for k, v in scheduled[tag].items())):
            raise ValueError(f"Mismatched run identity {tag}")
        layouts[result["scene"]].add(result["input_sha256"])
        per_scene[result["scene"], result["arm"], result["steps"]].append(result)
    if len(layouts) != 3 or any(len(hashes) != 1 for hashes in layouts.values()):
        raise ValueError("Scene identities or matched masks differ")
    fields = tuple(plan["thresholds"])
    medians = {}
    for keys, rows in per_scene.items():
        if sorted(r["seed"] for r in rows) != [42, 43, 44]:
            raise ValueError("Wrong paired-seed inventory")
        measures = {f: sorted(r["metrics"][f] if f in QUALITY else r[f] for r in rows) for f in fields}
        if any(not math.isfinite(x) or x <= 0 for series in measures.values() for x in series):
            raise ValueError("Missing or invalid paired measurement")
        medians[keys] = {field: series[1] for field, series in measures.items()}
    outcomes = {}
    for scene, arm, steps in per_scene:
        label = f"{arm}_{steps}"
        ratios = {f: medians[scene, arm, steps][f] / medians[scene, "four", 40][f] for f in fields}
        outcomes.setdefault(label, {"scene_ratios": {}})["scene_ratios"][scene] = ratios
    for label, result in outcomes.items():
        if set(result["scene_ratios"]) != set(layouts):
            raise ValueError(f"Missing paired scene for {label}")
        result["ratios"] = {f: math.fsum(result["scene_ratios"][s][f] for s in layouts)/len(layouts)
                            for f in fields}
    controls = [f"{arm}_{steps}" for arm in ("original", "four") for steps in (5, 10, 40)]

    def quality(key):
        return math.fsum(outcomes[key]["ratios"][f] for f in QUALITY)/2

    def resources(key):
        row = outcomes[key]["ratios"]
        return row["scene_seconds"], row["peak_allocated"]

    advancing = []
    for key, item in outcomes.items():
        if not key.startswith("cached"):
            continue
        met = all(item["ratios"][f] <= plan["thresholds"][f] for f in fields)
        matches = sorted((k for k in controls if resources(k)[0] <= resources(key)[0]),
                         key=lambda k: (quality(k), *resources(k)))
        chosen = matches[0] if matches else None
        better = None if chosen is None else (quality(key) < quality(chosen) and all(
            item["ratios"][f] <= 1.01*outcomes[chosen]["ratios"][f] for f in QUALITY))
        item.update(time_matched_control=chosen, matches_resource_and_quality=met,
                    beats_time_matched_control=better)
        if better and met:
            advancing.append(key)
    nominated = min(advancing, key=lambda k: (*resources(k), k)) if advancing else None
    verdict = "advance" if nominated else "missed_targets"
    verified = dict(verdict=verdict, comparisons=outcomes, nominated=nominated)
    if not equivalent(verified, published):
        raise ValueError("The published decision differs from independently recomputed medians and thresholds")
    return dict(verified=True, cases=len(measured), comparisons=len(outcomes), verdict=verdict, nominated=nominated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator-sensitivity", action="store_true")
    args = parser.parse_args()
    plan = read(ROOT/"manifests"/f"{STUDY}.json")
    if args.allocator_sensitivity:
        evidence = read(RESULTS/"partial_evidence.json")
        diagnostics = read(RESULTS/"allocator_diagnostics.json")
        evidence["rows"] += [r["result"] for r in diagnostics["attempts"] if r["result"] is not None]
        answer = verify(plan, evidence, read(RESULTS/"allocator_sensitivity_decision.json"))
    else:
        answer = verify(plan, read(RESULTS/"evidence.json"), read(RESULTS/"decision.json"))
    print(json.dumps(answer, indent=2))
