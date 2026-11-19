"""Bound the registered median-ratio gates when a paired job timed out."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tco_frozen_prefix_protocol import PUBLIC, collect, read, verify, write_new


def interval(values):
    ordered = sorted(values)
    if not 2 <= len(ordered) <= 3:
        raise ValueError("Two of three observations are needed for a finite median interval")
    return (ordered[0], ordered[1]) if len(ordered) == 2 else (ordered[1], ordered[1])


def bound(contract, evidence):
    jobs = {j["id"]: j for j in contract["jobs"]}
    found = {r["id"]: r for r in evidence["rows"]}
    if len(found) != len(evidence["rows"]) or not found.keys() <= jobs.keys():
        raise ValueError("Wrong evidence inventory")
    observed = defaultdict(list)
    for name, entry in found.items():
        if any(entry[k] != v for k, v in jobs[name].items()) or entry["rental_compute_usd"] != 0:
            raise ValueError("Registered identity or spend changed")
        observed[entry["scene"], entry["arm"], entry["steps"]].append(entry)
    scenes = sorted({j["scene"] for j in jobs.values()})
    variants = sorted({(j["arm"], j["steps"]) for j in jobs.values()})
    results, controls = {}, {}
    for arm, steps in variants:
        bounds = {}
        for field in contract["thresholds"]:
            limits = []
            for scene in scenes:
                def selection(variant, budget):
                    values = observed[scene, variant, budget]
                    numbers = [r[field] if field in r else r["metrics"][field] for r in values]
                    return interval(numbers)

                low, high = selection(arm, steps)
                base_lo, base_hi = selection("four", 40)
                if min(low, base_lo) <= 0:
                    raise ValueError("Ratio requires positive measurements")
                limits.append((low/base_hi, high/base_lo))
            bounds[field] = [1., 1.] if (arm, steps) == ("four", 40) else list(map(mean, zip(*limits)))
        failed = [field for field, (low, _) in bounds.items() if low > contract["thresholds"][field]]
        key = f"{arm}_{steps}"
        if arm.startswith("cached"):
            results[key] = dict(bounds=bounds, criteria_ruled_out=failed)
        else:
            controls[key] = bounds
    ruled_out = all(r["criteria_ruled_out"] for r in results.values())
    return dict(complete=len(found), pending=sorted(jobs.keys()-found.keys()), bounds=results,
                control_bounds=controls, every_candidate_ruled_out=ruled_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-incomplete", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.evidence:
        from tco_frozen_prefix_protocol import MANIFEST

        document = bound(read(MANIFEST), read(args.evidence))
        if document != read(args.evidence.with_name("partial_bounds.json")):
            raise ValueError("Published incomplete decision does not reproduce")
        print(f"Verified interval decisions from {document['complete']} completed runs")
    else:
        import json

        study = verify()
        evidence = collect(study)
        outcome = bound(study, evidence)
        print(json.dumps(outcome, indent=2))
        if args.publish_incomplete:
            if not outcome["pending"]:
                raise ValueError("Use the complete decision for a complete study")
            write_new(PUBLIC/"partial_evidence.json", evidence)
            write_new(PUBLIC/"partial_bounds.json", outcome)
