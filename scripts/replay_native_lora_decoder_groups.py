"""Recompute decoder group choices and development decisions from CPU evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from replay_native_lora_step import check, read, view_mean

ROOT = Path(__file__).resolve().parents[1]
STUDY = "native_lora_decoder_groups_v1"
RAW = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
OLD = ROOT / "reports/native_lora_step_v1"


def text_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def replay(manifest, seal, scores):
    policies = manifest["policies"]
    states = policies["scalar"] | policies["split"]
    identifiers = {entry["id"] for entry in manifest["development_cases"]}
    if len(identifiers) != 6 or set(seal["cases"]) != identifiers or set(scores["cases"]) != identifiers:
        raise ValueError("Develop­ment case inventory changed")
    if len(states) != 9 or any(len(p) != 7 for p in policies.values()):
        raise ValueError("Candidate cost or inventory changed")
    summaries = {"earlier": [], "later": []}
    valid_cases = 0
    for entry in manifest["development_cases"]:
        job = entry["id"]
        run, result = seal["cases"][job], scores["cases"][job]
        if (run["job"] != job or result["room"] != entry["room"] or result["side"] != entry["side"] or
                set(result["states"]) != set(states) or set(run["states"]) != set(states)):
            raise ValueError(f"Case or candidate binding changed for {job}")
        if result["coverage"] != entry["packet"]["held_coverage"]:
            raise ValueError("Held mask differs from registration")
        selector, held = {}, {}
        for key, pair in states.items():
            row, scored = run["states"][key], result["states"][key]
            if (row["candidate"] != key or row["multipliers"] != pair or row["g0"] <= 0 or
                    scored["applied_scale"] != row["g0"] or scored["native_sha256"] != row["native_sha256"]):
                raise ValueError("Prediction or scale differs from the seal")
            selector[key] = view_mean(row["selector_rows"], manifest["min_partition"], entry["packet"]["counts"][2])
            held[key] = view_mean(scored["views"], manifest["min_held"], [c["count"] for c in result["coverage"]])
            check(selector[key], row["selector_mae_mm"])
            check(held[key], result["errors_mm"][key])
        norms = run["group_norms_sq"]
        if len(norms) != 2 or any(not math.isfinite(n) or n <= 0 for n in norms):
            raise ValueError("Invalid adapter norms")
        choices, best, summary = {}, {}, {}
        for label, candidates in policies.items():
            minimum = min(selector[k] for k in candidates)
            ranked = sorted(candidates, key=lambda k: (sum(n*w*w for n, w in zip(norms, candidates[k])),
                                                        candidates[k]))
            choices[label] = next(k for k in ranked if selector[k] <= minimum + manifest["selection_tolerance_mm"])
            available = [(held[k], candidates[k], k) for k in candidates if held[k] is not None]
            best[label] = min(available)[-1] if len(available) == len(candidates) else None
            summary[f"{label}_best"] = held.get(best[label])
            summary[f"{label}_selected"] = held[choices[label]]
        original = sorted(manifest["archived"], key=states.__getitem__)
        limit = min(selector[k] for k in original) + manifest["selection_tolerance_mm"]
        legacy = next(k for k in original if selector[k] <= limit)
        summary.update(s000=held["s000"], s100=held["s100"], legacy_selected=held[legacy])
        if (choices != run["selected"] or choices != result["selected"] or
                best != result["hindsight_best"] or legacy != result["legacy_selected"]):
            raise ValueError("Published selection or hindsight winners differ")
        for key, value in summary.items():
            check(value, result["summary_mm"][key])
        valid_cases += all(x is not None for x in held.values())
        summaries[entry["side"]].append(summary)
    directional = {}
    for side, cases in summaries.items():
        averages = {name: mean(c[name] for c in cases) if all(c[name] is not None for c in cases) else None
                    for name in cases[0]}
        for name, value in averages.items():
            check(value, scores["directional"][side]["mean_mm"][name])
        gains = []
        for mode, field in (("best", "hindsight_gain"), ("selected", "selected_gain")):
            baseline, split = averages[f"scalar_{mode}"], averages[f"split_{mode}"]
            gain = 1 - split/baseline if baseline and split is not None else None
            check(gain, scores["directional"][side][field])
            gains.append(scores["directional"][side][field])
        directional[side] = gains
    advance = valid_cases == 6 and all(g[0] >= manifest["hindsight_relative_gain"] and g[1] > 0
                                      for g in directional.values())
    verdict = "advance" if advance else "stop_at_development" if valid_cases == 6 else "inconclusive"
    if (verdict, advance, valid_cases) != (scores["verdict"], scores["open_office1"], scores["valid_cases"]):
        raise ValueError("Published development gate differs from replay")
    return dict(verified=True, verdict=verdict, held_view_rows=len(states)*len(identifiers)*5)


def replay_private(manifest, seal, scores):
    import numpy as np

    prior = read(ROOT / "results/native_lora_step_v1/selection_seal.json")["cases"]
    compared = 0
    for entry in manifest["development_cases"]:
        job, env, side = entry["id"], entry["env"], entry["side"]
        other = "earlier" if side == "later" else "later"
        run, result = seal["cases"][job], scores["cases"][job]
        with np.load(OLD / "intake" / env / f"{side}.npz") as allowed, np.load(
                OLD / "intake" / env / f"{other}.npz") as unseen:
            for key, row in run["states"].items():
                if row["source"] == "archive":
                    base = OLD / "cases" / job / prior[job]["attempt"] / manifest["archived"][key]
                else:
                    base = RAW / "cases" / job / run["attempt"] / key
                with np.load(base / "native.npz") as output:
                    depth, gauge = output["depth"], float(output["g0"])
                ratios = []
                for v in range(5):
                    allowed_xy, allowed_rays = allowed[f"xy_{v}"], allowed[f"source_ids_{v}"]
                    permitted = allowed[f"fold_{v}"]
                    x, y = allowed_xy[permitted < 2].T
                    ratios.extend(allowed[f"depth_{v}"][permitted < 2] / depth[v, y, x].astype(float))
                    blocked_xy, blocked_rays = set(map(tuple, allowed_xy)), set(map(tuple, allowed_rays))
                    fixed_indices = [i for i, (xy, ray) in enumerate(zip(unseen[f"xy_{v}"], unseen[f"source_ids_{v}"]))
                                     if tuple(xy) not in blocked_xy and tuple(ray) not in blocked_rays]
                    fixed_xy = unseen[f"xy_{v}"][fixed_indices]
                    expected = result["coverage"][v]
                    mask_sha = hashlib.sha256(np.asarray(fixed_xy, dtype="<i8").tobytes()).hexdigest()
                    if len(fixed_xy) != expected["count"] or mask_sha != expected["xy_sha256"]:
                        raise ValueError("Private replay mask differs")
                    specs = ((unseen[f"depth_{v}"][fixed_indices], fixed_xy, result["states"][key]["views"][v]),
                             (allowed[f"depth_{v}"][permitted == 2], allowed_xy[permitted == 2],
                              row["selector_rows"][v]))
                    for measured, xy, evidence in specs:
                        pred = [float(depth[v, y, x])*gauge for x, y in xy]
                        bad = sum(not math.isfinite(z) or z <= 0 for z in pred)
                        if len(pred) != evidence["count"] or bad != evidence["invalid_predictions"]:
                            raise ValueError("Private replay prediction count differs")
                        check(None if bad else 1000*mean(abs(z-m) for z, m in zip(pred, measured)),
                              evidence["mae_mm"])
                        compared += 1
                check(median(ratios), gauge)
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "manifests" / f"{STUDY}.json"
    seal_path = PUBLIC / "selection_seal.json"
    manifest, seal, scores = read(manifest_path), read(seal_path), read(PUBLIC / "held_scores.json")
    if seal["manifest_sha256_lf"] != text_sha(manifest_path) or scores["seal_sha256_lf"] != text_sha(seal_path):
        raise ValueError("Public evidence does not match its seal")
    answer = replay(manifest, seal, scores)
    if args.private:
        answer["private_scorer_view_rows"] = replay_private(manifest, seal, scores)
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
