"""Independently verify step selection, held errors, and the public verdict."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/native_lora_step_v1"
RAW = ROOT / "reports/native_lora_step_v1"
STATES = ("zero", "quarter", "half", "full")
ROOMS = ("apt1", "apt2", "office2")
SIDES = ("earlier", "later")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(observed, expected):
    if None in (observed, expected):
        equal = observed is expected
    else:
        equal = math.isclose(observed, expected, abs_tol=1e-10, rel_tol=1e-12)
    if not equal:
        raise ValueError(f"Score differs: {observed}, {expected}")


def view_mean(rows, minimum, counts):
    if [row["view"] for row in rows] != list(range(5)) or [row["count"] for row in rows] != counts:
        raise ValueError("The fixed view or sample inventory changed")
    available = []
    for row in rows:
        valid = row["count"] >= minimum and row["invalid_predictions"] == 0
        if valid != (row["mae_mm"] is not None):
            raise ValueError("A missing native measurement was removed from the aggregation")
        if valid:
            available.append(row["mae_mm"])
    return mean(available) if len(available) == 5 else None


def replay(manifest, seal, scored):
    identifiers = {f"{env}_{side}" for env in ROOMS for side in SIDES}
    if set(seal["cases"]) != identifiers or set(scored["cases"]) != identifiers:
        raise ValueError("Case inventory changed")
    values, deltas, gains = {}, {}, {}
    count = 0
    for job, case in seal["cases"].items():
        result = scored["cases"][job]
        env, side = job.split("_")
        input_case = manifest["inputs"]["environments"][env]["packets"][side]
        if set(case["states"]) != set(STATES) or set(result["view_rows"]) != set(STATES):
            raise ValueError("Candidate state inventory changed")
        select_errors = []
        held_errors = {}
        if result["held_coverage"] != input_case["held_coverage"]:
            raise ValueError("Coverage differs from its prelaunch receipt")
        for i, state in enumerate(STATES):
            endpoint = case["states"][state]
            check(endpoint["multiplier"], (0, .25, .5, 1)[i])
            selection = view_mean(endpoint["selector_rows"], 8, input_case["counts"][2])
            check(selection, endpoint["selector_mae_mm"])
            select_errors.append(selection)
            held_errors[state] = view_mean(result["view_rows"][state], 16,
                                           [r["count"] for r in result["held_coverage"]])
            count += 5
        selected = next(state for state, error in zip(STATES, select_errors) if error - min(select_errors) <= .05)
        if result["chosen"] != selected or case["chosen"] != selected:
            raise ValueError("Published choice differs from the reserved adaptation loss")
        held_errors["selected"] = held_errors[selected]
        for model, value in held_errors.items():
            check(value, result["errors_mm"][model])
        x, y = held_errors["selected"], held_errors["full"]
        delta = None if x is None or y is None else x - y
        gain = None if delta is None or not y else -delta / y
        check(delta, result["delta_mm"])
        check(gain, result["relative_gain_vs_full"])
        values[job], deltas[job], gains[job] = held_errors, delta, gain
    all_valid = sum(all(v is not None for v in result.values()) for result in values.values())
    if all_valid != scored["valid_cases"]:
        raise ValueError("Valid case count differs")
    direction_gains = {}
    for side in SIDES:
        averages = {}
        for model in ("selected", *STATES):
            v = [values[f"{env}_{side}"][model] for env in ROOMS]
            averages[model] = None if None in v else mean(v)
            check(averages[model], scored["directional"][side]["errors_mm"][model])
        x, y = averages["selected"], averages["full"]
        gain = None if x is None or not y else 1. - x/y
        check(gain, scored["directional"][side]["relative_gain_vs_full"])
        direction_gains[side] = gain
    both = [env for env in ROOMS if all(deltas[f"{env}_{side}"] is not None and
                                         deltas[f"{env}_{side}"] < -.05 for side in SIDES)]
    worse = [job for job, gain in gains.items() if gain is not None and gain < -.01]
    passes = (all_valid == 6 and all(g is not None and g >= .01 for g in direction_gains.values())
              and len(both) >= 2 and not worse)
    verdict = "consistent_gain" if passes else "mixed_or_small_gain" if all_valid == 6 else "indeterminate"
    if (verdict != scored["verdict"] or both != scored["both_directions_improved"] or
            worse != scored["regressed_over_one_percent"]):
        raise ValueError("Registered success rule disagrees with replay")
    return dict(verified=True, verdict=verdict, held_view_rows=count)


def local_replay(seal, scored):
    import numpy as np

    comparisons = 0
    for job, case in seal["cases"].items():
        env, side = job.split("_")
        opposite = "earlier" if side == "later" else "later"
        def arrays(direction):
            with np.load(RAW / "intake" / env / f"{direction}.npz", allow_pickle=False) as archive:
                return [{key: archive[f"{key}_{i}"] for key in ("xy", "depth", "source_ids", "fold")}
                        for i in range(5)]
        adaptation, held = arrays(side), arrays(opposite)
        for state, endpoint in case["states"].items():
            with np.load(RAW / "cases" / job / case["attempt"] / state / "native.npz", allow_pickle=False) as archive:
                depth, saved_scale = archive["depth"], float(archive["g0"])
            ratios = [float(z)/float(depth[i,y,x]) for i, item in enumerate(adaptation)
                      for (x, y), z, fold in zip(item["xy"], item["depth"], item["fold"]) if fold != 2]
            gauge = median(ratios)
            check(gauge, saved_scale)
            for i, (a, h) in enumerate(zip(adaptation, held)):
                used_pixels, used_rays = {tuple(p) for p in a["xy"]}, {tuple(p) for p in a["source_ids"]}
                scored_pixels = [j for j, (pixel, ray) in enumerate(zip(h["xy"], h["source_ids"]))
                                 if tuple(pixel) not in used_pixels and tuple(ray) not in used_rays]
                digest = hashlib.sha256(np.asarray(h["xy"][scored_pixels], dtype="<i8").tobytes()).hexdigest()
                if digest != scored["cases"][job]["held_coverage"][i]["xy_sha256"]:
                    raise ValueError("Independent held mask differs")
                targets = [(h, scored_pixels, scored["cases"][job]["view_rows"][state][i], 16),
                           (a, np.flatnonzero(a["fold"] == 2), endpoint["selector_rows"][i], 8)]
                for packet, selected, receipt, minimum in targets:
                    differences = []
                    for j in selected:
                        x, y = packet["xy"][j]
                        predicted = gauge * float(depth[i, y, x])
                        if math.isfinite(predicted) and predicted > 0:
                            differences.append(abs(predicted - float(packet["depth"][j])) * 1000.)
                    if (len(selected) != receipt["count"] or
                            len(selected) - len(differences) != receipt["invalid_predictions"]):
                        raise ValueError("Independent view coverage differs")
                    check(mean(differences) if len(differences) == len(selected) >= minimum else None,
                          receipt["mae_mm"])
                    comparisons += 1
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "manifests/native_lora_step_v1.json"
    seal_path = PUBLIC / "selection_seal.json"
    seal, result = read(seal_path), read(PUBLIC / "held_scores.json")
    def text_sha(path):
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if seal["manifest_sha256_lf"] != text_sha(manifest_path) or result["seal_sha256_lf"] != text_sha(seal_path):
        raise ValueError("Manifest or prediction seal binding differs")
    report = replay(read(manifest_path), seal, result)
    if args.private:
        report["verified_local_view_scores"] = local_replay(seal, result)
    print(json.dumps(report, indent=2))
