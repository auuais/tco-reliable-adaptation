"""Independently verify the published directional verdict and optionally its local samples."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/native_lora_direction_v1"
PRIVATE = ROOT / "reports/native_lora_direction_v1"
STATES = ("zero", "minus_half", "plus_half", "minus_full", "plus_full", "repeat_zero", "repeat_full")
ROOMS = ("apt1", "apt2", "office2")
SIDES = ("earlier", "later")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(value, expected):
    if expected is None or value is None:
        if value is not expected:
            raise ValueError(f"Undefined metric disagrees: {value}, {expected}")
    elif not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError(f"Metric disagrees: {value}, {expected}")


def replay(result, seal, expected_seal):
    if set(result["cases"]) != {f"{r}_{s}" for r in ROOMS for s in SIDES}:
        raise ValueError("Wrong room and side inventory")
    if result["seal_sha256_lf"] != expected_seal or result["rental_compute_usd"] != 0:
        raise ValueError("Wrong prediction seal or rental accounting")
    outcomes, classes, mean, rows_seen = Counter(), {}, {}, 0
    for key, case in result["cases"].items():
        label = case["classification"]["status"]
        if not seal["cases"][key]["valid"]:
            if label != "invalid":
                raise ValueError("Missing or mislabeled invalid case")
            outcomes["invalid"] += 1
            classes[key] = "invalid"
            continue
        if set(case["view_rows"]) != set(STATES) or set(case["query_view_rows"]) != set(STATES):
            raise ValueError("Wrong signed-state inventory")
        endpoints = seal["cases"][key]["endpoints"]
        for state in STATES:
            verify(case["g0"][state], endpoints[state]["g0"])
            expected_sign = dict(zip(STATES, (0, -.5, .5, -1, 1, 0, 1)))[state]
            if endpoints[state]["multiplier"] != expected_sign:
                raise ValueError("Signed state disagrees with the prediction seal")
        for table, published in (("view_rows", "mae_mm"), ("query_view_rows", "query_mae_mm")):
            values = {}
            for state in STATES:
                rows = case[table][state]
                if [row["view"] for row in rows] != list(range(5)):
                    raise ValueError("View inventory or order differs")
                if table == "view_rows":
                    rows_seen += 5
                    if [row["count"] for row in rows] != [sample["held_count"] for sample in case["sampled"]]:
                        raise ValueError("Fixed held mask coverage changed")
                for row in rows:
                    defined = row["count"] >= result["config"]["min_valid_per_view"] and row["invalid_predictions"] == 0
                    if (row["mae_mm"] is not None) != defined:
                        raise ValueError("Invalid coverage was treated as a defined metric")
                summary = [row["mae_mm"] for row in rows]
                values[state] = statistics.mean(summary) if None not in summary else None
                verify(values[state], case[published][state])
            if table == "view_rows":
                mean[key] = values
        e = mean[key]
        if None in e.values():
            if label != "invalid":
                raise ValueError("A case with undefined view error was accepted")
            classes[key] = "invalid"
            outcomes["invalid"] += 1
            continue
        b = max(.05, 2 * abs(e["repeat_zero"] - e["zero"]), 2 * abs(e["repeat_full"] - e["plus_full"]))
        signed = [e["plus_half"]-e["minus_half"], e["plus_full"]-e["minus_full"], e["plus_full"]-e["zero"]]
        limits = [int(x > b) - int(x < -b) for x in signed]
        direction = case["direction"]
        descent = direction["dot"] < 0 and direction["states"]["zero"]["J"] > direction["states"]["plus_full"]["J"]
        expected = "surrogate_non_descent"
        if descent:
            expected = {(-1, -1, -1): "aligned", (1, 1, 1): "opposed"}.get(tuple(limits), "mixed")
        real = {-1: "helped", 0: "within_deadband", 1: "worsened"}[limits[2]]
        classes[key] = expected
        outcomes[real] += 1
        info = case["classification"]
        if (label, info["realized"], info["signs"], info["surrogate_descent"]) != (expected, real, limits, descent):
            raise ValueError("Case classification disagrees")
        for field, value in (("deadband_mm", b), ("slope_half_mm", signed[0]), ("slope_full_mm", signed[1]/2),
                             ("actual_change_mm", signed[2]),
                             ("relative_gain_percent", -100*signed[2]/e["zero"] if e["zero"] else None)):
            verify(value, info[field])
    verdict = "indeterminate" if "invalid" in classes.values() else "mixed"
    if verdict == "mixed":
        paired = Counter(classes[f"{r}_earlier"] for r in ROOMS
                         if classes[f"{r}_earlier"] == classes[f"{r}_later"])
        for name, contrary in (("aligned", "opposed"), ("opposed", "aligned")):
            if paired[name] >= 2 and contrary not in classes.values():
                verdict = name
    for side in SIDES:
        for state in STATES:
            values = [mean.get(f"{room}_{side}", {}).get(state) for room in ROOMS]
            verify(statistics.mean(values) if None not in values else None,
                   result["equal_rooms_mae_mm"][side][state])
    if verdict != result["verdict"] or any(outcomes[k] != v for k, v in result["actual_steps"].items()):
        raise ValueError("Published study decision disagrees")
    return dict(verified=True, verdict=verdict, rows=rows_seen, actual_steps=dict(outcomes))


def local_replay(result, manifest):
    import numpy as np

    comparisons = 0
    for key, case in result["cases"].items():
        if "view_rows" not in case:
            continue
        room, side = key.split("_")
        other = "later" if side == "earlier" else "earlier"
        home = PRIVATE / manifest["intake"] / room
        with np.load(home / f"{side}.npz") as train, np.load(home / f"{other}.npz") as held:
            for state in STATES:
                with np.load(PRIVATE / "cases" / key / state / "native.npz") as raw:
                    raster, saved_gauge = raw["depth"], float(raw["g0"])
                ratios = [float(z) / float(raster[i, y, x]) for i in range(5)
                          for (x,y), z in zip(train[f"xy_{i}"], train[f"depth_{i}"])]
                gauge = statistics.median(ratios)
                verify(gauge, saved_gauge)
                for i in range(5):
                    train_pixels = {tuple(xy) for xy in train[f"xy_{i}"]}
                    train_rays = {tuple(src) for src in train[f"source_ids_{i}"]}
                    held_indices = [j for j, (xy, src) in enumerate(zip(held[f"xy_{i}"], held[f"source_ids_{i}"]))
                                    if tuple(xy) not in train_pixels and tuple(src) not in train_rays]
                    matched_pixels = np.asarray(held[f"xy_{i}"][held_indices], dtype="<i8")
                    if (hashlib.sha256(matched_pixels.tobytes()).hexdigest()
                            != case["sampled"][i]["held_mask_sha256"]):
                        raise ValueError("Fixed held mask digest differs")
                    for packet, selected, table in ((held, held_indices, "view_rows"),
                                                    (train, np.flatnonzero(train[f"fold_{i}"]), "query_view_rows")):
                        values = []
                        for j in selected:
                            x, y = packet[f"xy_{i}"][j]
                            z = gauge * float(raster[i, y, x])
                            if math.isfinite(z) and z > 0:
                                values.append(1000*abs(z - float(packet[f"depth_{i}"][j])))
                        record = case[table][state][i]
                        if (record["count"] != len(selected) or
                                record["invalid_predictions"] != len(selected)-len(values)):
                            raise ValueError("Independent local coverage differs")
                        value = statistics.mean(values) if len(values) == len(selected) and len(values) >= 8 else None
                        verify(value, record["mae_mm"])
                        comparisons += 1
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    seal = PUBLIC / "prediction_seal.json"
    scores = read(PUBLIC / "held_source_scores.json")
    digest = hashlib.sha256(seal.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    report = replay(scores, read(seal), digest)
    if args.private:
        report["verified_local_view_scores"] = local_replay(
            scores, read(ROOT / "manifests/native_lora_direction_execution_v1.json"))
    print(json.dumps(report, indent=2))
