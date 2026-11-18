"""Seal the full prediction inventory, then evaluate fixed unseen temporal samples."""
import argparse
import hashlib
import math

import numpy as np

from native_lora_direction_inputs import RAW, SIDES, STATES
from native_lora_direction_registration import MANIFEST, RESULTS, inventory, verify_registration
from prepare_profiled_inputs import read, sha
from run_profiled_d0 import write_new


SEAL = RESULTS / "prediction_seal.json"
SCORES = RESULTS / "held_source_scores.json"


def case_id(room, side):
    return f"{room}_{side}"


def seal_inventory():
    cases = {}
    launches = RAW / "launches"
    for room, side in inventory():
        key = case_id(room, side)
        directory = RAW / "cases" / key
        meta = []
        for folder in sorted(launches.glob(f"tco-profiled-native-direction-{room}-{side}-*")):
            terminal = folder / "terminal.json"
            if terminal.is_file():
                record = read(terminal)
                if record["stopped"]:
                    meta.append(dict(name=folder.name, **record))
        if not meta:
            raise ValueError(f"No stopped launch receipt for {key}")
        if not any(m["code"] == 0 and m["error"] is None for m in meta):
            cases[key] = dict(valid=False, launches=meta)
            continue
        direction_path = directory / "direction.json"
        direction = read(direction_path)
        if (not direction["model_unchanged"] or direction["optimizer_updates"] != 1 or
                (direction["environment"], direction["side"]) != (room, side)):
            raise ValueError("Direction receipt violates the one-step frozen-backbone contract")
        for name, expected in direction["files"].items():
            if sha(directory / name) != expected:
                raise ValueError("Registered direction factors or gradient changed")
        records = {}
        for state, multiplier in STATES.items():
            source = directory / state / "complete.json"
            receipt = read(source)
            label = {"repeat_zero": "zero", "repeat_full": "plus_full"}.get(state, state)
            if (receipt["state"] != state or receipt["multiplier"] != multiplier
                    or receipt["adapter_sha256"] != direction["states"][label]["adapter_sha256"]
                    or not receipt["model_unchanged"] or receipt["optimizer_updates"] != 0):
                raise ValueError(f"Wrong signed state or model status for {key}/{state}")
            for filename, field in (("native.npz", "native_sha256"), ("trace.json", "trace_sha256")):
                if sha(source.with_name(filename)) != receipt[field]:
                    raise ValueError(f"Native evidence changed for {key}/{state}/{filename}")
            records[state] = dict(receipt_sha256=sha(source), **receipt)
        cases[key] = dict(valid=True, direction_sha256=sha(direction_path), endpoints=records, launches=meta)
    return dict(manifest_sha256_lf=sha(MANIFEST, text=True), cases=cases, rental_compute_usd=0)


def data_packet(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Temporal packet digest differs: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        return [{k: data[f"{k}_{i}"] for k in ("xy", "depth", "source_ids", "fold")}
                for i in range(5)]


def held_mask(allowed, opposite):
    pixels = set(map(tuple, allowed["xy"]))
    sources = set(map(tuple, allowed["source_ids"]))
    return np.asarray([tuple(xy) not in pixels and tuple(ray) not in sources
                       for xy, ray in zip(opposite["xy"], opposite["source_ids"])], dtype=bool)


def evaluate(rasters, packets, masks, scale, minimum):
    rows = []
    for view, (raster, packet, mask) in enumerate(zip(rasters, packets, masks)):
        xy = packet["xy"][mask]
        measured = packet["depth"][mask]
        if len(xy) != len(np.unique(xy, axis=0)) or np.any(xy < 0) or (
                np.any(xy >= [raster.shape[1], raster.shape[0]])):
            raise ValueError("Fixed native coordinates are duplicated or out of bounds")
        predicted = raster[xy[:, 1], xy[:, 0]].astype(np.float64) * scale
        valid = np.isfinite(predicted) & (predicted > 0)
        if np.any(~np.isfinite(measured)) or np.any(measured <= 0):
            raise ValueError("Invalid depth target on a fixed mask")
        mae = float(1000 * np.mean(np.abs(predicted - measured))) if len(xy) >= minimum and valid.all() else None
        rows.append(dict(view=view, count=len(xy), invalid_predictions=int((~valid).sum()), mae_mm=mae))
    return rows


def means(rows):
    return math.fsum(row["mae_mm"] for row in rows) / 5 if all(
        row["mae_mm"] is not None for row in rows) else None


def sign(value, deadband):
    return -1 if value < -deadband else 1 if value > deadband else 0


def classify(values, gradient_dot, objectives, config):
    if any(v is None for v in values.values()):
        return dict(status="invalid", realized="invalid")
    zero, full = values["zero"], values["plus_full"]
    b = max(config["floor_mm"], config["repeat_multiplier"] * abs(zero-values["repeat_zero"]),
            config["repeat_multiplier"] * abs(full-values["repeat_full"]))
    half_diff = values["plus_half"] - values["minus_half"]
    full_diff = full - values["minus_full"]
    effect = full - zero
    indicators = [sign(x, b) for x in (half_diff, full_diff, effect)]
    descent = gradient_dot < 0 and objectives["plus_full"] < objectives["zero"]
    if not descent:
        status = "surrogate_non_descent"
    elif indicators == [-1, -1, -1]:
        status = "aligned"
    elif indicators == [1, 1, 1]:
        status = "opposed"
    else:
        status = "mixed"
    return dict(status=status, realized={-1: "helped", 0: "within_deadband", 1: "worsened"}[indicators[-1]],
                deadband_mm=b, signs=indicators, slope_half_mm=half_diff, slope_full_mm=full_diff/2,
                actual_change_mm=effect, relative_gain_percent=-100*effect/zero if zero else None,
                surrogate_descent=descent)


def verdict(cases):
    statuses = {key: result["classification"]["status"] for key, result in cases.items()}
    if len(statuses) != 6 or "invalid" in statuses.values():
        return "indeterminate"
    for agreed, contrary in (("aligned", "opposed"), ("opposed", "aligned")):
        both = sum(all(statuses[case_id(room, side)] == agreed for side in SIDES)
                   for room in {room for room, _ in inventory()})
        if both >= 2 and contrary not in statuses.values():
            return agreed
    return "mixed"


def score(registration, seal):
    raw_inputs = RAW / registration["intake"]
    results = {}
    cfg = registration["scoring"]
    minimum = cfg["min_valid_per_view"]
    for room, side in inventory():
        opposite = next(s for s in SIDES if s != side)
        key = case_id(room, side)
        case = seal["cases"][key]
        if not case["valid"]:
            results[key] = dict(classification=dict(status="invalid", realized="invalid"), reason="local run failed")
            continue
        sources = registration["inputs"]["environments"][room]
        allowed, unseen = [data_packet(raw_inputs / room / f"{s}.npz", sources["packets"][s]["sha256"])
                           for s in (side, opposite)]
        masks = [held_mask(a, b) for a, b in zip(allowed, unseen)]
        query = [p["fold"] == 1 for p in allowed]
        samples = [dict(view=i, held_count=int(mask.sum()), excluded=int((~mask).sum()),
                        held_mask_sha256=hashlib.sha256(
                            np.ascontiguousarray(packet["xy"][mask], dtype="<i8").tobytes()).hexdigest())
                   for i, (packet, mask) in enumerate(zip(unseen, masks))]
        output = RAW / "cases" / key
        direction = read(output / "direction.json")
        rows, adapted, g0 = {}, {}, {}
        for label in STATES:
            with np.load(output / label / "native.npz", allow_pickle=False) as data:
                rasters, scale = data["depth"], float(data["g0"])
            if rasters.shape != (5, 384, 512) or not math.isfinite(scale) or scale <= 0 or (
                    scale != case["endpoints"][label]["g0"]):
                raise ValueError("Native depth or metric gauge violates the seal")
            g0[label] = scale
            rows[label] = evaluate(rasters, unseen, masks, scale, minimum)
            adapted[label] = evaluate(rasters, allowed, query, scale, minimum)
        values = {label: means(view_rows) for label, view_rows in rows.items()}
        loss = {name: item["J"] for name, item in direction["states"].items()}
        results[key] = dict(room=sources["room"], side=side, sampled=samples, g0=g0,
                            view_rows=rows, query_view_rows=adapted, mae_mm=values,
                            query_mae_mm={label: means(v) for label, v in adapted.items()},
                            direction=direction,
                            surrogate_repeat_difference={name: case["endpoints"][label]["surrogate_repeat"] - loss[name]
                                for label, name in (("repeat_zero", "zero"), ("repeat_full", "plus_full"))},
                            classification=classify(values, direction["dot"], loss, cfg))
    totals = {}
    for label in ("helped", "worsened", "within_deadband", "invalid"):
        totals[label] = sum(item["classification"]["realized"] == label for item in results.values())
    equal_rooms = {}
    for side in SIDES:
        side_cases = [results[case_id(room, s)] for room, s in inventory() if s == side]
        equal_rooms[side] = {state: float(np.mean([c["mae_mm"][state] for c in side_cases]))
                            if all(c.get("mae_mm", {}).get(state) is not None for c in side_cases)
                            else None for state in STATES}
    return dict(seal_sha256_lf=sha(SEAL, text=True), config=cfg, verdict=verdict(results), actual_steps=totals,
                equal_rooms_mae_mm=equal_rooms, cases=results, rental_compute_usd=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "score"))
    args = parser.parse_args()
    registration = verify_registration(require_committed=True)
    current = seal_inventory()
    if args.mode == "seal":
        write_new(SEAL, current)
    else:
        if read(SEAL) != current:
            raise ValueError("Prediction seal does not reproduce")
        write_new(SCORES, score(registration, current))
