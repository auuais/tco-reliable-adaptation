"""Seal adaptation choices before scoring the opposite temporal depth packet."""
import argparse
import math

import numpy as np

from native_lora_step_inputs import (BASE_GAIN, EPSILON_MM, MANIFEST, MIN_HELD, PUBLIC, RAW, ROOMS,
                                     SIDES, STATES, array_digest, cases, verify_registration)
from prepare_profiled_inputs import read, sha
from run_profiled_d0 import write_new
from score_native_lora_direction import data_packet, evaluate, held_mask, means


SEAL = PUBLIC / "selection_seal.json"
SCORES = PUBLIC / "held_scores.json"


def choose(errors):
    if errors.keys() != STATES.keys() or any(not math.isfinite(value) for value in errors.values()):
        raise ValueError("Incomplete or invalid native selector scores")
    near = min(errors.values()) + EPSILON_MM
    return min((state for state, error in errors.items() if error <= near), key=STATES.__getitem__)


def build_seal():
    results = {}
    for env, side in cases():
        key = f"{env}_{side}"
        successes = []
        retired = []
        for launch in sorted((RAW / "launches" / key).glob("*/terminal.json")):
            receipt = read(launch)
            if (receipt["job"] == key and receipt["attempt"] == launch.parent.name and
                    receipt["stopped"] and receipt["exit_code"] == 0 and not receipt["error"]):
                successes.append((launch, receipt))
            else:
                retired.append(receipt)
        if len(successes) != 1:
            raise ValueError(f"Exactly one completed stopped case is required for {key}")
        launch, receipt = successes[0]
        path = RAW / "cases" / key / receipt["attempt"]
        selection = read(path / "choice.json")
        if (selection["env"] != env or selection["side"] != side or not selection["frozen_weights"] or
                selection["direction"]["optimizer_steps"] != 1 or selection["states"].keys() != STATES.keys()):
            raise ValueError("Invalid case identity, direction count or frozen model evidence")
        for name, expected in selection["direction"]["factor_files"].items():
            if sha(path / f"{name}.npz") != expected:
                raise ValueError("Direction factor file changed")
        errors = {}
        for state, multiplier in STATES.items():
            row = selection["states"][state]
            local = path / state
            if row != read(local / "complete.json") or row["multiplier"] != multiplier:
                raise ValueError("Native state differs from the choice record")
            for name, field in (("native.npz", "native_sha256"), ("trace.json", "trace_sha256")):
                if sha(local / name) != row[field]:
                    raise ValueError("Sealed native prediction or trace changed")
            errors[state] = means(row["selector_rows"])
            if errors[state] != row["selector_mae_mm"]:
                raise ValueError("Selector aggregation changed")
        if selection["chosen"] != choose(errors):
            raise ValueError("Chosen step does not match the adaptation rule")
        results[key] = dict(choice_sha256=sha(path / "choice.json"), launch_sha256=sha(launch),
                            launch=receipt, retired_attempts=retired, **selection)
    return dict(manifest_sha256_lf=sha(MANIFEST, text=True), cases=results, rental_compute_usd=0)


def summarize(scored):
    models = ["selected", *STATES]
    directional = {}
    valid = all(c["valid"] for c in scored.values())
    for side in SIDES:
        grouped = [scored[f"{env}_{side}"] for env in ROOMS]
        errors = {model: math.fsum(c["errors_mm"][model] for c in grouped)/len(grouped)
                  if all(c["errors_mm"][model] is not None for c in grouped) else None for model in models}
        gain = (errors["full"] - errors["selected"]) / errors["full"] if (
            errors["selected"] is not None and errors["full"]) else None
        directional[side] = dict(errors_mm=errors, relative_gain_vs_full=gain)
    improving = [env for env in ROOMS if all(
        scored[f"{env}_{side}"]["delta_mm"] is not None and scored[f"{env}_{side}"]["delta_mm"] < -EPSILON_MM
        for side in SIDES)]
    severe = [key for key, value in scored.items()
              if value["relative_gain_vs_full"] is not None and value["relative_gain_vs_full"] < -BASE_GAIN]
    passes = (valid and all(c["relative_gain_vs_full"] is not None and c["relative_gain_vs_full"] >= BASE_GAIN
                           for c in directional.values()) and len(improving) >= 2 and not severe)
    verdict = "consistent_gain" if passes else "mixed_or_small_gain" if valid else "indeterminate"
    return dict(verdict=verdict, directional=directional, both_directions_improved=improving,
                regressed_over_one_percent=severe, valid_cases=sum(c["valid"] for c in scored.values()))


def score(registration, seal):
    scored = {}
    for env, side in cases():
        other = "later" if side == "earlier" else "earlier"
        key = f"{env}_{side}"
        case = seal["cases"][key]
        record = registration["inputs"]["environments"][env]
        base = RAW / "intake" / env
        train, held = (data_packet(base / f"{s}.npz", record["packets"][s]["sha256"]) for s in (side, other))
        masks = [held_mask(a, b) for a, b in zip(train, held)]
        coverage = [dict(view=i, count=int(mask.sum()), excluded=int((~mask).sum()),
                         xy_sha256=array_digest(p["xy"][mask])) for i, (p, mask) in enumerate(zip(held, masks))]
        if coverage != record["packets"][side]["held_coverage"] or min(p["count"] for p in coverage) < MIN_HELD:
            raise ValueError("Fixed held coverage differs from preflight")
        result = {}
        for state, row in case["states"].items():
            with np.load(RAW / "cases" / key / case["attempt"] / state / "native.npz", allow_pickle=False) as output:
                depth, scale = output["depth"], float(output["g0"])
            if scale != row["g0"] or not math.isfinite(scale) or scale <= 0 or depth.shape != (5, 384, 512):
                raise ValueError("Native prediction or gauge does not match the sealed choice")
            result[state] = evaluate(depth, held, masks, scale, MIN_HELD)
        x = {state: means(rows) for state, rows in result.items()}
        x["selected"] = x[case["chosen"]]
        diff = x["selected"]-x["full"] if x["selected"] is not None and x["full"] is not None else None
        scored[key] = dict(room=record["room"], side=side, chosen=case["chosen"], held_coverage=coverage,
                           view_rows=result, errors_mm=x, delta_mm=diff,
                           relative_gain_vs_full=-diff/x["full"] if diff is not None and x["full"] else None,
                           valid=all(v is not None for v in x.values()))
    return dict(seal_sha256_lf=sha(SEAL, text=True), cases=scored,
                rental_compute_usd=0, **summarize(scored))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "score"))
    args = parser.parse_args()
    config = verify_registration(committed=True)
    current = build_seal()
    if args.mode == "seal":
        write_new(SEAL, current)
    else:
        if read(SEAL) != current:
            raise ValueError("The complete prospective selection seal does not reproduce")
        write_new(SCORES, score(config, current))
