"""Seal the redesign predictions and replay the registered held-source verdict."""
import argparse

import numpy as np

from profiled_compensation_protocol import (ARMS, EVALUATION, EXECUTION, PUBLIC, ROOT,
                                             SCENES, SIDES, file_identity, require)
from run_profiled_d0 import now, read, write_new
from score_profiled_lora import FRAMES, average_or_none, held_mask, load_endpoints, native_rows, verified_npz, view_metric
from seal_profiled_predictions import audit_attempt, timestamp

SEAL = f"{PUBLIC}/prediction_seal.json"
SCORES = f"{PUBLIC}/held_source_scores.json"
SAMPLES = f"{PUBLIC}/held_source_replay.npz"
ACCESS = f"{PUBLIC}/score_access.json"
ATTEMPT = "screen01"
COMPARATORS = ("P1", "P3", "R0", "R2")


def summarize(rows, failure_count):
    expected = {(scene, side, arm, frame) for scene in SCENES for side in SIDES
                for arm in ARMS for frame in FRAMES}
    fields = ("scene", "side", "arm", "frame")
    indexed = {tuple(row[k] for k in fields): row for row in rows}
    require(len(rows) == len(expected) and set(indexed) == expected,
            "Full five-view state grid required for both source directions")
    cases, directional, validity = {}, {}, {}
    for endpoint in ("native", "final"):
        cases[endpoint], directional[endpoint], validity[endpoint] = {}, {}, {}
        for scene in SCENES:
            for side in SIDES:
                for arm in ARMS:
                    run = f"{scene}_{side}_{arm}"
                    values = [indexed[scene, side, arm, frame][endpoint] for frame in FRAMES]
                    cases[endpoint][run] = average_or_none([value["mae_m"] for value in values])
                    validity[endpoint][run] = all(value["all_samples_valid"] for value in values)
        directional[endpoint] = {side: {arm: average_or_none(
            [cases[endpoint][f"{scene}_{side}_{arm}"] for scene in SCENES])
            for arm in ARMS} for side in SIDES}
    final = cases["final"]
    valid = validity["final"]
    contrasts = {}
    reasons = ["numerical_failures"] if failure_count else []
    for control in COMPARATORS:
        by_side = {}
        for side in SIDES:
            candidate, baseline = (directional["final"][side][arm] for arm in ("R1", control))
            eligible = all(valid[f"{scene}_{side}_{arm}"] for scene in SCENES for arm in ("R1", control))
            gain = (1. - candidate/baseline if eligible and candidate is not None and
                    baseline is not None and baseline > 0 else None)
            by_side[side] = gain
            if gain is None or (gain < .01 if control == "P1" else gain <= 0):
                reasons.append(f"R1_did_not_pass_against_{control}_{side}")
        contrasts[control] = by_side
    improving = [scene for scene in SCENES if all(
        valid[f"{scene}_{side}_{arm}"] for side in SIDES for arm in ("R1", "P1")) and all(
        final[f"{scene}_{side}_R1"] < final[f"{scene}_{side}_P1"] for side in SIDES)]
    if len(improving) < 2:
        reasons.append("fewer_than_two_same_scenes_improve_over_P1_in_both_directions")
    return dict(case_mae_m=cases, directional_mae_m=directional, validity=validity,
                r1_directional_relative_gain=contrasts, same_scenes_improving_over_p1=improving,
                rejection_reasons=reasons, nomination=None if reasons else "R1",
                scene_level_test_performed=False, rental_gate_passed=False)


def build_seal(registry):
    rows, intervals, first_steps = [], [], {}
    execution = file_identity(EXECUTION)
    for job in registry["jobs"]:
        for field in ("public_attempt_parent", "raw_attempt_parent"):
            parent = ROOT / job[field]
            require(parent.is_dir() and {p.name for p in parent.iterdir()} == {ATTEMPT},
                    "Missing or unexpected attempt: " + job["run_id"])
        inspected = audit_attempt(ROOT, job, registry, EXECUTION, execution["sha256"], ATTEMPT)
        require(inspected["status"] == "complete", "Prediction grid has an invalid attempt")
        name = "tco-profiled-compensation-" + job["run_id"].lower().replace("_", "-")
        terminal = f"{PUBLIC}/launches/{name}/terminal.json"
        receipt = read(ROOT / terminal)
        require(receipt["verified_stopped"] is True and receipt["state"]["ExitCode"] == 0,
                "Local attempt has no verified successful shutdown")
        complete = read(ROOT / inspected["report"])
        require(complete["config_sha256"] == registry["config_sha256"]
                and complete["endpoint_decoder_calls"] == 20, "Configuration or fresh capture differs")
        if job["arm"] in ("P3", "R0", "R1"):
            step = complete["steps"][0]
            fields = ("loss_before_update", "gradient_sha256_unclipped", "adapter_state_sha256")
            first_steps.setdefault((job["scene"], job["adaptation_source"]), []).append(
                tuple(step[field] for field in fields))
        intervals.append((timestamp(inspected["started_at_utc"]), timestamp(inspected["finished_at_utc"])))
        rows.append(dict(run_id=job["run_id"], scene=job["scene"], side=job["adaptation_source"],
                         arm=job["arm"], selected=inspected, launch_terminal=file_identity(terminal)))
    require(len(first_steps) == 6 and all(len(v) == 3 and len(set(v)) == 1 for v in first_steps.values()),
            "Fresh P3/R0/R1 first-step loss, gradients or updated adapter differ")
    require(len({r["selected"]["original_state_sha256"] for r in rows}) == 1,
            "Model identities differ across attempts")
    intervals.sort()
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
            "Scientific model attempt intervals overlap")
    return dict(schema="profiled_compensation_prediction_seal_v1", pass_all=True,
                execution=execution, evaluation=file_identity(EVALUATION),
                completed_states=36, numerical_failures=0, sample_arrays_loaded=False,
                exact_first_step_state_match_for_all_six_conditions=True, cases=rows)


def calculate(seal):
    inputs = read(ROOT / "results/lora_profiled_v1/inputs.json")
    split_path = ROOT / "results/lora_profiled_v1/splits.npz"
    split_keys = [f"{scene}_{side}_{frame}__xy" for scene in SCENES for side in SIDES for frame in FRAMES]
    splits = verified_npz(split_path, inputs["splits_binary_sha256"], split_keys)
    packet_keys = [f"{scene}_{frame}" for scene in SCENES for frame in FRAMES]
    packets = {side: verified_npz(ROOT / f"results/mast3r_partner_shape/{side}.npz",
                inputs["registration"]["packet_sha256"][side], packet_keys) for side in SIDES}
    masks, samples = {}, {}
    for scene in SCENES:
        for side in SIDES:
            opposite = "later" if side == "earlier" else "earlier"
            for frame in FRAMES:
                prefix = f"{scene}_{side}_{frame}"
                name = f"{scene}_{frame}"
                used = splits[prefix+"__xy"]
                full, _ = native_rows(packets[side][name])
                require(set(map(tuple, used)) == set(map(tuple, full)),
                        "Fixed overlap exclusion omits allowed adaptation pixels")
                xy, measured, counts = held_mask(used, packets[opposite][name])
                masks[prefix] = (xy, measured, counts)
                samples[prefix+"__xy"], samples[prefix+"__measured_m"] = xy, measured
    rows, runtimes, gauges = [], {}, {}
    for case in seal["cases"]:
        scene, side, arm = (case[name] for name in ("scene", "side", "arm"))
        prediction = load_endpoints(ROOT, case["selected"])
        for view, frame in enumerate(FRAMES):
            prefix = f"{scene}_{side}_{frame}"
            xy, measured, counts = masks[prefix]
            row = dict(scene=scene, side=side, arm=arm, frame=frame, **counts)
            for endpoint in ("native", "final"):
                values = prediction[endpoint][view, xy[:,1], xy[:,0]].astype(np.float64)
                samples[f"{prefix}__{arm}_{endpoint}_m"] = values
                row[endpoint] = view_metric(values, measured)
            rows.append(row)
        runtimes[case["run_id"]] = case["selected"]["seconds"]
        gauges[case["run_id"]] = prediction["metric_gauge"]
    return rows, samples, runtimes, gauges


def score(replay=False):
    from register_profiled_compensation import require_scoring_runtime, verify_release
    registry, evaluator = verify_release()
    require_scoring_runtime(evaluator)
    sealed = read(ROOT / SEAL)
    require(build_seal(registry) == sealed, "Prediction seal does not reproduce")
    access = dict(seal=file_identity(SEAL), evaluation=file_identity(EVALUATION))
    if (ROOT / ACCESS).exists():
        require(read(ROOT / ACCESS)["bound"] == access, "Previous access used a different seal")
    else:
        require(not replay, "No original scoring access")
        write_new(ROOT / ACCESS, dict(bound=access, first_access_at_utc=now()))
    if not replay:
        require(not (ROOT / SCORES).exists() and not (ROOT / SAMPLES).exists(), "Preserve original score artifacts")
    rows, samples, runtimes, gauges = calculate(sealed)
    report = dict(schema="profiled_compensation_scores_v1", rows=rows,
                  runtimes_seconds=runtimes, endpoint_gauges=gauges, evidence=access,
                  verdict=summarize(rows, sealed["numerical_failures"]), numerical_failures=0,
                  result_scope="observed development scenes, fixed held-source mask",
                  rental_compute_usd=0)
    if replay:
        original = read(ROOT / SCORES)
        require(report == {k: v for k, v in original.items() if k != "sample_replay"},
                "Exact source-backed score replay differs")
        original_samples = verified_npz(ROOT / SAMPLES, original["sample_replay"]["sha256"])
        require(samples.keys() == original_samples.keys() and all(
            np.array_equal(value, original_samples[key], equal_nan=True) for key, value in samples.items()),
            "Exact sampled predictions differ")
        write_new(ROOT / PUBLIC / "source_replay.json", dict(pass_all=True, report=file_identity(SCORES),
                  sample_replay=file_identity(SAMPLES), elapsed_validation_completed_at_utc=now()))
    else:
        with (ROOT / SAMPLES).open("xb") as output:
            np.savez_compressed(output, **samples)
        report["sample_replay"] = file_identity(SAMPLES)
        write_new(ROOT / SCORES, report)
    print(report["verdict"]["nomination"], report["verdict"]["r1_directional_relative_gain"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("seal", "score", "replay"))
    args = p.parse_args()
    if args.mode == "seal":
        from register_profiled_compensation import verify_release
        registry, _ = verify_release()
        require(not (ROOT / SEAL).exists(), "Preserve the existing prediction seal")
        write_new(ROOT / SEAL, build_seal(registry))
    else:
        score(replay=args.mode == "replay")


if __name__ == "__main__":
    main()
