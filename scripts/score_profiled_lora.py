"""Prospectively register, then score globally sealed profiled-LoRA predictions.

``prepare`` runs synthetic fixtures and hashes source/input bytes without loading
study observations. ``score`` and ``verify`` cannot open study arrays until the
evaluation registration and all 36 prediction attempts have been verified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform

import numpy as np

from seal_profiled_predictions import (ROOT, SCENES, SIDES, ARMS, RULES, FAILURE_POLICY, build_seal,
    check_map, dump_new, read, relative, require, sha, timestamp, verify_seal)


FRAMES = (25, 225, 425, 625, 825)
SOURCES = ("scripts/seal_profiled_predictions.py", "scripts/score_profiled_lora.py",
           "scripts/test_profiled_scoring.py", "scripts/register_profiled_execution.py",
           "scripts/prepare_profiled_inputs.py", "scripts/partner_shape.py",
           "scripts/score_partner_shape.py")
INPUTS = ("manifests/lora_profiled_design_v1.json", "results/lora_profiled_v1/inputs.json",
          "results/lora_profiled_v1/splits.npz")
ACCESS = "results/lora_profiled_v1/scoring_access.json"


def dependencies():
    return dict(python_major_minor=".".join(platform.python_version().split(".")[:2]), numpy=np.__version__)


def source_hashes(root):
    return {name: sha(relative(root, name), text=True) for name in SOURCES}


def validate_rules(rules):
    require(rules.get("schema") == "profiled_evaluation_rules_v1" and
            rules.get("candidate_arms") == ["P3", "P4"] and
            rules.get("control_arms") == ["P0", "P1", "P2", "P5"], "Unsupported prospective evaluation rules")
    nomination = rules["nomination"]
    require(nomination["minimum_directional_gain_each"] == .01 and
            nomination["minimum_same_scenes_improving_both_directions"] == 2 and
            nomination["numerical_failures_allowed"] == 0 and nomination["max_candidates"] == 1,
            "Nomination constants differ from this validated scorer")
    require(rules["global_prediction_seal"]["required_states"] == 36 and
            rules["held_source_metric"]["candidate_independent_mask"] is True,
            "Global inventory or mask contract differs")


def prepare_registration(root, output_name, fixture_name, rules_name=RULES):
    """Bind executable scoring and synthetic evidence before study-score access."""
    root = Path(root).resolve()
    output, fixture_path = relative(root, output_name), relative(root, fixture_name)
    require(not output.exists() and not fixture_path.exists(), "Preserve existing evaluation registration/evidence")
    require(not relative(root, ACCESS).exists(), "Study-score access already began; do not retroactively register evaluation changes")
    rules = read(relative(root, rules_name))
    validate_rules(rules)
    # Lazy import avoids a module cycle: fixtures exercise the public pure APIs.
    from test_profiled_scoring import run_fixtures
    fixtures = run_fixtures()
    require(fixtures.get("pass_all") is True and fixtures.get("synthetic_only") is True,
            "Synthetic scoring/sealing fixtures did not pass")
    sources = source_hashes(root)
    fixtures.update(source_sha256_lf=sources, dependencies=dependencies())
    inputs = {name: sha(relative(root, name)) for name in INPUTS}
    preparation = read(relative(root, "results/lora_profiled_v1/inputs.json"))
    packets = {f"results/mast3r_partner_shape/{side}.npz": value
               for side, value in preparation["registration"]["packet_sha256"].items()}
    require(set(packets) == {f"results/mast3r_partner_shape/{side}.npz" for side in SIDES},
            "Unexpected source-packet inventory")
    check_map(root, packets)
    require(inputs["results/lora_profiled_v1/splits.npz"] == preparation["splits_binary_sha256"],
            "Prepared split archive changed")
    dump_new(fixture_path, fixtures)
    result = dict(schema="profiled_evaluation_execution_v1", prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Executable scorer frozen after synthetic fixtures; no study arrays or scores accessed",
        rules=rules_name, rules_sha256=sha(relative(root, rules_name)), source_sha256_lf=sources,
        failure_policy=FAILURE_POLICY,
        dependencies=dependencies(), input_sha256=inputs, packet_sha256=packets,
        fixtures=fixture_name, fixtures_sha256=sha(fixture_path), synthetic_fixtures_pass=True,
        study_arrays_loaded=False, study_scores_read=False, rental_compute_usd=0)
    dump_new(output, result)
    return result


def verify_evaluation_registration(root, name):
    root = Path(root).resolve()
    path = relative(root, name)
    record = read(path)
    require(record.get("schema") == "profiled_evaluation_execution_v1" and
            record.get("synthetic_fixtures_pass") is True and
            record.get("study_arrays_loaded") is False and record.get("study_scores_read") is False,
            "Evaluation execution is not prospectively registered")
    timestamp(record.get("prepared_at_utc"))
    require(record["source_sha256_lf"] == source_hashes(root), "Registered evaluator source changed")
    require(record["dependencies"] == dependencies(), "Registered evaluator dependency versions differ")
    require(record.get("failure_policy") == FAILURE_POLICY, "Failure classification/review policy differs")
    require(sha(relative(root, record["rules"])) == record["rules_sha256"], "Evaluation rules changed")
    validate_rules(read(relative(root, record["rules"])))
    require(set(record["input_sha256"]) == set(INPUTS), "Evaluation input inventory differs")
    require(set(record["packet_sha256"]) == {f"results/mast3r_partner_shape/{side}.npz" for side in SIDES},
            "Evaluation packet inventory differs")
    check_map(root, record["input_sha256"])
    check_map(root, record["packet_sha256"])
    fixture = relative(root, record["fixtures"])
    require(sha(fixture) == record["fixtures_sha256"], "Registered fixture evidence changed")
    evidence = read(fixture)
    require(evidence.get("pass_all") is True and evidence.get("synthetic_only") is True and
            evidence.get("source_sha256_lf") == record["source_sha256_lf"] and
            evidence.get("dependencies") == record["dependencies"], "Fixture provenance does not match the scorer")
    require(evidence.get("checks_passed", 0) >= 36 and len(evidence.get("checks", [])) == evidence["checks_passed"]
            and len(set(evidence["checks"])) == evidence["checks_passed"], "Fixture check inventory is incomplete")
    return dict(file=name, sha256=sha(path))


def register_score_access(root, evaluation, seal_name):
    """Freeze a receipt before the first study-array decode, including failures."""
    identity = dict(evaluation_execution=evaluation, prediction_seal=seal_name,
                    prediction_seal_sha256=sha(relative(root, seal_name)))
    path = relative(root, ACCESS)
    if path.exists():
        previous = read(path)
        require(all(previous.get(key) == value for key, value in identity.items()),
                "Previous score access used another evaluation or prediction seal")
    else:
        dump_new(path, dict(identity, first_access_at_utc=datetime.now(timezone.utc).isoformat(),
                            purpose="Prospective evaluator sealed before any study observations are decoded"))
    return sha(path)


def native_rows(rows):
    """Exact stable nearest-depth mapping used by partner_shape.native_rows."""
    rows = np.asarray(rows)
    require(rows.ndim == 2 and rows.shape[1] >= 3 and np.isfinite(rows[:, :3]).all() and
            (rows[:, 2] > 0).all(), "Invalid measurement rows")
    pixels = np.rint((rows[:, :2] + .5) * [512/518, 384/392] - .5).astype(np.int64)
    nearest = {}
    for index in np.argsort(rows[:, 2], kind="stable"):
        x, y = pixels[index]
        if 0 <= x < 512 and 0 <= y < 384:
            nearest.setdefault((int(x), int(y)), int(index))
    selected = list(nearest.values())
    return pixels[selected].reshape(-1, 2), rows[selected, 2]


def held_mask(training_xy, opposite_rows):
    """Mask is fixed by observation identity, never candidate predictions."""
    training_xy = np.asarray(training_xy)
    require(training_xy.ndim == 2 and training_xy.shape[1] == 2 and
            np.issubdtype(training_xy.dtype, np.integer), "Expected native integer adaptation pixels")
    require(((training_xy >= 0) & (training_xy < [512, 384])).all(), "Adaptation pixel outside native raster")
    require(len(np.unique(training_xy, axis=0)) == len(training_xy), "Duplicate adaptation pixels")
    xy, depth = native_rows(opposite_rows)
    used = set(map(tuple, training_xy.tolist()))
    keep = np.asarray([tuple(pixel) not in used for pixel in xy.tolist()], dtype=bool)
    require(bool(keep.any()), "Empty held-source view after fixed overlap exclusion")
    return xy[keep], depth[keep].astype(np.float64), dict(
        mapped_opposite_count=len(xy), overlapping_excluded=int((~keep).sum()), held_count=int(keep.sum()),
        adaptation_count=len(training_xy))


def view_metric(predicted, measured):
    predicted, measured = np.asarray(predicted, dtype=np.float64), np.asarray(measured, dtype=np.float64)
    require(predicted.ndim == measured.ndim == 1 and predicted.shape == measured.shape and len(measured) > 0,
            "Empty or mismatched held-source vectors")
    require(np.isfinite(measured).all() and (measured > 0).all(), "Invalid fixed held-source observations")
    finite = np.isfinite(predicted)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.abs(predicted[finite] - measured[finite])
    representable = np.isfinite(errors)
    finite_errors = errors[representable]
    error_mean = (math.fsum(float(value)/len(finite_errors) for value in finite_errors)
                  if len(finite_errors) else None)
    return dict(count=len(measured), nonfinite_count=int((~finite).sum()),
        nonpositive_count=int((finite & (predicted <= 0)).sum()),
        mae_m=error_mean if finite.all() and representable.all() else None,
        finite_error_count=len(finite_errors), finite_error_mae_m=error_mean,
        unrepresentable_error_count=int((~representable).sum()),
        all_samples_valid=bool(finite.all() and representable.all() and (predicted > 0).all()),
        candidate_dependent_exclusions=0)


def average_or_none(values):
    return None if any(value is None for value in values) else math.fsum(float(value)/len(values) for value in values)


def verified_npz(path, expected, names=None):
    """Decode exactly the bytes whose hash was checked, avoiding a read race."""
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, f"Sealed array bytes changed: {path}")
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        selected = loaded.files if names is None else names
        return {name: loaded[name].copy() for name in selected}


def load_endpoints(root, selected):
    native = verified_npz(relative(root, selected["native_file"]), selected["native_sha256"], ("depth",))["depth"]
    final = verified_npz(relative(root, selected["final_file"]), selected["final_sha256"], ("depth","g0","scales"))
    depth, gauge, scales = final["depth"], final["g0"], final["scales"]
    require(native.shape == depth.shape == (5,384,512), "Endpoint must contain five native depth rasters")
    require(gauge.shape == () and scales.shape == (5,), "Expected scalar g0 and five relative endpoint scales")
    require(all(np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all()
                for value in (native,depth,gauge,scales)),
            "Nonfinite or nonfloating endpoint artifact; consistency cannot be established without exclusions")
    require(float(gauge) > 0 and (scales > 0).all(), "Endpoint g0 and relative scales must be positive")
    g0 = float(gauge)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = (scales.astype(np.float64)*g0)[:,None,None]*native.astype(np.float64)
        native_metric = g0*native.astype(np.float64)
    require(np.isfinite(expected).all() and np.isfinite(native_metric).all(),
            "Metric endpoint arithmetic is nonfinite; no pixel exclusions are permitted")
    epsilon = max(np.finfo(value.dtype).eps for value in (depth,gauge,scales))
    rtol = 64.*max(float(epsilon),np.finfo(np.float64).eps)
    atol = rtol*max(1.,float(np.max(np.abs(expected))))
    require(np.isfinite(expected).all() and np.allclose(depth,expected,rtol=rtol,atol=atol),
            "Final endpoint does not equal scales*g0*native; wrong gauge or artifact")
    return dict(native=native_metric, final=depth,
                metric_gauge=dict(g0=g0, relative_scales=scales.tolist(),
                    secondary_definition="native_metric_depth = g0 * native_depth; no per-view scale",
                    final_consistency_rtol=rtol,final_consistency_atol=atol,
                    final_consistency_max_absolute_error=float(np.max(np.abs(depth-expected))),
                    invalid_pixel_exclusions=0))


def summarize(rows, runtimes, failures):
    """Equal-view/equal-scene metrics and the predeclared primary-only nominee."""
    expected = {(scene, side, arm, frame) for scene in SCENES for side in SIDES for arm in ARMS for frame in FRAMES}
    keys = [(row["scene"], row["side"], row["arm"], row["frame"]) for row in rows]
    require(len(rows) == len(expected) and set(keys) == expected, "Missing or duplicate scored view/arm")
    lookup = dict(zip(keys, rows))
    cases, means, validity = {}, {}, {}
    for endpoint in ("final", "native"):
        case_values, directional, valid = {}, {}, {}
        for scene in SCENES:
            for side in SIDES:
                for arm in ARMS:
                    values = [lookup[scene, side, arm, frame][endpoint] for frame in FRAMES]
                    case_values[f"{scene}_{side}_{arm}"] = average_or_none([value["mae_m"] for value in values])
                    valid[f"{scene}_{side}_{arm}"] = all(value["all_samples_valid"] for value in values)
        for side in SIDES:
            directional[side] = {arm: average_or_none([case_values[f"{scene}_{side}_{arm}"] for scene in SCENES]) for arm in ARMS}
        cases[endpoint], means[endpoint], validity[endpoint] = case_values, directional, valid
    candidate_details = {}
    for arm in ("P3", "P4"):
        reasons = []
        gains = {}
        total_failures = sum(failures.get(f"{scene}_{side}_{arm}", 0) for scene in SCENES for side in SIDES)
        if total_failures:
            reasons.append("reported_numerical_failure")
        for side in SIDES:
            baseline, value = means["final"][side]["P1"], means["final"][side][arm]
            baseline_valid = all(validity["final"][f"{scene}_{side}_P1"] for scene in SCENES)
            if baseline is None or baseline <= 0 or not baseline_valid or value is None:
                gains[side] = None
                reasons.append(f"invalid_primary_metric_or_P1_control_{side}")
            else:
                gains[side] = 1. - value / baseline
                if gains[side] < .01:
                    reasons.append(f"directional_gain_below_0.01_{side}")
        if not all(validity["final"][f"{scene}_{side}_{arm}"] for scene in SCENES for side in SIDES):
            reasons.append("invalid_primary_predictions")
        improving = {}
        for side in SIDES:
            improving[side] = [scene for scene in SCENES
                if cases["final"][f"{scene}_{side}_{arm}"] is not None
                and cases["final"][f"{scene}_{side}_P1"] is not None
                and cases["final"][f"{scene}_{side}_{arm}"] < cases["final"][f"{scene}_{side}_P1"]]
        both = sorted(set(improving["earlier"]) & set(improving["later"]))
        if len(both) < 2:
            reasons.append("fewer_than_two_same_scenes_improve_both_directions")
        cost = sum(runtimes[f"{scene}_{side}_{arm}"] for scene in SCENES for side in SIDES)
        require(math.isfinite(cost) and cost >= 0, "Invalid measured candidate runtime")
        minimum = None if any(value is None for value in gains.values()) else min(gains.values())
        candidate_details[arm] = dict(eligible=not reasons, reasons=reasons, directional_gain=gains,
            improving_scenes=improving, same_scenes_improving_both=both, numerical_failed_attempts=total_failures,
            minimum_directional_gain=minimum, ranking_gain_rounded_12=None if minimum is None else round(minimum, 12),
            total_measured_seconds_six_cases=float(cost))
    eligible = [arm for arm, value in candidate_details.items() if value["eligible"]]
    eligible.sort(key=lambda arm: (-candidate_details[arm]["ranking_gain_rounded_12"],
                                   candidate_details[arm]["total_measured_seconds_six_cases"], int(arm[1:])))
    contrasts = {}
    for full, detached in (("P3", "P2"), ("P4", "P5")):
        contrasts[full+"_versus_"+detached] = {}
        for side in SIDES:
            first, second = means["final"][side][full], means["final"][side][detached]
            valid = all(validity["final"][f"{scene}_{side}_{arm}"] for scene in SCENES for arm in (full, detached))
            contrasts[full+"_versus_"+detached][side] = dict(
                valid=valid, relative_gain=None if not valid or first is None or second is None or second <= 0 else 1-first/second,
                absolute_mae_difference_m=None if first is None or second is None else first-second)
    return dict(case_mae_m=cases, directional_equal_scene_mae_m=means, case_validity=validity,
        candidates=candidate_details, nominee=eligible[0] if eligible else None,
        candidate_ranking=eligible, mechanism_contrasts=contrasts,
        primary_endpoint="final", secondary_endpoint="native",
        accuracy_improvement_established=False, publication_test_performed=False,
        mechanism_repeatability_established=False, rental_gate_passed=False)


def calculate_scores(root, seal):
    """Called only after the caller verifies evaluation registration and seal."""
    root = Path(root).resolve()
    require(seal.get("failure_policy") == FAILURE_POLICY and seal.get("failure_review_required") is False,
            "failure_review_required: unclassified failures block all study-score release")
    access = read(relative(root, ACCESS))
    require(access.get("evaluation_execution") == seal["evaluation_execution"],
            "Study arrays require a matching prospective score-access receipt")
    inputs = read(relative(root, "results/lora_profiled_v1/inputs.json"))
    packet_hashes = inputs["registration"]["packet_sha256"]
    packet_paths = {side: relative(root, f"results/mast3r_partner_shape/{side}.npz") for side in SIDES}
    require(all(sha(packet_paths[side]) == packet_hashes[side] for side in SIDES), "Source packet bytes changed")
    split_path = relative(root, "results/lora_profiled_v1/splits.npz")
    require(sha(split_path) == inputs["splits_binary_sha256"], "Spatial folds changed")
    # This is the first point where the execution path decodes study observations.
    split_arrays = verified_npz(split_path, inputs["splits_binary_sha256"])
    split = {key: value for key, value in split_arrays.items() if key.endswith("__xy")}
    packets = {}
    for side, path in packet_paths.items():
        packets[side] = verified_npz(path, packet_hashes[side])
    masks, replay = {}, {}
    for scene in SCENES:
        for side in SIDES:
            opposite = "later" if side == "earlier" else "earlier"
            for frame in FRAMES:
                prefix = f"{scene}_{side}_{frame}"
                train_xy = split[prefix+"__xy"]
                full_xy, _ = native_rows(packets[side][f"{scene}_{frame}"])
                require(set(map(tuple, train_xy.tolist())) == set(map(tuple, full_xy.tolist())),
                        "Overlap mask omits allowed adaptation observations")
                xy, depth, counts = held_mask(train_xy, packets[opposite][f"{scene}_{frame}"])
                masks[scene, side, frame] = (xy, depth, counts)
                replay[prefix+"__xy"], replay[prefix+"__measured_m"] = xy, depth
    rows, runtimes, failures, gauges = [], {}, {}, {}
    for case in seal["cases"]:
        scene, side, arm = case["scene"], case["adaptation_source"], case["arm"]
        selected = case["selected"]
        predictions = load_endpoints(root, selected)
        gauges[case["run_id"]] = predictions["metric_gauge"]
        for view, frame in enumerate(FRAMES):
            xy, measured, counts = masks[scene, side, frame]
            metrics = {}
            for endpoint in ("native", "final"):
                values = predictions[endpoint][view, xy[:, 1], xy[:, 0]].astype(np.float64)
                metrics[endpoint] = view_metric(values, measured)
                replay[f"{scene}_{side}_{frame}__{arm}_{endpoint}_m"] = values
            rows.append(dict(scene=scene, side=side, arm=arm, frame=frame, **counts, **metrics))
        runtimes[case["run_id"]] = selected["seconds"]
        failures[case["run_id"]] = case["numerical_failure_count"]
    report = dict(schema="profiled_held_source_scores_v1", rows=rows, runtimes_seconds=runtimes,
        numerical_failures_by_run_id=failures, source_packet_sha256=packet_hashes,
        all_failed_attempt_counts={case["run_id"]: case["failed_attempt_count"] for case in seal["cases"]},
        failure_review_required=False, failure_policy=FAILURE_POLICY,
        split_sha256=sha(split_path), **summarize(rows, runtimes, failures),
        endpoint_metric_gauges=gauges, secondary_metric_depth="g0*native_depth in meters, without per-view relative scales",
        aggregation="fixed point mean, equal five-view mean, equal three-scene mean; directions separate",
        finite_error_means_are_diagnostics_only=True, target_depth_labels_loaded=False,
        candidate_dependent_masks=False)
    return report, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify-registration", "score", "verify"))
    parser.add_argument("--registration", default="manifests/lora_profiled_evaluation_execution_v1.json")
    parser.add_argument("--fixtures", default="results/lora_profiled_v1/scoring_fixtures.json")
    parser.add_argument("--seal", default="results/lora_profiled_v1/prediction_seal.json")
    parser.add_argument("--output", default="results/lora_profiled_v1/held_source_scores.json")
    parser.add_argument("--samples", default="results/lora_profiled_v1/held_source_score_samples.npz")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_registration(ROOT, args.registration, args.fixtures)
        print(json.dumps(dict(registered=True, synthetic_fixtures_pass=result["synthetic_fixtures_pass"], study_arrays_loaded=False)))
        return
    identity = verify_evaluation_registration(ROOT, args.registration)
    if args.action == "verify-registration":
        print(json.dumps(dict(verified=True, evaluation=identity, study_arrays_loaded=False)))
        return
    seal = verify_seal(ROOT, args.seal, identity)
    from register_profiled_execution import verify_registry
    verify_registry(relative(ROOT, seal["common_scientific_registry"]), "scientific")
    require(seal["failure_review_required"] is False,
            "failure_review_required: resolve unclassified failures before accessing any study scores")
    access_sha256 = register_score_access(ROOT, identity, args.seal)
    report, samples = calculate_scores(ROOT, seal)
    report.update(prediction_seal=args.seal, prediction_seal_sha256=sha(relative(ROOT, args.seal)),
                  evaluation_execution=identity, evaluation_rules_sha256=seal["evaluation_rules_sha256"],
                  score_access_receipt=ACCESS, score_access_receipt_sha256=access_sha256)
    output, sample_path = relative(ROOT, args.output), relative(ROOT, args.samples)
    if args.action == "verify":
        old = read(output)
        require(sha(sample_path) == old["replay_samples_sha256"], "Compact replay sample bytes changed")
        with np.load(sample_path, allow_pickle=False) as saved:
            require(set(saved.files) == set(samples), "Replay sample inventory differs")
            for name, value in samples.items():
                np.testing.assert_array_equal(saved[name], value)
        report.update(replay_samples=args.samples, replay_samples_sha256=old["replay_samples_sha256"])
        require(report == old, "Recomputed held-source scores differ")
    else:
        require(not output.exists() and not sample_path.exists(), "Preserve existing scores and compact evidence")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("xb") as stream:
            np.savez_compressed(stream, **samples)
        report.update(replay_samples=args.samples, replay_samples_sha256=sha(sample_path))
        dump_new(output, report)
    print(json.dumps(dict(nominee=report["nominee"], primary_endpoint="final", rental_gate_passed=False)))


if __name__ == "__main__":
    main()
