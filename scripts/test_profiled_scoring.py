"""Synthetic-only adversarial fixtures for prospective sealing and scoring.

No project observations, prediction arrays, model, GPU, or target-depth labels
are opened. Temporary synthetic files exercise the same seal audit used in the
study. The default CLI prints evidence and writes no repository artifact.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from seal_profiled_predictions import (SCENES, SIDES, ARMS, RULES, FAILURE_POLICY, build_seal,
    dump_new, inventory, read, sha, verify_seal)
from score_profiled_lora import (ACCESS, FRAMES, INPUTS, SOURCES, calculate_scores, dependencies, held_mask,
    load_endpoints, native_rows, prepare_registration, register_score_access, source_hashes, summarize,
    verify_evaluation_registration, view_metric)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


@contextmanager
def changed(path, value):
    before = path.read_bytes()
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        write(path, value)
    try:
        yield
    finally:
        path.write_bytes(before)


def fixture_tree(root):
    (root / "source.py").write_text("# synthetic source only\n", encoding="utf-8")
    write(root / "input.json", {"synthetic": True})
    write(root / RULES, {"schema": "synthetic fixture rules; not a study registration"})
    write(root / "evaluation.json", {"synthetic": True})
    jobs = []
    for scene in SCENES:
        for side in SIDES:
            for arm in ARMS:
                run_id = f"{scene}_{side}_{arm}"
                jobs.append(dict(run_id=run_id, scene=scene, adaptation_source=side, arm=arm,
                    adapt_lora=arm != "P0", optimizer_steps=0 if arm == "P0" else 8,
                    public_attempt_parent=f"public/{run_id}", raw_attempt_parent=f"raw/{run_id}"))
    registry = dict(phase="scientific", scientific_ready_at_registration=True, jobs=jobs,
        source_sha256_lf={"source.py": sha(root / "source.py", text=True)},
        artifact_sha256={name: sha(root / name) for name in ("input.json", RULES)},
        input_sha256={"input.json": sha(root / "input.json")}, checkpoint_sha256="a"*64,
        docker_image="sha256:"+"b"*64, runtime_policy={"synthetic_arithmetic_policy": True})
    write(root / "registry.json", registry)
    identity = dict(file="evaluation.json", sha256=sha(root / "evaluation.json"))
    start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    for index, job in enumerate(jobs):
        make_attempt(root, job, registry, "attempt01", start+timedelta(days=index))
    return registry, identity


def make_attempt(root, job, registry, attempt, start, failed=False):
    public, raw = root / job["public_attempt_parent"] / attempt, root / job["raw_attempt_parent"] / attempt
    public.mkdir(parents=True)
    (raw / "endpoint").mkdir(parents=True)
    runtime = dict(run_id=job["run_id"], attempt=attempt, job=job, manifest="registry.json",
        manifest_sha256=sha(root / "registry.json"), source_sha256_lf=registry["source_sha256_lf"],
        artifact_sha256=registry["artifact_sha256"], input_sha256=registry["input_sha256"],
        checkpoint_sha256=registry["checkpoint_sha256"], docker_image=registry["docker_image"],
        runtime_policy=registry["runtime_policy"],
        public_attempt=public.relative_to(root).as_posix(), raw_attempt=raw.relative_to(root).as_posix(),
        started_at_utc=start.isoformat(), accuracy_scoring_performed=False,
        held_source_development_arrays_loaded=False, target_depth_arrays_loaded=False)
    write(public / "started.json", runtime)
    if failed:
        write(public / "failure.json", dict(runtime, pass_all=False, error_type="SyntheticFailure",
            message="Deliberate fixture failure", failed_at_utc=(start+timedelta(minutes=5)).isoformat()))
        return public, raw
    for endpoint in ("native", "final"):
        # Invalid NPZ bytes are intentional: the sealer must never decode arrays.
        (raw / "endpoint" / (endpoint+".npz")).write_bytes((job["run_id"]+attempt+endpoint).encode())
    if job["adapt_lora"]:
        (raw / "adapters.npz").write_bytes(b"synthetic adapter bytes")
    steps = [dict(step=i, optimizer_steps=1, optimizer_updates_cumulative=i+1,
                  graph_free_decoder_calls=20, differentiable_decoder_calls=20)
             for i in range(job["optimizer_steps"])]
    for index, row in enumerate(steps):
        write(public / f"step_{index:02d}.json", row)
    endpoint = {key: value for kind in ("native", "final") for key, value in (
        (kind+"_file", (raw / "endpoint" / (kind+".npz")).relative_to(root).as_posix()),
        (kind+"_sha256", sha(raw / "endpoint" / (kind+".npz"))))}
    endpoint["capture"] = dict(prior_cache_files=0, pairs=20, coarse_iterations=300,
                               fine_iterations=300, native_size=[384,512])
    complete = dict(runtime, pass_all=True, status="predictions_sealed_without_accuracy_scoring",
        original_state_unchanged=True, original_state_sha256="c"*64,
        pre_endpoint_original_state_sha256="c"*64, final_original_state_sha256="c"*64,
        optimizer_updates=job["optimizer_steps"], steps=steps, seconds=10., endpoint=endpoint,
        adapter_module_count=144 if job["adapt_lora"] else 0,
        trainable_parameters=1032192 if job["adapt_lora"] else 0,
        adapter_final_sha256="d"*64 if job["adapt_lora"] else None,
        raw_files=inventory(raw), public_files=inventory(public),
        completed_at_utc=(start+timedelta(minutes=5)).isoformat())
    write(public / "complete.json", complete)
    return public, raw


def remove_fixture(path, root):
    assert path.resolve().is_relative_to(root.resolve()) and path.resolve() != root.resolve()
    shutil.rmtree(path)


def metric_rows(error=.9):
    rows = []
    runtimes, failures = {}, {}
    for scene in SCENES:
        for side in SIDES:
            for arm in ARMS:
                run_id = f"{scene}_{side}_{arm}"
                runtimes[run_id], failures[run_id] = 1., 0
                for frame in FRAMES:
                    value = error if arm in ("P3", "P4") else 1.
                    metric = view_metric(np.asarray([1.+value]), np.asarray([1.]))
                    rows.append(dict(scene=scene, side=side, arm=arm, frame=frame,
                                     final=dict(metric), native=dict(metric)))
    return rows, runtimes, failures


def set_error(rows, arm, value, scene=None, side=None, endpoint="final"):
    for row in rows:
        if row["arm"] == arm and (scene is None or row["scene"] == scene) and (side is None or row["side"] == side):
            row[endpoint] = view_metric(np.asarray([1.+value]), np.asarray([1.]))


def run_fixtures():
    checks = []

    def passed(name, condition=True):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, action, message=None):
        try:
            action()
        except (ValueError, OSError) as error:
            if message is not None and message not in str(error):
                raise AssertionError(f"Wrong rejection for {name}: {error}") from error
            passed(name)
        else:
            raise AssertionError("Expected rejection: "+name)

    with tempfile.TemporaryDirectory(prefix="profiled-scoring-fixture-") as temporary:
        root = Path(temporary).resolve()
        registry, evaluation = fixture_tree(root)
        seal = lambda: build_seal(root, "registry.json", evaluation)
        baseline = seal()
        passed("complete_36_state_inventory_and_explicit_frozen_P0", baseline["completed_states"] == 36 and
               baseline["cases"][0]["selected"]["adapter"]["frozen_P0"] is True)
        passed("seal_never_decodes_prediction_arrays", baseline["prediction_arrays_deserialized"] is False)
        write(root / "seal.json", baseline)
        passed("unchanged_global_seal_verifies", verify_seal(root, "seal.json", evaluation) == baseline)
        altered = dict(registry, jobs=registry["jobs"][:-1])
        with changed(root / "registry.json", altered):
            rejects("missing_registered_run_rejected", seal)
        altered = dict(registry, jobs=[*registry["jobs"][:-1], registry["jobs"][0]])
        with changed(root / "registry.json", altered):
            rejects("duplicate_registered_run_rejected", seal)
        first = registry["jobs"][0]
        public = root / first["public_attempt_parent"] / "attempt01"
        raw = root / first["raw_attempt_parent"] / "attempt01"
        with changed(raw / "endpoint/final.npz", b"tampered output"):
            rejects("changed_endpoint_bytes_rejected", seal)
        record = read(public / "complete.json")
        with changed(public / "complete.json", dict(record, manifest_sha256="e"*64)):
            rejects("mixed_registry_completion_rejected", seal)
        with changed(public / "complete.json", dict(record, runtime_policy={"wrong_policy": True})):
            rejects("mixed_arithmetic_runtime_policy_rejected", seal)
        with changed(public / "complete.json", dict(record, final_original_state_sha256="f"*64)):
            rejects("changed_frozen_backbone_identity_rejected", seal)
        with changed(public / "complete.json", dict(record, native_file="wrong.npz")):
            # The authoritative nested endpoint is unchanged; an irrelevant field
            # changes the seal hash, so verification must reject the old seal.
            rejects("changed_completion_bytes_break_global_seal", lambda: verify_seal(root, "seal.json", evaluation))
        write(public / "failure.json", {"unexpected": True})
        rejects("unreported_conflicting_failure_rejected", seal)
        (public / "failure.json").unlink()
        partial_public = root / first["public_attempt_parent"] / "partial"
        partial_raw = root / first["raw_attempt_parent"] / "partial"
        partial_public.mkdir()
        partial_raw.mkdir()
        rejects("partial_attempt_is_not_silently_skipped", seal)
        remove_fixture(partial_public, root)
        remove_fixture(partial_raw, root)
        start = datetime(2026,1,1,8,tzinfo=timezone.utc)
        failed = make_attempt(root, first, registry, "failed_earlier", start, failed=True)
        earlier = make_attempt(root, first, registry, "z_earlier_valid", start+timedelta(hours=1))
        selected = seal()["cases"][0]
        passed("UTC_first_valid_selection_not_lexical_or_mtime", selected["selected_attempt"] == "z_earlier_valid")
        passed("reported_failures_are_preserved", selected["failed_attempt_count"] == 1 and len(selected["attempts"]) == 3)
        passed("unknown_failure_requires_engineering_review_not_negative_science", seal()["failure_review_required"] is True and
               selected["unclassified_failure_count"] == 1 and selected["numerical_failure_count"] == 0)
        rejects("unknown_failure_blocks_scores_before_study_array_access", lambda: calculate_scores(root, seal()),
                message="failure_review_required")
        overlap = make_attempt(root, first, registry, "overlap", start+timedelta(hours=1,minutes=4))
        rejects("overlapping_attempt_intervals_rejected", seal)
        for path in (*overlap, *earlier, *failed):
            remove_fixture(path, root)
        with changed(root / "evaluation.json", {"changed": True}):
            rejects("changed_evaluator_registration_rejected", seal)
        altered = dict(baseline, cases=baseline["cases"][:-1])
        with changed(root / "seal.json", altered):
            rejects("missing_global_seal_case_rejected", lambda: verify_seal(root, "seal.json", evaluation))
        trained = registry["jobs"][1]
        trace = root / trained["public_attempt_parent"] / "attempt01/step_00.json"
        with changed(trace, {"wrong": "trace"}):
            rejects("trace_mismatch_rejected", seal)
        selected = {}
        native = np.full((5,384,512),1.1,dtype=np.float32)
        endpoint_arrays = dict(native=dict(depth=native),
            final=dict(depth=3.*native.astype(np.float64),g0=np.asarray(2.,dtype=np.float64),scales=np.full(5,1.5)))
        for endpoint, arrays in endpoint_arrays.items():
            path = root / ("numeric_"+endpoint+".npz")
            np.savez_compressed(path, **arrays)
            selected[endpoint+"_file"], selected[endpoint+"_sha256"] = path.name, sha(path)
        decoded = load_endpoints(root, selected)
        passed("native_and_final_file_roles_not_swapped", float(decoded["native"][0,0,0]) <
               float(decoded["final"][0,0,0]) and np.isclose(decoded["final"][0,0,0],3.3))
        passed("native_secondary_uses_common_metric_g0_without_view_scale", np.isclose(decoded["native"][0,0,0],2.2))
        with changed(root / selected["final_file"], b"wrong endpoint bytes"):
            rejects("endpoint_decoder_checks_the_exact_bytes", lambda: load_endpoints(root,selected))
        final_path = root / selected["final_file"]
        original_final = final_path.read_bytes()
        for field, value, name in (
            ("g0",np.asarray(1.),"wrong_metric_gauge_rejected"),
            ("g0",np.asarray(0.),"zero_metric_gauge_rejected"),
            ("g0",np.asarray(np.nan),"nonfinite_metric_gauge_rejected"),
            ("scales",np.asarray([1.5,1.5,0.,1.5,1.5]),"nonpositive_endpoint_scale_rejected"),
            ("scales",np.ones(4),"wrong_scale_shape_rejected")):
            bad = dict(endpoint_arrays["final"])
            bad[field] = value
            np.savez_compressed(final_path,**bad)
            altered = dict(selected,final_sha256=sha(final_path))
            rejects(name,lambda: load_endpoints(root,altered))
        final_path.write_bytes(original_final)
        bad = dict(endpoint_arrays["final"])
        bad["depth"] = bad["depth"].copy()
        bad["depth"][0,0,0] = np.nan
        np.savez_compressed(final_path, **bad)
        rejects("nonfinite_endpoint_rejected_without_pixel_exclusions", lambda: load_endpoints(root,
                dict(selected,final_sha256=sha(final_path))),message="Nonfinite")
        final_path.write_bytes(original_final)
        # Independent registration-integrity fixture. All referenced sources and
        # packet bytes below are dummy synthetic files, never real study data.
        for name in SOURCES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic registration dependency\n", encoding="utf-8")
        for name in INPUTS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic input bytes; not an array archive")
        rules = dict(schema="profiled_evaluation_rules_v1", candidate_arms=["P3","P4"],
            control_arms=["P0","P1","P2","P5"], nomination=dict(minimum_directional_gain_each=.01,
            minimum_same_scenes_improving_both_directions=2,numerical_failures_allowed=0,max_candidates=1),
            global_prediction_seal=dict(required_states=36), held_source_metric=dict(candidate_independent_mask=True))
        write(root / RULES, rules)
        sources = source_hashes(root)
        evidence = dict(pass_all=True,synthetic_only=True,source_sha256_lf=sources,dependencies=dependencies(),
                        checks_passed=36,checks=[f"synthetic_metadata_check_{i}" for i in range(36)])
        write(root / "fixture_evidence.json", evidence)
        packet_hashes = {}
        for side in SIDES:
            name = f"results/mast3r_partner_shape/{side}.npz"
            path = root / name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b"synthetic packet bytes; not an array archive")
            packet_hashes[name] = sha(path)
        registration = dict(schema="profiled_evaluation_execution_v1",synthetic_fixtures_pass=True,
            study_arrays_loaded=False,study_scores_read=False,prepared_at_utc="2025-12-01T00:00:00+00:00",
            source_sha256_lf=sources,dependencies=dependencies(),failure_policy=FAILURE_POLICY,
            rules=RULES,rules_sha256=sha(root/RULES), input_sha256={name:sha(root/name) for name in INPUTS},
            packet_sha256=packet_hashes, fixtures="fixture_evidence.json",fixtures_sha256=sha(root/"fixture_evidence.json"))
        write(root / "evaluation_exec.json", registration)
        identity = verify_evaluation_registration(root,"evaluation_exec.json")
        rejects("missing_evaluation_registration_blocks_access", lambda: verify_evaluation_registration(root,"missing.json"))
        passed("prospective_evaluation_registration_integrity", identity["sha256"] == sha(root/"evaluation_exec.json"))
        with changed(root / "fixture_evidence.json", dict(evidence,pass_all=False)):
            rejects("changed_fixture_evidence_blocks_score_access", lambda: verify_evaluation_registration(root,"evaluation_exec.json"))
        with changed(root / SOURCES[0], b"changed scoring source"):
            rejects("changed_scorer_source_blocks_score_access", lambda: verify_evaluation_registration(root,"evaluation_exec.json"))
        with changed(root / "evaluation_exec.json", dict(registration,dependencies={"numpy":"wrong"})):
            rejects("changed_dependency_version_blocks_score_access", lambda: verify_evaluation_registration(root,"evaluation_exec.json"))
        receipt = register_score_access(root,identity,"seal.json")
        passed("prospective_score_access_receipt_replays", register_score_access(root,identity,"seal.json") == receipt)
        rejects("score_access_cannot_switch_registrations", lambda: register_score_access(root,evaluation,"seal.json"))
        rejects("post_access_evaluation_reregistration_rejected", lambda: prepare_registration(root,"future_eval.json","future_fixture.json"),
                message="already began")

    native_xy = np.asarray([[10,10], [11,11], [11,11], [12,12], [13,13], [-20,20]], dtype=np.float64)
    packet_xy = (native_xy+.5)*[518/512,392/384]-.5
    packet = np.column_stack((packet_xy, [3., 5., 1., 2., 4., 1.]))
    xy, depth, counts = held_mask(np.asarray([[10,10],[12,12]], dtype=np.int64), packet)
    passed("half_pixel_mapping_nearest_depth_and_both_fold_overlap", xy.tolist() == [[11,11],[13,13]] and
           depth.tolist() == [1.,4.] and counts == dict(mapped_opposite_count=4, overlapping_excluded=2,
                                                       held_count=2, adaptation_count=2))
    rejects("empty_held_view_rejected", lambda: held_mask(np.asarray([[10,10]],dtype=np.int64), packet[:1]))
    negative = view_metric(np.asarray([-1., 2.]), np.ones(2))
    passed("nonpositive_predictions_kept_in_error_not_masked", negative["mae_m"] == 1.5 and
           negative["nonpositive_count"] == 1 and negative["candidate_dependent_exclusions"] == 0)
    nonfinite = view_metric(np.asarray([np.nan, 2.]), np.ones(2))
    passed("nonfinite_mean_not_replaced_by_finite_subset", nonfinite["mae_m"] is None and
           nonfinite["finite_error_mae_m"] == 1. and nonfinite["finite_error_count"] == 1)
    rows, times, failures = metric_rows()
    rejects("missing_scored_view_rejected", lambda: summarize(rows[:-1], times, failures))
    rejects("duplicated_scored_view_rejected", lambda: summarize([*rows[:-1],rows[0]], times, failures))
    for row in rows:
        if row["scene"] == "heads" and row["side"] == "earlier" and row["arm"] == "P3":
            index = FRAMES.index(row["frame"])
            error, count = (0.,1) if index == 0 else ((2.,100) if index == 1 else (1.,3))
            row["final"] = view_metric(np.full(count, 1.+error), np.ones(count))
    summary = summarize(rows,times,failures)
    passed("equal_view_weighting_despite_unequal_point_counts", summary["case_mae_m"]["final"]["heads_earlier_P3"] == 1.)
    passed("equal_scene_weighting", abs(summary["directional_equal_scene_mae_m"]["final"]["earlier"]["P3"]-2.8/3) < 1e-14)
    rows,times,failures = metric_rows()
    set_error(rows,"P4",1.2)
    for side, scene in (("earlier","fire"),("later","heads")):
        set_error(rows,"P3",.5,side=side)
        set_error(rows,"P3",1.5,side=side,scene=scene)
    summary = summarize(rows,times,failures)
    passed("no_scene_cherrypicking_across_directions", summary["nominee"] is None and
           summary["candidates"]["P3"]["same_scenes_improving_both"] == ["chess"])
    rows,times,failures = metric_rows()
    for key in times:
        if key.endswith("_P3"):
            times[key] = 2.
    passed("runtime_breaks_gain_tie", summarize(rows,times,failures)["nominee"] == "P4")
    times = {key:1. for key in times}
    passed("numeric_arm_breaks_gain_and_runtime_tie", summarize(rows,times,failures)["nominee"] == "P3")
    set_error(rows,"P3",1.-.10000000000004)
    set_error(rows,"P4",1.-.10000000000001)
    times = {key:(2. if key.endswith("_P3") else 1.) for key in times}
    passed("ranking_uses_declared_12_decimal_precision", summarize(rows,times,failures)["nominee"] == "P4")
    set_error(rows,"P3",1.-.100000000002)
    passed("gain_beyond_rounding_beats_runtime", summarize(rows,times,failures)["nominee"] == "P3")
    rows,times,failures = metric_rows()
    failures["heads_earlier_P3"] = 1
    passed("failed_attempt_disqualifies_even_after_valid_retry", summarize(rows,times,failures)["nominee"] == "P4")
    rows,times,failures = metric_rows()
    set_error(rows,"P4",1.2)
    chosen = next(row for row in rows if row["arm"] == "P3")
    chosen["final"] = nonfinite
    passed("nonfinite_candidate_disqualified", summarize(rows,times,failures)["nominee"] is None)
    chosen["final"] = negative
    passed("nonpositive_candidate_disqualified", summarize(rows,times,failures)["nominee"] is None)
    rows,times,failures = metric_rows()
    for arm in ("P3","P4"):
        set_error(rows,arm,1.1)
        set_error(rows,arm,.1,endpoint="native")
    passed("wrong_secondary_endpoint_cannot_nominate", summarize(rows,times,failures)["nominee"] is None)
    set_error(rows,"P2",.01)
    passed("detached_control_cannot_be_renamed_candidate", summarize(rows,times,failures)["nominee"] is None)
    summary = summarize(*metric_rows())
    passed("both_required_mechanism_contrasts_reported", set(summary["mechanism_contrasts"]) ==
           {"P3_versus_P2","P4_versus_P5"} and summary["mechanism_repeatability_established"] is False)
    return dict(schema="profiled_scoring_fixtures_v1", synthetic_only=True,
                checks=checks, checks_passed=len(checks), pass_all=True,
                real_study_arrays_loaded=False, gpu_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional new fixture artifact; refuses overwrite")
    args = parser.parse_args()
    result = run_fixtures()
    if args.report:
        dump_new(args.report, result)
    print(json.dumps(result, indent=2))
