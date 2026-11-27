"""Check actual renderer and matched optimizer fixtures for the local TCO study."""

import argparse
import json
import math

from replay_tco_native_stalls_quality_capacity import ROOT, STUDY, PUBLIC
from run_profiled_d0 import read, require, sha, write_new


PRIVATE = ROOT / "reports" / STUDY
STEPS = ("1", "2")
FAMILIES = {"preclip", "grad", "parameter", "exp_avg", "exp_avg_sq", "adam_step", "delta"}
STATE_FIELDS = ("anchor_sha256", "pre_state_sha256", "boundary", "source_views", "target_views",
                "pre_forward_rng_sha256", "pre_permutation_rng_sha256", "post_permutation_rng_sha256",
                "pre_backward_rng_sha256")
INPUTS = ("dataset", "scene", "frames", "width", "seed", "steps", "input_sha256", "raster_shape")


def completed(identifier):
    """Read untouched result and signed evidence, checking all consumed raw artifacts."""
    launch = PRIVATE / "launches" / identifier
    terminals = [read(path) for path in sorted(launch.glob("attempt*/terminal.json"))]
    successful = [row for row in terminals if row["verdict"] == "completed"]
    if not successful:
        return None
    require(len(successful) == 1, f"Ambiguous fixture completion for {identifier}")
    attempt = successful[0]
    folder = PRIVATE / "cases" / identifier / attempt["attempt"] / "case"
    receipt = folder / "complete.json"
    require(attempt["job"] == identifier and sha(receipt) == attempt["complete_sha256"]
            and attempt["stopped"] and attempt["exit_code"] == 0, f"Invalid fixture receipt for {identifier}")
    result = read(receipt)
    require(result["id"] == identifier and result["rental_compute_usd"] == 0, f"Changed fixture {identifier}")
    for name, digest in result.get("artifact_sha256", {}).items():
        artifact = folder / name
        require(artifact.parent == folder and artifact.is_file() and sha(artifact) == digest,
                f"Fixture artifact changed for {identifier}/{name}")
    return dict(directory=folder, result=result, digest=attempt["complete_sha256"])


def get_summary(case):
    filename = "gradient_summary.json"
    require(filename in case["result"]["artifact_sha256"], "Missing registered gradient summary")
    return read(case["directory"] / filename)


def native_work(result):
    audit = result["audit"]
    return (result["lora_tensors"] == 384 and result["updates"] == 2 and
            result["native_depth_batches"] == [[5]] and result["head_taps"] == [4, 11, 17, 23] and
            audit["frozen_start_sha256"] == audit["frozen_end_sha256"])


def paired_checks(read_case, candidate, reference, anchor, negative=False):
    from tco_five_view_gradients import compare_cases

    bundle = [read_case(name) for name in (candidate, reference, anchor)]
    if any(row is None for row in bundle):
        return dict(passed=False, reason="missing_fixture")
    actual, control, captured = (row["result"] for row in bundle)
    a, b, c = map(get_summary, bundle)
    controls = dict(
        same_inputs=all(actual[key] == control[key] == captured[key] for key in INPUTS),
        original_capture=c["mode"] == "capture" and all(row["signed_total_loss"] for row in (a, b, c)),
        correct_mode=b["mode"] == "replay" and a["mode"] == ("negative" if negative else "replay"),
        original_signed_q=all(row.get("source_job") == anchor and not row.get("q_source_job")
                              for row in (actual, control)),
        backend=all(row["backend"]["deterministic"] and row["backend"]["cublas"] == ":4096:8" for row in (a, b, c)),
        model=all(native_work(row) for row in (actual, control, captured)) and
        actual["audit"]["frozen_start_sha256"] == control["audit"]["frozen_start_sha256"] ==
        captured["audit"]["frozen_start_sha256"],
    )
    for step in STEPS:
        x, y, z = (row["matched_updates"][step] for row in (a, b, c))
        controls[f"state_and_forward_{step}"] = all(x[field] == y[field] == z[field] for field in STATE_FIELDS)
        original_q = c["raw_artifact_sha256"][f"q_step{step}.pt"]
        controls[f"q_source_{step}"] = x["q_artifact_sha256"] == y["q_artifact_sha256"] == original_q
        controls[f"derivative_{step}"] = (
            bool(x.get("flipped")) and y["derivatives"] == z["derivatives"] if negative
            else x["derivatives"] == y["derivatives"] == z["derivatives"]
        )
    comparison = compare_cases(*(row["directory"] for row in bundle))
    complete = all(set(values) == FAMILIES and all(row["count"] == 384 for row in values.values())
                   for values in comparison.values())
    controls["inventory"] = set(comparison) == set(STEPS) and complete
    passed = all(controls.values()) and all(
        not rows["preclip"]["passed"] if negative else all(row["passed"] for row in rows.values())
        for rows in comparison.values()
    )
    final_count = actual.get("optimizer", {}).get("final_gradients")
    control_count = control.get("optimizer", {}).get("final_gradients")
    controls["final_gradient_availability"] = final_count == control_count == 384 and (
        actual["optimizer"]["gradients_on_return"] == control["optimizer"]["gradients_on_return"] == 384
    )
    return dict(passed=passed and controls["final_gradient_availability"], controls=controls, comparison=comparison)


def same_prediction(read_case, left, right, dummy_reference=False):
    pair = read_case(left), read_case(right)
    if any(row is None for row in pair):
        return False
    first, second = (row["result"] for row in pair)
    return (all(first.get(field) == second.get(field) for field in
                ("input_sha256", "output_sha256", "updates", "native_depth_batches", "losses"))
            and (first["reference_sha256"] != second["reference_sha256"]
                 if dummy_reference else first["reference_sha256"] == second["reference_sha256"])
            and first["audit"]["adapter_final"] == second["audit"]["adapter_final"])


def production_spread(read_case):
    from check_tco_render_lifetime import compare_factors

    names = ("qual_production_native_a", "qual_production_native_b", "qual_production_camera")
    rows = [read_case(name) for name in names]
    report = {}
    for index, key in ((1, "native_repeat"), (2, "checkpoint_vs_native")):
        a, b = rows[index], rows[0]
        if a is None or b is None:
            report[key] = dict(reason="missing_fixture")
            continue
        one, two = a["result"], b["result"]
        if any(one[field] != two[field] for field in INPUTS):
            raise ValueError(f"Production qualification used mismatched registered inputs in {key}")
        report[key] = dict(comparison=compare_factors(a["directory"], b["directory"], fixed=False),
                           same_forward=[x == y for x, y in zip(one["audit"]["update_predictions"],
                                                               two["audit"]["update_predictions"])])
    return report


def common_gates(result, stage):
    if stage == "quality":
        return {"camera": all(result["checks"].values())}
    shared = all(result["checks"].values())
    return {name: shared and result["renderer_qualified"].get(name, False) and
            result["matched_optimizer"][name]["passed"] for name in ("camera", "n", "g", "ng")}


def wrong_moments_fixture():
    """Confirm the numeric verifier rejects resetting the moments between Adam updates."""
    import torch

    def step_twice(corrupt):
        factor = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
        optimizer = torch.optim.Adam([factor], lr=0.0003)
        moments = []
        for index in range(2):
            if index and corrupt:
                optimizer.state[factor]["exp_avg"].zero_()
                optimizer.state[factor]["exp_avg_sq"].zero_()
            optimizer.zero_grad(set_to_none=True)
            loss = factor.square().sum()
            loss.backward()
            optimizer.step()
            moments.append({key: value.detach().clone() for key, value in optimizer.state[factor].items()})
        return moments

    from tco_five_view_gradients import array_stats

    intact, broken = step_twice(False), step_twice(True)
    first_update_same = all(torch.equal(intact[0][field], broken[0][field]) for field in intact[0])
    counter_same = torch.equal(intact[1]["step"], broken[1]["step"])
    comparisons = {name: array_stats(broken[1][name], intact[1][name]) for name in ("exp_avg", "exp_avg_sq")}
    return dict(passed=first_update_same and counter_same and any(not row["passed"] for row in comparisons.values()),
                first_update_same=first_update_same, counter_same=counter_same, second_update=comparisons)


def qualify(stage):
    cache = {}

    def get(identifier):
        if identifier not in cache:
            cache[identifier] = completed(identifier)
        return cache[identifier]

    report = dict(study=STUDY, stage=stage, checks={}, rental_compute_usd=0)
    if stage == "quality":
        fixture = get("qual_camera_head")
        camera = fixture["result"]["fixture"] if fixture else {}
        required = {"checkpoint/4/aux0/inference0", "checkpoint/1/aux0/inference0",
                    "checkpoint/4/aux1/inference0", "checkpoint/4/aux0/inference1"}
        report["checks"]["camera_head_fixture"] = (camera.get("detached_last_token_detected") is True
            and required.issubset(camera.get("cases", {})) and
            all(part["pass_tolerance"] for name in required for part in camera["cases"][name]["values"]))
        report["checks"]["zero_update_identity"] = same_prediction(get, "qual_native_identity", "qual_camera_identity")
        case = ("qual_camera", "qual_native_replay", "qual_anchor")
        report["deterministic"] = paired_checks(get, *case)
        report["checks"]["matched_actual_signed_gradient"] = report["deterministic"]["passed"]
        report["negative"] = paired_checks(get, "qual_negative", "qual_native_replay", "qual_anchor", negative=True)
        report["checks"]["wrong_sign_detected"] = report["negative"]["passed"]
        report["checks"]["dummy_reference_invariance"] = same_prediction(
            get, "qual_leak_base", "qual_leak_dummy", dummy_reference=True
        )
        report["production"] = production_spread(get)
    else:
        cpu = get("qual_cpu")
        renderer = get("qual_renderer")
        fixture = renderer["result"]["renderer"] if renderer else {}
        anchor = get("qual_capacity_anchor")
        report["moment_negative"] = wrong_moments_fixture()
        report["checks"] = dict(
            cpu_fixture=bool(cpu and cpu["result"]["fixture"]["passed"]),
            clearing_moments_detected=report["moment_negative"]["passed"],
            detached_opacity_detected=bool(len(fixture.get("real", [])) == 2 and
                                          all(row["negative_detected"] for row in fixture["real"])),
            matched_renderer_capture=bool(anchor and renderer and
                renderer["result"]["source_completion_sha256"] == anchor["digest"] and
                [row["capture_sha256"] for row in fixture["real"]] ==
                [anchor["result"]["artifact_sha256"][f"renderer_inputs_step{step}.pt"] for step in STEPS]),
        )
        report["renderer_qualified"] = fixture.get("qualified", {})
        report["matched_optimizer"] = {
            policy: paired_checks(get, f"qual_capacity_{policy}", "qual_capacity_camera", "qual_capacity_anchor")
            for policy in ("camera", "n", "g", "ng")
        }
        for policy in ("g", "ng"):
            fixture = get(f"qual_capacity_{policy}")
            evidence = fixture["result"].get("optimizer", {}) if fixture else {}
            row = report["matched_optimizer"][policy]
            row["gradient_lifetime"] = evidence.get("pre_forward_gradients")
            row["passed"] = row["passed"] and row["gradient_lifetime"] == [0, 0]
    report["qualified"] = common_gates(report, stage)
    report["receipt_sha256"] = {identifier: row["digest"] for identifier, row in sorted(cache.items()) if row}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("quality", "capacity"))
    args = parser.parse_args()
    result = qualify(args.stage)
    name = "quality_qualification.json" if args.stage == "quality" else "qualification.json"
    output = PUBLIC / name
    if output.exists():
        require(read(output) == result, f"Existing {name} differs from the verified fixtures")
    else:
        write_new(output, result)
    print(json.dumps(result["qualified"], sort_keys=True))


if __name__ == "__main__":
    main()
