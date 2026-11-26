"""Prospective secondary qualification on the deterministic backend's own real-loss anchor.

Uses private tensors on the CPU. The standard production qualification remains an
independent result; this verifier neither edits it nor changes its tolerances.
"""

import argparse
from pathlib import Path

from run_profiled_d0 import read, sha, write_new
from summarize_tco_five_view_gradients import PUBLIC, STUDY, camera_vjp, model_comparison, receipt, record


ANCHOR = "grad_deterministic_capture"
REFERENCE = "grad_deterministic_fallback_fixed_1"
POLICIES = {"screen": ("camera", "last"),
            "development": ("retention", "combined", "uniform", "camera_retention")}
PRODUCTION = {"screen": "qualification.json", "development": "retention_qualification.json"}
FAMILIES = {"preclip", "grad", "parameter", "exp_avg", "exp_avg_sq", "adam_step", "delta"}
CONFIG = ("camera", "retention", "sites", "operation")
INPUTS = ("dataset", "scene", "frames", "width", "seed", "steps", "input_sha256")


def backend_valid(summary):
    backend = summary["backend"]
    return backend["deterministic"] is True and backend["cublas"] == ":4096:8"


def same_input(left, right):
    return all(left[field] == right[field] for field in INPUTS)


def paired_check(candidate, ref=REFERENCE, anchor=ANCHOR):
    """Review the complete, identical signed q and state before any tolerance assessment."""
    labels = (candidate, ref, anchor)
    actual, native, captured = (record(name) for name in labels)
    if any(item is None for item in (actual,native,captured)):
        return dict(status="unavailable", passed=False)
    sources = {name: receipt(name)[2] for name in labels}
    summaries = [receipt(name)[1] for name in labels]
    if any(value is None for value in summaries):
        return dict(status="missing_gradient_receipt", passed=False, sources=sources)
    correct_backend = all(backend_valid(summary) for summary in summaries)
    controls = dict(deterministic_backend=correct_backend,
                    source_capture=summaries[2]["mode"] == "capture",
                    original_signed_derivative=all(row["gradient_mode"] == "replay"
                        and row.get("source_job") == anchor and not row.get("q_source_job") for row in (actual,native)),
                    same_inputs=same_input(actual,native) and same_input(native,captured),
                    same_frozen_model=len({row["audit"]["frozen_start_sha256"] for row in
                        (actual,native,captured)}) == 1 and all(row["audit"]["frozen_start_sha256"] ==
                        row["audit"]["frozen_end_sha256"] for row in (actual,native,captured)),
                    full_inventory=all(row["updates"]==2 and row["lora_tensors"]==384 and
                        row["native_depth_batches"]==[[5]] and row["head_taps"]==[4,11,17,23]
                        for row in (actual,native,captured)))
    for step in ("1", "2"):
        rows = [s["matched_updates"][step] for s in summaries]
        for field in ("anchor_sha256", "pre_state_sha256", "boundary", "derivatives", "source_views",
                      "target_views", "pre_forward_rng_sha256", "pre_permutation_rng_sha256",
                      "post_permutation_rng_sha256", "pre_backward_rng_sha256"):
            controls[f"update_{step}_{field}"] = rows[0][field] == rows[1][field] == rows[2][field]
        controls[f"update_{step}_captured_q_sha256"] = (rows[0]["q_artifact_sha256"]
            == rows[1]["q_artifact_sha256"] == summaries[2]["raw_artifact_sha256"][f"q_step{step}.pt"])
    if not all(controls.values()):
        return dict(status="mismatched_control", passed=False, controls=controls, sources=sources)
    comp = model_comparison(candidate,ref,anchor)
    steps, exact, passed = {}, True, True
    for step in ("1","2"):
        cam = camera_vjp(candidate,ref,int(step))
        families = comp[step]
        if set(families) != FAMILIES or any(row["count"] != 384 for row in families.values()):
            raise ValueError(f"Incomplete native optimizer comparison in {candidate}, update {step}")
        strict = (cam["same_tokens"] and cam["signed_gradient"]["passed"]
                  and all(row["passed"] for row in families.values()))
        identical = strict and cam["signed_gradient"]["diff_norm"] == 0 and all(
            row["diff_norm"] == 0 for row in families.values())
        steps[step] = dict(camera_tokens=cam, loRA_adam=families, passed=strict, exact=identical)
        passed = passed and strict
        exact = exact and identical
    status = "exact" if exact else ("within_registered_tolerances" if passed else "outside_tolerance")
    return dict(status=status, passed=passed, exact=exact, updates=steps, controls=controls, sources=sources)


def supplement(stage):
    if stage == "initial":
        stage = "screen"
    if stage not in POLICIES:
        raise ValueError(f"Unsupported supplement stage: {stage}")
    path = PUBLIC / PRODUCTION[stage]
    production = read(path)
    if production["study"] != STUDY:
        raise ValueError("Wrong production qualification authority")
    out = dict(study=STUDY, stage=stage, backend="deterministic",
               reference=REFERENCE, anchor=ANCHOR, tolerance=dict(atol=1e-6, gradient_rtol=1e-4),
               original_production_sha256=sha(path), original_production_eligible=production["eligible"],
               eligible=list(production["eligible"]), newly_supported=[], policy_checks={}, rental_compute_usd=0)
    for policy in POLICIES[stage]:
        registered = production["policy_checks"].get(policy, {"status":"unavailable"})
        candidate = "det_match_" + policy
        current = record(candidate)
        previous = record(registered["matched_job"]) if registered.get("matched_job") else None
        row = out["policy_checks"][policy] = dict(production_status=registered["status"],
                                                  candidate=candidate, additional_route=False)
        if not current or not previous:
            row["status"] = "unavailable"
            continue
        row["production_structural_controls"] = dict(registered.get("controls",{}))
        row["same_policy_configuration"] = all(current.get(key) == previous.get(key) for key in CONFIG)
        row["production_receipt"] = receipt(registered["matched_job"])[2]
        row["deterministic"] = paired_check(candidate)
        row["status"] = row["deterministic"]["status"]
        eligible = (bool(row["production_structural_controls"]) and all(row["production_structural_controls"].values())
                    and row["same_policy_configuration"] and row["deterministic"]["passed"])
        if eligible and policy not in out["eligible"] and registered["status"] == "unresolved_outside_observed_native_spread":
            out["eligible"].append(policy)
            out["newly_supported"].append(policy)
            row["additional_route"] = True
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("screen","initial","development"))
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = supplement(args.stage)
    target = args.output or PUBLIC/f"deterministic_{result['stage']}_qualification.json"
    if target.exists():
        if read(target) != result:
            raise ValueError("Existing supplemental result differs; use a new output path")
    else:
        write_new(target,result)
    print(f"{target}: eligible {result['eligible']}, additional {result['newly_supported']}")


if __name__ == "__main__":
    main()
