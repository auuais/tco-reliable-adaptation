"""Independently reduce unscored training and resource records from stopped cases."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from profiled_compensation_protocol import EXECUTION, PUBLIC, ROOT, file_identity, require
from register_profiled_compensation import verify_release
from run_profiled_batch import verify_completed
from run_profiled_d0 import now, read, write_new
from score_profiled_compensation import ATTEMPT

INITIAL_FIELDS = ("loss_before_update", "gradient_sha256_unclipped", "adapter_state_sha256")
FORWARD_FIELDS = ("loss_before_update", "scales_before_update", "compensation_terms")
ORIGINAL_SEAL = "results/lora_profiled_v1/prediction_seal.json"
REPEATED_ARMS = {"P0", "P1", "P3"}


def close(first, second):
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def view_vector(values):
    require(len(values) == 5 and all(math.isfinite(n) and n >= 0 for n in values),
            "Expected five finite nonnegative view values")
    return values


def check_trace(arm, row, index):
    require(row["step"] == index and row["optimizer_updates_cumulative"] == index+1,
            "Training step count differs")
    require(row["graph_free_decoder_calls"] == row["differentiable_decoder_calls"] == 20,
            "Training step did not collect and replay all directed pairs")
    require(math.isfinite(row["adapter_update_l2"]) and row["adapter_update_l2"] > 0,
            "Expected a finite optimizer update")
    require(math.isfinite(row["loss_before_update"]) and row["loss_before_update"] >= 0,
            "Expected a finite nonnegative objective")
    stats = row["compensation_terms"]
    if arm == "P1":
        require(stats is None and row["support_fold"] is None, "P1 must report the matched D loss")
        return
    require(row["support_fold"] == index % 2 and stats is not None, "Wrong adaptation fold")
    scales = row["scales_before_update"]
    require(len(scales) == 5 and all(math.isfinite(v) and v > 0 for v in scales),
            "Expected five positive fitted scales")
    view_compensation = view_vector(stats["compensation_l2_by_view"])
    view_damping = view_vector(stats["matched_support_penalty_by_view"])
    energy = math.fsum(v*v for v in view_compensation)
    require(close(stats["compensation_l2"], math.sqrt(energy)), "Per-view compensation norm differs")
    penalty = {"P3": 0., "R0": math.fsum(view_damping), "R1": .5*energy, "R2": .5*energy}[arm]
    require(close(stats["penalty"], penalty) and close(stats["base"]+penalty, row["loss_before_update"]),
            "Reported training loss differs from the registered objective")
    if index == 0:
        require(penalty == 0. and stats["compensation_l2"] == 0.,
                "The pristine state must have zero compensation and penalty")


def condition_checks(grouped):
    result = {}
    for condition, cases in grouped.items():
        fresh = {arm: cases[arm]["steps"][0] for arm in ("P3", "R0", "R1") if arm in cases}
        full = None
        if len(fresh) == 3:
            full = all(len({row[k] for row in fresh.values()}) == 1 for k in INITIAL_FIELDS)
            require(full, f"{condition} registered common initial full-gradient step differs")
        detached_forward = None
        detached_gradient_differs = None
        if {"R1", "R2"} <= cases.keys():
            first, second = (cases[arm]["steps"][0] for arm in ("R1", "R2"))
            detached_forward = all(first[k] == second[k] for k in FORWARD_FIELDS)
            require(detached_forward, f"{condition} R1 and R2 disagree on the initial forward value")
            detached_gradient_differs = first["gradient_sha256_unclipped"] != second["gradient_sha256_unclipped"]
        result[condition] = dict(p3_r0_r1_first_full_step_exact=full,
                                 r1_r2_first_forward_exact=detached_forward,
                                 r1_r2_first_gradient_differs=detached_gradient_differs)
    return result


def historical_control_checks(registry, complete):
    """Compare fresh fingerprints with the sealed M2 controls without loading arrays."""
    seal = read(ROOT / ORIGINAL_SEAL)
    require(seal["pass_all"] and seal["completed_states"] == 36, "Original prediction seal is incomplete")
    originals = {case["run_id"]: case["selected"] for case in seal["cases"]
                 if case["arm"] in REPEATED_ARMS}
    expected = {job["run_id"] for job in registry["jobs"] if job["arm"] in REPEATED_ARMS}
    require(originals.keys() == expected and len(expected) == 18, "Original and fresh controls differ")
    rows = []
    for run, fresh in complete.items():
        if run not in originals:
            continue
        bound = originals[run]
        identity = file_identity(bound["report"])
        require(identity["sha256"] == bound["report_sha256"], "Original control report changed: " + run)
        original = read(ROOT / bound["report"])
        require(original["run_id"] == run and original["optimizer_updates"] == fresh["optimizer_updates"],
                "Original and fresh control schedules differ: " + run)
        matches = {key: bound[key] == fresh["endpoint"][key] for key in ("native_sha256", "final_sha256")}
        matches["original_state_sha256"] = bound["original_state_sha256"] == fresh["original_state_sha256"]
        matches["all_adapter_update_hashes"] = (
            [step["adapter_state_sha256"] for step in original["steps"]]
            == [step["adapter_state_sha256"] for step in fresh["steps"]])
        rows.append(dict(run_id=run, historical_report=identity, matches=matches,
                         all_fingerprints_equal=all(matches.values())))
    return dict(prediction_seal=file_identity(ORIGINAL_SEAL), expected=len(expected), checked=len(rows),
                exact=sum(row["all_fingerprints_equal"] for row in rows), cases=rows)


def build(registry, *, complete_only=True):
    cases, grouped, completed_reports, pending = [], {}, {}, []
    manifest = ROOT / EXECUTION
    for job in registry["jobs"]:
        run = job["run_id"]
        name = "tco-profiled-compensation-" + run.lower().replace("_", "-")
        term = f"{PUBLIC}/launches/{name}/terminal.json"
        if not (ROOT / term).is_file():
            pending.append(run)
            continue
        terminal = read(ROOT / term)
        require(terminal["verified_stopped"] and terminal["state"]["ExitCode"] == 0,
                "Attempt without a successful verified shutdown: " + run)
        completed = verify_completed(ROOT, manifest, registry, job, ATTEMPT)
        require(completed is not None, "Terminal attempt has no completion: " + run)
        report = read(ROOT / completed["complete"])
        require(report["config_sha256"] == registry["config_sha256"]
                and report["endpoint_decoder_calls"] == 20, "Checkpoint config or capture scope differs")
        steps = report["steps"]
        for index, row in enumerate(steps):
            check_trace(job["arm"], row, index)
        grouped.setdefault(f"{job['scene']}_{job['adaptation_source']}", {})[job["arm"]] = report
        completed_reports[run] = report
        entry = dict(run_id=run, scene=job["scene"], side=job["adaptation_source"], arm=job["arm"],
                     report=file_identity(completed["complete"]), terminal=file_identity(term),
                     model_seconds=report["seconds"], local_wall_seconds=terminal["wall_seconds"],
                     peak_allocated_mib=report["peak_allocated_mib"],
                     peak_reserved_mib=report["peak_reserved_mib"],
                     optimizer_updates=report["optimizer_updates"], original_state_unchanged=True,
                     training_decoder_calls=report["training_graph_free_decoder_calls"]+
                                            report["training_differentiable_decoder_calls"])
        entry["last_observed_before_update_by_fold"] = {
            str(row["support_fold"]): dict(step=row["step"], **row["compensation_terms"])
            for row in steps[-2:] if row["compensation_terms"] is not None}
        cases.append(entry)

    require(cases and (not complete_only or not pending), "Complete archive requires all 36 states")
    conditions = condition_checks(grouped)
    totals = {}
    for arm in dict.fromkeys(row["arm"] for row in cases):
        rows = [row for row in cases if row["arm"] == arm]
        totals[arm] = dict(cases=len(rows),
                           median_local_wall_seconds=median(row["local_wall_seconds"] for row in rows),
                           peak_allocated_mib=max(row["peak_allocated_mib"] for row in rows))
    return dict(schema="profiled_compensation_training_audit_v1", pass_all=True,
                scope="Training traces and resource summaries. Report excludes held-source scores.",
                verified_at_utc=now(), cases=cases, complete_count=len(cases), pending=pending,
                conditions=conditions, by_arm=totals,
                historical_control_reproduction=historical_control_checks(registry, completed_reports),
                total_optimizer_updates=sum(row["optimizer_updates"] for row in cases),
                total_local_wall_seconds=math.fsum(row["local_wall_seconds"] for row in cases),
                peak_allocated_mib=max(row["peak_allocated_mib"] for row in cases),
                peak_reserved_mib=max(row["peak_reserved_mib"] for row in cases),
                rental_compute_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="store_true", help="Audit only verified stopped cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry, _ = verify_release()
    report = build(registry, complete_only=not args.checkpoint)
    write_new(args.output, report)
    print(json.dumps({k: report[k] for k in ("pass_all", "complete_count", "total_optimizer_updates",
                                            "total_local_wall_seconds", "peak_allocated_mib")}))


if __name__ == "__main__":
    main()
