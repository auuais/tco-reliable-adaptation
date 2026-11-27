"""Summarize the registered five-view stall and repeatability evidence on CPU."""

import argparse
from collections import Counter
import json
from math import log

from replay_tco_courtyard_camera import payload_digest
from replay_tco_native_stalls_quality_capacity import (
    ENDPOINTS, native_repeatability, summarize_resources, verify_attempts, verify_model_result,
)
from run_profiled_d0 import read, sha, write_new
from tco_native_stalls_quality_capacity_protocol import INITIAL, PUBLIC, ROOT, STUDY


SLOTS = {"repeat", "trace", "small"}
RESULT = PUBLIC / "native_stalls.json"


def checkpoint_fields(trace, key):
    row = trace.get(key)
    if not row:
        return None
    return {name: row[name] for name in (
        "phase", "edge", "elapsed_seconds", "updates_done", "allocated", "reserved",
        "peak_allocated", "peak_reserved", "exception",
    ) if name in row}


def trace_attempt(attempt):
    trace = attempt.get("trace")
    result = dict(attempt=attempt["attempt"], verdict=attempt["verdict"],
                  wall_seconds=attempt["wall_seconds"], log_sha256=attempt["log_sha256"])
    if trace is None:
        return result
    result.update({name: checkpoint_fields(trace, name) for name in (
        "enter", "exit", "cuda_end_submitted", "cuda_complete", "error", "cuda_poll_error", "heartbeat",
    )})
    result.update(
        trace_sha256=trace["trace_sha256"],
        host_telemetry_sha256=attempt.get("host_telemetry_sha256"),
        row_count=trace["row_count"],
        device_confirmed_updates=trace["last_device_confirmed_adam_update"],
        open_host_phases=[row["phase"] for row in trace["open_host_phases"]],
        trace_elapsed_seconds=trace["elapsed_sec"],
        cuda_phase_timings={name: dict(count=data["count"], median_ms=data["median_ms"],
                                      slowest_ms=data["slowest"]["cuda_elapsed_ms"],
                                      slowest_update=data["slowest"]["update_at_entry"])
                            for name, data in trace["cuda_phase_timings"].items()},
    )
    return result


def repeatability(records):
    result = native_repeatability(records)
    for pair in result:
        prefix = f"repeat_{pair['scene']}_{pair['seed']}_native"
        pair["source_jobs"] = [f"{prefix}_a", f"{prefix}_b"]
        pair["prediction_sha256"] = [
            records[name]["result"]["output_sha256"] if records[name]["result"] else None
            for name in pair["source_jobs"]
        ]
        verified = {}
        for name in pair["source_jobs"]:
            record = records[f"rescore_{name}"]
            score = record["result"]
            values = None
            if score is not None:
                source = records[name]
                if not (score["source_completion_sha256"] == source["attempts"][-1]["complete_sha256"]
                        and score["output_sha256"] == source["result"]["output_sha256"]):
                    raise ValueError(f"Changed rescore source for {name}")
                values = {key: 1000 * (score["metrics"][key] - source["result"]["metrics"][key])
                          for key in ENDPOINTS}
            verified[name] = dict(verdict=record["verdict"], deltas_mm=values,
                                  within_0p01_mm=all(abs(delta) <= 0.01 for delta in values.values()) if values else None)
        pair["rescores"] = verified
        for field in (pair["errors"] or {}).values():
            ratio = field["b_over_a"]
            field.update(signed_difference_pct=100 * (ratio - 1), absolute_log_ratio=abs(log(ratio)))
    return result


def prior_repeatability():
    source = ROOT / "results/tco_render_lifetime_v1/decision.json"
    previous = read(source)
    return dict(study=previous["study"], digest_sha256_lf=sha(source, text=True),
                path=source.relative_to(ROOT).as_posix())


def report(plan, ledger, historical):
    if (plan["study"] != ledger["study"] or ledger["study"] != STUDY
            or ledger["registration_sha256"] != sha(INITIAL, text=True)
            or ledger["rental_compute_usd"] != 0 or historical["rental_compute_usd"] != 0):
        raise ValueError("Changed study registration or spend")
    indexed = {job["id"]: job for job in plan["jobs"]}
    jobs = {name: job for name, job in indexed.items() if job["cohort"] in SLOTS}
    jobs.update({name: job for name, job in indexed.items() if job["cohort"] == "rescore"
                 and indexed[job["source_job"]]["cohort"] == "repeat"})
    records = {row["id"]: row for row in ledger["runs"] if row["id"] in jobs}
    if records.keys() != jobs.keys():
        raise ValueError("Stage 1 evidence omitted registered cases")
    for name, job in jobs.items():
        row = records[name]
        from tco_native_stalls_quality_capacity_protocol import receipt_correction
        limit = receipt_correction(plan).get("retry_limits", {}).get(name, plan["max_attempts"])
        verify_attempts(job, row, limit)
        if row["result"] and job["cohort"] in SLOTS:
            verify_model_result(job, row["result"])

    attempts = []
    traces = []
    for name, row in records.items():
        job = jobs[name]
        if job["cohort"] == "trace":
            traces.append(dict(job=name, scene=job["scene"], policy=job["policy"],
                               verdict=row["verdict"], gate_reason=row.get("gate_reason"),
                               attempts=[trace_attempt(attempt) for attempt in row["attempts"]]))
        if job["cohort"] not in {"repeat", "small"}:
            continue
        case = row["result"]
        attempts.append(dict(job=name, cohort=job["cohort"], scene=job["scene"], seed=job["seed"],
                             width=job["width"], policy=job["policy"], verdict=row["verdict"],
                             gate_reason=row.get("gate_reason"),
                             succeeded_after_retry=bool(case and len(row["attempts"]) > 1),
                             attempts=[trace_attempt(attempt) for attempt in row["attempts"]],
                             case_metrics=dict(scene_seconds=case["scene_seconds"], optimizer_updates=case["updates"],
                                               peak_allocated_bytes=case["peak_allocated"],
                                               peak_reserved_bytes=case["peak_reserved"],
                                               score_seconds=case["score_seconds"],
                                               serialization_seconds=case["serialization_seconds"],
                                               errors_mm={key: case["metrics"][key] * 1000 for key in ENDPOINTS})
                             if case else None))

    historical_counts = Counter(row["verdict"] for row in historical["attempts"])
    repeat_counts = Counter(row["verdict"] for row in attempts if row["cohort"] == "repeat")
    native_chess = [row for row in attempts if row["scene"] == "chess/seq-03"
                    and row["cohort"] == "repeat" and row["policy"] == "native"]
    return dict(
        study=STUDY, rental_compute_usd=0, registration_sha256=ledger["registration_sha256"],
        stage_evidence_sha256=payload_digest({"runs": list(records.values())}),
        historical=dict(digest_sha256=payload_digest(historical), counts=dict(sorted(historical_counts.items())),
                        loss_rows="zero-based and emitted before backward, clipping, and Adam",
                        optimizer_count_source="verified successful completions"),
        earlier_native_repeatability=prior_repeatability(),
        clean_repeat_counts=dict(sorted(repeat_counts.items())),
        native_chess_timeouts=sum(item["verdict"] == "timeout" for row in native_chess for item in row["attempts"]),
        clean_resources=summarize_resources(jobs, records, {"repeat"}),
        reduced_resolution_resources=summarize_resources(jobs, records, {"small"}),
        cases=attempts, traces=traces, repeatability=repeatability(records),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    plan, ledger, historical = (read(path) for path in (
        INITIAL, PUBLIC / "evidence.json", PUBLIC / "historical_stalls.json",
    ))
    result = report(plan, ledger, historical)
    if RESULT.is_file():
        if read(RESULT) != result:
            raise ValueError(f"{RESULT.name} differs from CPU replay")
    elif args.publish:
        rows = [*result["cases"], *result["traces"]]
        rows.extend(row for pair in result["repeatability"] for row in pair["rescores"].values())
        if any(row["verdict"] == "unattempted" for row in rows):
            raise ValueError("Finish or gate the registered stage 1 cases before publishing")
        write_new(RESULT, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
