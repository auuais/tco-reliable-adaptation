"""Summarize Windows memory and utilization sampled alongside diagnostic TCO traces."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import re
from statistics import median

from run_profiled_d0 import read, sha, write_new
from tco_native_stall_trace import TRACE_FILE, json_lines
from tco_native_stalls_quality_capacity_protocol import PRIVATE, PUBLIC, STUDY


GPU_NAME = "NVIDIA GeForce RTX 3080"
GPU_FIELDS = ("total_mib", "used_mib", "free_mib", "utilization_gpu_pct", "utilization_memory_pct")
HOST_FIELDS = ("AvailableMBytes", "CommittedBytes", "CommitLimit", "PagesInputPersec", "PagesOutputPersec")
TRACKED_PHASES = {"transformer", "backward", "adam", "final.transformer", "render.rgb"}
POWERSHELL_TICKS = re.compile(r"(\.\d{6})\d(?=[+-])")


def clock(row):
    # PowerShell emits .NET ticks. Python 3.9 datetime has microsecond precision.
    return datetime.fromisoformat(POWERSHELL_TICKS.sub(r"\1", row["utc"]))


def nvidia_values(row):
    for line in row.get("nvidia_smi") or ():
        fields = next(csv.reader([line], skipinitialspace=True))
        if len(fields) == 7 and fields[1] == GPU_NAME:
            return {key: int(value) for key, value in zip(GPU_FIELDS, fields[2:]) if value.isdecimal()}
    return {}


def stats(values):
    return dict(count=len(values), first=values[0], minimum=min(values), median=median(values),
                maximum=max(values), last=values[-1])


def counter_summary(rows):
    counters = defaultdict(list)
    if not rows:
        return dict(samples=0)
    for row in rows:
        for name, value in nvidia_values(row).items():
            counters["nvidia", name].append(value)
        for adapter in row.get("wddm_adapter_memory") or ():
            for name in ("DedicatedUsage", "SharedUsage"):
                if type(adapter.get(name)) is int:
                    counters["wddm", f"{adapter['Name']}/{name}"].append(adapter[name])
        for name in HOST_FIELDS:
            value = (row.get("host_memory") or {}).get(name)
            if type(value) is int:
                counters["host", name].append(value)
        if isinstance(row.get("sample_seconds"), (float, int)):
            counters["sampler", "sample_seconds"].append(row["sample_seconds"])
    values = defaultdict(dict)
    for (group, name), series in sorted(counters.items()):
        values[group][name] = stats(series)
    return dict(samples=len(rows), first_utc=rows[0]["utc"], last_utc=rows[-1]["utc"], counters=dict(values),
                unavailable=sorted({value for row in rows for value in row.get("unavailable", ())}))


def phase_windows(events):
    entered, finished, slowest = {}, {}, {}
    for event in events:
        phase, edge = event["phase"], event["edge"]
        if edge == "enter":
            entered[event["serial"]] = event
        elif edge in {"exit", "error", "early_stop"}:
            finished[event["serial"]] = event
        if edge == "cuda_complete" and phase in TRACKED_PHASES:
            if phase not in slowest or event["cuda_elapsed_ms"] > slowest[phase]["cuda_elapsed_ms"]:
                slowest[phase] = event
    for phase in ("score_alignment_icp", "serialize_prediction"):
        host_completions = [event for serial, event in finished.items()
                            if event["phase"] == phase and serial in entered]
        if host_completions:
            slowest[phase] = max(host_completions,
                                 key=lambda event: clock(event) - clock(entered[event["serial"]]))
    ranges = {f"slowest_{name}": (entered[row["serial"]], row) for name, row in sorted(slowest.items())
              if row["serial"] in entered}
    open_phases = [row for serial, row in entered.items() if serial not in finished]
    if open_phases:
        ranges["deepest_open_at_trace_end"] = (open_phases[-1], None)
    return ranges


def summarize_attempt(host_path, phases_path):
    samples = json_lines(host_path)
    events = json_lines(phases_path)
    windows = {}
    for name, (begin, end) in phase_windows(events).items():
        selected = [row for row in samples if clock(begin) <= clock(row) and (end is None or clock(row) <= clock(end))]
        windows[name] = dict(
            phase=begin["phase"], updates_at_entry=begin["updates_done"], begin_utc=begin["utc"],
            polled_end_utc=end["utc"] if end else None, end_edge=end["edge"] if end else None,
            cuda_elapsed_ms=end.get("cuda_elapsed_ms") if end else None, telemetry=counter_summary(selected),
        )
    return dict(host_sha256=sha(host_path), phase_sha256=sha(phases_path) if phases_path.is_file() else None,
                full_attempt=counter_summary(samples), phase_windows=windows)


def summarize_study(private):
    attempts = defaultdict(dict)
    for path in sorted((private / "host_telemetry").glob("*/*.jsonl")):
        job, attempt = path.parent.name, path.stem
        terminal = private / "launches" / job / attempt / "terminal.json"
        if terminal.is_file():
            phases = private / "cases" / job / attempt / "case" / TRACE_FILE
            attempts[job][attempt] = summarize_attempt(path, phases)
    return dict(study=STUDY, rental_compute_usd=0, attempts=dict(attempts),
                method="NVIDIA-SMI device counters, WDDM counters per adapter, Windows host memory and observed "
                       "start-to-polled-end phase windows. WDDM values are adapter-wide. A completion can be "
                       "observed later than it occurred, and short scopes can fall between samples.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = summarize_study(PRIVATE)
    output = PUBLIC / "host_telemetry.json"
    if args.publish:
        if output.is_file():
            if read(output) != result:
                raise ValueError("Previously published host telemetry differs")
        else:
            write_new(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
