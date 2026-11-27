"""Seal local TCO stall, geometry and capacity evidence on a second physical disk."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import physical_disks, snapshot_path, verify_snapshot
from profiled_artifacts import logical_inventory, resolve_logical
from replay_tco_native_stalls_quality_capacity import PUBLIC, ROOT, STUDY, require
from replay_tco_native_stalls_quality_capacity_release import replay
from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock
from summarize_tco_host_telemetry import summarize_study
from tco_native_stall_trace import TRACE_FILE, summarize_trace
from tco_native_stalls_quality_capacity_protocol import AMENDMENT, CAPACITY, INITIAL, LOCKED, receipt_correction


PRIVATE = ROOT / "reports" / STUDY
DESTINATION = Path("E:/ResearchData/TCO/research_results") / STUDY
RECEIPT = PUBLIC / "archive_receipt.json"
REGISTRATIONS = (INITIAL, LOCKED, CAPACITY)
TOOLS = (
    Path(__file__).resolve(),
    ROOT / "scripts/archive_profiled_compensation.py",
    ROOT / "scripts/archive_profiled_evidence.py",
    ROOT / "scripts/profiled_artifacts.py",
    ROOT / "scripts/replay_tco_native_stalls_quality_capacity_release.py",
    ROOT / "scripts/summarize_tco_host_telemetry.py",
)


def logical_name(path):
    return path.relative_to(ROOT).as_posix()


def inventory(source_names):
    names = {logical_name(path) for path in (*REGISTRATIONS, *TOOLS)} | set(source_names)
    if AMENDMENT.is_file():
        names.add(logical_name(AMENDMENT))
    for directory in (PRIVATE, PUBLIC):
        names.update(f"{logical_name(directory)}/{name}" for name in logical_inventory(directory))
    names.discard(logical_name(RECEIPT))
    return names


def effective_sources(plan, correction):
    return {**plan["source_sha256_lf"],
            **{name: row["after"] for name, row in correction.get("source_changes", {}).items()}}


def validate_corrected_attempts(correction, targets):
    if not correction:
        return
    require(correction["rental_compute_usd"] == 0 and correction["scientific_changes"] == [],
            "Receipt correction changes the cost or scientific design")
    prefix = f"{logical_name(PRIVATE)}/launches/"
    archived = {row["terminal"]: row["sha256"] for row in targets}
    prior = correction["prior_attempts"]
    require(bool(prior), "Correction lists no preceding attempts")
    for attempt, expected in prior.items():
        require(re.fullmatch(r"[a-z0-9_]+/attempt[0-9]{2}", attempt) is not None
                and isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected) is not None,
                f"Invalid corrected attempt identity or digest {attempt}")
        terminal = f"{prefix}{attempt}/terminal.json"
        require(archived.get(terminal) == expected, f"Corrected receipt is missing or changed for {attempt}")


def terminal_targets(markers, ledger, attempt_counts):
    prefix = logical_name(PRIVATE)
    launch = f"{prefix}/launches"
    starts = {name.removesuffix("/start.json") for name in markers
              if name.startswith(f"{launch}/") and name.endswith("/start.json")}
    terminals = {name.removesuffix("/terminal.json") for name in markers
                 if name.startswith(f"{launch}/") and name.endswith("/terminal.json")}
    targets = []
    seen = set()
    observed = Counter()
    for run in ledger["runs"]:
        identifier = run["id"]
        require(run["verdict"] != "unattempted", f"Unresolved registered slot {identifier}")
        if run["verdict"] == "gated":
            gate = read(resolve_logical(ROOT, f"{prefix}/gates/{identifier}.json", must_exist=True))
            require(gate["reason"] == run["gate_reason"], f"Changed gate reason for {identifier}")
        for published in run["attempts"]:
            attempt = published["attempt"]
            launch_dir = f"{launch}/{identifier}/{attempt}"
            case_dir = f"{prefix}/cases/{identifier}/{attempt}/case"
            terminal = f"{launch_dir}/terminal.json"
            require(launch_dir in terminals and launch_dir not in seen, f"Missing or duplicate receipt {terminal}")
            seen.add(launch_dir)
            raw = read(resolve_logical(ROOT, terminal, must_exist=True))
            start = read(resolve_logical(ROOT, f"{launch_dir}/start.json", must_exist=True))
            extras = {"oom_message", "trace", "host_telemetry_sha256"}
            require(raw == {key: value for key, value in published.items() if key not in extras},
                    f"Public and raw terminal receipts differ for {identifier}/{attempt}")
            require((raw["job"], raw["attempt"], raw["rental_compute_usd"],
                     start["job"], start["attempt"], start["rental_compute_usd"],
                     start["outer_seconds"]) == (identifier, attempt, 0, identifier, attempt, 0, raw["outer_seconds"]),
                    f"Changed execution identity, deadline or cost in {terminal}")
            observed[raw["verdict"]] += 1
            log = f"{launch_dir}/stdout.log"
            require(markers.get(log) == raw["log_sha256"], f"Mismatch in stdout digest for {terminal}")
            if raw["verdict"] == "oom":
                line = published["oom_message"]
                text = resolve_logical(ROOT, log, must_exist=True).read_text(encoding="utf-8", errors="replace")
                require("CUDA out of memory" in line and any(line == raw_line.strip() for raw_line in text.splitlines()),
                        f"Missing CUDA allocator exception for {terminal}")
            if raw["verdict"] == "completed":
                complete = f"{case_dir}/complete.json"
                require(markers.get(complete) == raw["complete_sha256"] is not None,
                        f"Missing or changed completion {complete}")
                result = read(resolve_logical(ROOT, complete, must_exist=True))
                require(result == run["result"] and raw["stopped"] and raw["exit_code"] == 0,
                        f"Invalid completion for {identifier}/{attempt}")
                for artifact, digest in result.get("artifact_sha256", {}).items():
                    require(Path(artifact).name == artifact and markers.get(f"{case_dir}/{artifact}") == digest,
                            f"Changed output {artifact} for {identifier}/{attempt}")
            if "trace" in published:
                # The trace may be absent if startup failed before the case directory existed.
                raw_trace = markers.get(f"{case_dir}/{TRACE_FILE}")
                trace = summarize_trace(ROOT / case_dir)
                require(trace == published["trace"] and raw_trace == (trace or {}).get("trace_sha256"),
                        f"Trace differs from its compact evidence for {terminal}")
            telemetry = f"{prefix}/host_telemetry/{identifier}/{attempt}.jsonl"
            require(markers.get(telemetry) == published.get("host_telemetry_sha256"),
                    f"Changed telemetry bytes for {terminal}")
            targets.append(dict(terminal=terminal, sha256=markers[terminal]))
    require(starts == terminals == seen, "Raw start/terminal inventory differs from public evidence")
    require(dict(observed) == attempt_counts, "Archived attempts differ from replayed totals")
    return sorted(targets, key=lambda row: row["terminal"])


def archive_plan():
    require((PUBLIC / "README.md").is_file(), "Finish the public report before archiving")
    decision = replay("full")
    require(decision == read(PUBLIC / "decision.json"), "Published decision differs from independent CPU replay")
    telemetry = PUBLIC / "host_telemetry.json"
    if telemetry.is_file():
        require(summarize_study(PRIVATE) == read(telemetry), "Published telemetry differs from the raw samples")
    authority = read(INITIAL)
    correction = receipt_correction(authority)
    source_hashes = effective_sources(authority, correction)
    files, markers = [], {}
    for name in sorted(inventory(source_hashes)):
        path = resolve_logical(ROOT, name, must_exist=True)
        before = path.stat()
        digest = sha(path)
        if name in source_hashes:
            require(sha(path, text=True) == source_hashes[name], f"Registered source changed: {name}")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed while hashing: {name}")
        files.append(dict(file=name, bytes=after.st_size, sha256=digest))
        markers[name] = digest
    ledger = read(PUBLIC / "evidence.json")
    targets = terminal_targets(markers, ledger, decision["terminal_attempt_counts"])
    validate_corrected_attempts(correction, targets)
    correction_sha = markers.get(logical_name(AMENDMENT))
    return dict(schema="profiled_archive_selection_v1", study=STUDY, targets=targets,
                completion_guards=markers, files=files, total_bytes=sum(row["bytes"] for row in files),
                arrays_deserialized=False, scores_computed=False, receipt_correction_sha256=correction_sha)


def archive(action, snapshot, check_source):
    target = snapshot_path(ROOT, DESTINATION, snapshot)
    if action == "verify":
        verified = verify_snapshot(DESTINATION, snapshot, source_root=ROOT if check_source else None)
        receipt = read(RECEIPT)
        require(receipt["snapshot"] == snapshot and receipt["manifest_sha256"] == sha(target / "manifest.json"),
                "Archive manifest differs from the public receipt")
        selection = read(target / "manifest.json")["selection"]
        require(verified["files_verified"] == receipt["files_verified"]
                and verified["total_bytes"] == receipt["total_bytes"]
                and receipt["terminal_receipts"] == len(selection["targets"])
                and receipt["receipt_correction_sha256"] == selection["receipt_correction_sha256"],
                "Public archive totals or receipt authority differ")
        if check_source:
            require(receipt["physical_disks"] == physical_disks(ROOT, DESTINATION), "Physical disk mapping changed")
        return verified
    require(not target.exists() and not RECEIPT.exists(), "Preserve prior snapshots and use a fresh study receipt")
    with host_lock():
        disks = physical_disks(ROOT, DESTINATION)
        plan = archive_plan()
        if action == "plan":
            return dict(plan_only=True, files=len(plan["files"]), total_bytes=plan["total_bytes"],
                        terminal_receipts=len(plan["targets"]), physical_disks=disks)
        verified = copy_plan(DESTINATION, snapshot, plan, disks)
        authority = read(INITIAL)
        names = {row["file"] for row in plan["files"]}
        require(names == inventory(effective_sources(authority, receipt_correction(authority))),
                "Study inventory changed during the copy")
        result = dict(**verified, manifest_sha256=sha(target / "manifest.json"),
                      terminal_receipts=len(plan["targets"]), physical_disks=disks, rental_compute_usd=0,
                      receipt_correction_sha256=plan["receipt_correction_sha256"])
        write_new(RECEIPT, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "copy", "verify"))
    parser.add_argument("--snapshot", default="study-v1")
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    if args.check_source and args.action != "verify":
        parser.error("--check-source is available for verify")
    print(json.dumps(archive(args.action, args.snapshot, args.check_source), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
