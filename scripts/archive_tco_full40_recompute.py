"""Snapshot the completed TCO recomputation study to the second physical disk."""

import argparse
import json
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import physical_disks, require, snapshot_path, verify_snapshot
from profiled_artifacts import logical_inventory, resolve_logical
from replay_tco_full40_recompute import replay
from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock
from run_tco_full40_recompute_local import case_path
from tco_full40_recompute_protocol import INITIAL, LOCKED, PRIVATE, PUBLIC, ROOT, STUDY, verify
from tco_one_percent_protocol import CONFIRM_MANIFEST as PRIOR_CONFIRMATION
from tco_one_percent_protocol import PUBLIC as PRIOR_PUBLIC


ARCHIVE = Path("E:/ResearchData/TCO/research_results") / STUDY
RECEIPT = PUBLIC / "archive_receipt.json"
PROVENANCE = (
    Path(__file__).resolve(),
    PRIOR_CONFIRMATION,
    PRIOR_PUBLIC / "confirmation_inputs.json",
    ROOT / "scripts/archive_profiled_compensation.py",
    ROOT / "scripts/archive_profiled_evidence.py",
    ROOT / "scripts/profiled_artifacts.py",
)


def inventory():
    """Include every retained private attempt and public result, including failures."""
    paths = set()
    for root in (PRIVATE, PUBLIC):
        prefix = root.relative_to(ROOT).as_posix()
        paths.update(f"{prefix}/{name}" for name in logical_inventory(root))
    paths.discard(RECEIPT.relative_to(ROOT).as_posix())
    manifests = sorted((ROOT / "manifests").glob("tco_full40_recompute*.json"))
    paths.update(path.relative_to(ROOT).as_posix() for path in (*PROVENANCE, *manifests))
    return paths


def plan_archive():
    require((PUBLIC / "README.md").is_file(), "Finish the public study summary before archiving")
    require(INITIAL.is_file(), "Missing initial roster")
    initial = verify("initial")
    source_hashes = initial["source_sha256_lf"]
    if LOCKED.is_file():
        source_hashes = verify("locked")["source_sha256_lf"]
        replay()
    paths = inventory() | source_hashes.keys()
    rows = []
    markers = {}
    for name in sorted(paths):
        path = resolve_logical(ROOT, name, must_exist=True)
        before = path.stat()
        digest = sha(path)
        if name in source_hashes:
            require(sha(path, text=True) == source_hashes[name], f"Registered source differs: {name}")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed while hashing: {name}")
        rows.append(dict(file=name, bytes=after.st_size, sha256=digest))
        if path.suffix in (".json", ".jsonl", ".log") or name in source_hashes:
            markers[name] = digest

    by_name = {row["file"]: row["sha256"] for row in rows}
    targets = []
    for name in sorted(markers):
        if not name.startswith(PRIVATE.relative_to(ROOT).as_posix() + "/") or not name.endswith("/terminal.json"):
            continue
        receipt = read(resolve_logical(ROOT, name, must_exist=True))
        relative_parent = name.rsplit("/", 1)[0]
        log_sha = receipt.get("log_sha256") or receipt.get("stdout_sha256")
        if log_sha is not None:
            require(by_name.get(f"{relative_parent}/stdout.log") == log_sha, f"Terminal log hash mismatch: {name}")
        if receipt.get("verdict") == "completed":
            complete = case_path({"id": receipt["job"]}, receipt["attempt"]) / "complete.json"
            require(by_name.get(complete.relative_to(ROOT).as_posix()) == receipt["complete_sha256"],
                    f"Terminal completion hash mismatch: {name}")
        targets.append(dict(terminal=name, sha256=by_name[name]))
    return dict(schema="profiled_archive_selection_v1", study=STUDY, targets=targets,
                completion_guards=markers, files=rows, total_bytes=sum(row["bytes"] for row in rows),
                mutable_progress_or_logs_included=True, arrays_deserialized=False, scores_computed=False)


def archive(action, snapshot, check_source=False):
    target = snapshot_path(ROOT, ARCHIVE, snapshot)
    if action == "verify":
        result = verify_snapshot(ARCHIVE, snapshot, source_root=ROOT if check_source else None)
        if RECEIPT.is_file():
            receipt = read(RECEIPT)
            if receipt["snapshot"] == snapshot:
                require(sha(target / "manifest.json") == receipt["manifest_sha256"],
                        "Archive manifest differs from the public receipt")
        return result

    require(not target.exists(), "Keep existing snapshots and select a fresh ID")
    if action == "copy":
        require(not RECEIPT.exists(), "A verified public archive receipt already exists")
    # The same lock is used by the local study launcher. Stopped-attempt logs
    # are evidence and must be copied after the serial job queue is idle.
    with host_lock():
        disks = physical_disks(ROOT, ARCHIVE)
        plan = plan_archive()
        if action == "plan":
            return dict(plan_only=True, files=len(plan["files"]), total_bytes=plan["total_bytes"],
                        terminal_receipts=len(plan["targets"]), physical_disks=disks)
        result = copy_plan(ARCHIVE, snapshot, plan, disks)
        current = inventory()
        require(current <= {row["file"] for row in plan["files"]}, "Study inventory grew during archival")
        receipt = dict(**result, manifest_sha256=sha(target / "manifest.json"),
                       terminal_receipts=len(plan["targets"]), physical_disks=disks, rental_compute_usd=0)
        write_new(RECEIPT, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "copy", "verify"))
    parser.add_argument("--snapshot", default="study-v1")
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    if args.check_source and args.action != "verify":
        parser.error("--check-source is available only for verify")
    print(json.dumps(archive(args.action, args.snapshot, args.check_source), indent=2))


if __name__ == "__main__":
    main()
