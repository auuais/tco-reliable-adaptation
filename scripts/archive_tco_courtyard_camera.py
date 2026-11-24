"""Snapshot the completed local Courtyard and camera study on the second physical disk."""

import argparse
import json
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import physical_disks, require, snapshot_path, verify_snapshot
from profiled_artifacts import logical_inventory, resolve_logical
from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock
from run_tco_courtyard_camera_local import case_path
from tco_courtyard_camera_protocol import INITIAL, LOCKED, PRIVATE, PUBLIC, ROOT, STUDY, verify
from tco_full40_recompute_protocol import LOCKED as PRIOR_LOCKED, PUBLIC as PRIOR_PUBLIC


DESTINATION = Path("E:/ResearchData/TCO/research_results") / STUDY
RECEIPT = PUBLIC / "archive_receipt.json"
PROVENANCE = (
    Path(__file__).resolve(),
    INITIAL,
    PRIOR_LOCKED,
    PRIOR_PUBLIC / "archive_receipt.json",
    PRIOR_PUBLIC / "evidence.json",
    ROOT / "scripts/archive_profiled_compensation.py",
    ROOT / "scripts/archive_profiled_evidence.py",
    ROOT / "scripts/profiled_artifacts.py",
)


def inventory():
    """Include all attempts, stdout, snapshots, and public evidence."""
    files = {item.relative_to(ROOT).as_posix() for item in PROVENANCE}
    if LOCKED.is_file():
        files.add(LOCKED.relative_to(ROOT).as_posix())
    for root in (PRIVATE, PUBLIC):
        prefix = root.relative_to(ROOT).as_posix()
        files.update(f"{prefix}/{name}" for name in logical_inventory(root))
    files.discard(RECEIPT.relative_to(ROOT).as_posix())
    return files


def terminal_targets(markers):
    targets = []
    prefix = PRIVATE.relative_to(ROOT).as_posix() + "/launches/"
    attempts = {name.rsplit("/", 1)[0] for name in markers if name.startswith(prefix) and name.endswith("/start.json")}
    terminal = {name.rsplit("/", 1)[0]: name for name in markers
                if name.startswith(prefix) and name.endswith("/terminal.json")}
    require(set(terminal) == attempts, "Every retained launch must have a terminal receipt")
    for directory, name in sorted(terminal.items()):
        row = read(resolve_logical(ROOT, name, must_exist=True))
        job_id, attempt_id = directory.removeprefix(prefix).split("/")
        require((row["job"], row["attempt"], row["rental_compute_usd"]) == (job_id, attempt_id, 0),
                f"Invalid attempt identity or rental total: {name}")
        require(markers.get(f"{directory}/stdout.log") == row["log_sha256"], f"Terminal log hash mismatch: {name}")
        if row["verdict"] == "completed":
            receipt = case_path({"id": job_id}, attempt_id) / "complete.json"
            require(row["stopped"] is True and row["exit_code"] == 0 and bool(row["log_sha256"]),
                    f"Successful attempt lacks verified closure: {name}")
            require(bool(row["complete_sha256"])
                    and markers.get(receipt.relative_to(ROOT).as_posix()) == row["complete_sha256"],
                    f"Case completion hash mismatch: {name}")
        targets.append(dict(terminal=name, sha256=markers[name]))
    return targets


def plan_archive():
    require((PUBLIC / "README.md").is_file(), "Finish the public study report before archiving")
    initial = verify("initial")
    source_hashes = initial["source_sha256_lf"]
    if LOCKED.is_file():
        from replay_tco_courtyard_camera import replay

        replay()
        source_hashes = verify("locked")["source_sha256_lf"]

    rows, markers = [], {}
    for name in sorted(inventory() | source_hashes.keys()):
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

    return dict(schema="profiled_archive_selection_v1", study=STUDY, targets=terminal_targets(markers),
                completion_guards=markers, files=rows, total_bytes=sum(row["bytes"] for row in rows),
                mutable_progress_or_logs_included=True, arrays_deserialized=False, scores_computed=False)


def archive(action, snapshot, check_source=False):
    target = snapshot_path(ROOT, DESTINATION, snapshot)
    if action == "verify":
        result = verify_snapshot(DESTINATION, snapshot, source_root=ROOT if check_source else None)
        if RECEIPT.is_file():
            receipt = read(RECEIPT)
            if receipt["snapshot"] == snapshot:
                require(sha(target / "manifest.json") == receipt["manifest_sha256"],
                        "Archive manifest differs from the public receipt")
        return result

    require(not target.exists(), "Keep existing snapshots and select a fresh ID")
    if action == "copy":
        require(not RECEIPT.exists(), "A verified archive receipt already exists")
    with host_lock():
        disks = physical_disks(ROOT, DESTINATION)
        plan = plan_archive()
        if action == "plan":
            return dict(plan_only=True, files=len(plan["files"]), total_bytes=plan["total_bytes"],
                        terminal_receipts=len(plan["targets"]), physical_disks=disks)
        result = copy_plan(DESTINATION, snapshot, plan, disks)
        require(inventory() <= {row["file"] for row in plan["files"]}, "Study inventory grew during archival")
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
