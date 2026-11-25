"""Copy and verify the TCO render-lifetime evidence on a second physical disk."""

import argparse
import json
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import physical_disks, snapshot_path, verify_snapshot
from profiled_artifacts import logical_inventory, resolve_logical
from replay_tco_render_lifetime import INITIAL, LOCKED, PUBLIC, ROOT, STUDY, replay, require
from run_profiled_d0 import read, sha, write_new
from run_profiled_d1_local import host_lock


PRIVATE = ROOT / "reports" / STUDY
DESTINATION = Path("E:/ResearchData/TCO/research_results") / STUDY
RECEIPT = PUBLIC / "archive_receipt.json"
TOOLS = (
    Path(__file__).resolve(),
    ROOT / "scripts/archive_profiled_compensation.py",
    ROOT / "scripts/archive_profiled_evidence.py",
    ROOT / "scripts/profiled_artifacts.py",
    ROOT / "scripts/replay_tco_render_lifetime.py",
    ROOT / "scripts/test_tco_render_lifetime_replay.py",
)


def inventory():
    names = {path.relative_to(ROOT).as_posix() for path in (INITIAL, LOCKED, *TOOLS)}
    for path in (PRIVATE, PUBLIC):
        prefix = path.relative_to(ROOT).as_posix()
        names.update(f"{prefix}/{name}" for name in logical_inventory(path))
    names.discard(RECEIPT.relative_to(ROOT).as_posix())
    return names


def terminal_targets(markers):
    prefix = PRIVATE.relative_to(ROOT).as_posix()
    launches = f"{prefix}/launches/"
    started = {name.removesuffix("/start.json") for name in markers
               if name.startswith(launches) and name.endswith("/start.json")}
    stopped = {name.removesuffix("/terminal.json") for name in markers
               if name.startswith(launches) and name.endswith("/terminal.json")}
    require(started == stopped, "Runtime attempt receipts have an open launch")
    targets = []
    for directory in sorted(stopped):
        name = f"{directory}/terminal.json"
        terminal = read(ROOT / name)
        case, attempt = directory.removeprefix(launches).split("/")
        require((terminal["job"], terminal["attempt"], terminal["rental_compute_usd"]) == (case, attempt, 0),
                f"Archived receipt identity or budget differs: {name}")
        require(markers.get(f"{directory}/stdout.log") == terminal["log_sha256"],
                f"Stdout does not match the terminal receipt: {name}")
        if terminal["verdict"] == "completed":
            complete = f"{prefix}/cases/{case}/{attempt}/case/complete.json"
            require(terminal["stopped"] and terminal["exit_code"] == 0 and
                    markers.get(complete) == terminal["complete_sha256"] is not None,
                    f"Invalid archived completion: {name}")
            for artifact, digest in read(ROOT / complete).get("artifact_sha256", {}).items():
                require(markers.get(f"{complete.removesuffix('complete.json')}{artifact}") == digest,
                        f"Archived output differs from the completion receipt: {artifact} for {case}")
        targets.append(dict(terminal=name, sha256=markers[name]))
    return targets


def archive_plan():
    require((PUBLIC / "README.md").is_file(), "Finish the public report before archiving")
    decision = replay()
    source_hashes = read(LOCKED)["source_sha256_lf"]
    files, markers = [], {}
    for name in sorted(inventory() | source_hashes.keys()):
        path = resolve_logical(ROOT, name, must_exist=True)
        before = path.stat()
        digest = sha(path)
        if name in source_hashes:
            require(sha(path, text=True) == source_hashes[name], f"Registered source differs: {name}")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed while hashing: {name}")
        files.append(dict(file=name, bytes=after.st_size, sha256=digest))
        if path.suffix in (".json", ".jsonl", ".log", ".npz", ".pt") or name in source_hashes:
            markers[name] = digest
    targets = terminal_targets(markers)
    require(len(targets) == sum(decision["attempt_counts"].values()), "Archived attempt inventory differs")
    return dict(schema="profiled_archive_selection_v1", study=STUDY, targets=targets,
                completion_guards=markers, files=files, total_bytes=sum(row["bytes"] for row in files),
                mutable_progress_or_logs_included=True, arrays_deserialized=False, scores_computed=False)


def archive(action, snapshot, check_source=False):
    target = snapshot_path(ROOT, DESTINATION, snapshot)
    if action == "verify":
        result = verify_snapshot(DESTINATION, snapshot, source_root=ROOT if check_source else None)
        if RECEIPT.is_file() and read(RECEIPT)["snapshot"] == snapshot:
            require(sha(target / "manifest.json") == read(RECEIPT)["manifest_sha256"],
                    "Archive manifest hash differs")
        return result
    require(not target.exists(), "Choose a fresh snapshot ID")
    with host_lock():
        disks = physical_disks(ROOT, DESTINATION)
        plan = archive_plan()
        if action == "plan":
            return dict(plan_only=True, files=len(plan["files"]), total_bytes=plan["total_bytes"],
                        terminal_receipts=len(plan["targets"]), physical_disks=disks)
        require(not RECEIPT.exists(), "A previous archive receipt exists")
        result = copy_plan(DESTINATION, snapshot, plan, disks)
        require(inventory() <= {row["file"] for row in plan["files"]}, "Study inventory changed during archival")
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
        parser.error("--check-source is available for verify")
    print(json.dumps(archive(args.action, args.snapshot, args.check_source), indent=2))


if __name__ == "__main__":
    main()
