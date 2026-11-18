"""Back up the compact developer-run evidence and verify the second-disk copy."""
import argparse
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import Selection, physical_disks
from native_lora_decoder_groups import (ARCHIVED, BRIDGE, MANIFEST, PREVIOUS_RAW, PREVIOUS_SCORES,
                                        PREVIOUS_SEAL, PUBLIC, RAW, ROOT, SCORES, SEAL, archive_case,
                                        build_seal, endpoint_path, registered)
from run_profiled_d0 import now, read, sha, write_new


def archive(destination):
    contract = registered()
    seal = read(SEAL)
    if seal != build_seal(contract):
        raise ValueError("Selection seal changed")
    replay = PUBLIC / "independent_replay.json"
    if not read(replay)["verified"] or not read(replay)["private_scorer_view_rows"]:
        raise ValueError("Complete the independent private replay before archiving")
    selection = Selection(ROOT)

    def add(path, digest=None):
        selection.add(path.relative_to(ROOT).as_posix(), expected=digest)

    selection.provenance(contract)
    for marker in (MANIFEST, PREVIOUS_SEAL, PREVIOUS_SCORES, SEAL, SCORES, replay):
        selection.marker(marker.relative_to(ROOT).as_posix())
    for entry in contract["development_cases"]:
        job, env = entry["id"], entry["env"]
        case = seal["cases"][job]
        path = RAW / "cases" / job / case["attempt"]
        prior_path, prior = archive_case(job)
        add(path / "choice.json", case["receipt_sha256"])
        add(prior_path / "choice.json", prior["choice_sha256"])
        for label, digest in case["factor_files"].items():
            add(prior_path / f"{label}.npz", digest)
        for name in ("images.npz", "earlier.npz", "later.npz"):
            add(PREVIOUS_RAW / "intake" / env / name)
        for terminal in (RAW / "launches" / job).glob("*/terminal.json"):
            add(terminal)
            add(terminal.with_name("start.json"))
        for key, row in case["states"].items():
            base = endpoint_path(path, prior_path, row)
            add(base / "native.npz", row["native_sha256"])
            add(base / "trace.json", row["trace_sha256"])
            add(base / "complete.json")
            if job == BRIDGE and key in ARCHIVED:
                historical = prior["states"][ARCHIVED[key]]
                add(prior_path / ARCHIVED[key] / "native.npz", historical["native_sha256"])
        if case["bridge"]:
            repeat = case["bridge"]["repeat"]
            add(path / "repeat_full/native.npz", repeat["native_sha256"])
            add(path / "repeat_full/trace.json", repeat["trace_sha256"])
            add(path / "repeat_full/complete.json")
    disks = physical_disks(ROOT, destination)
    snapshot = "development-v1"
    report = copy_plan(destination, snapshot, selection.finish(), disks)
    proof = destination / "snapshots" / snapshot / "manifest.json"
    receipt = dict(snapshot=snapshot, files_verified=report["files_verified"],
                   total_bytes=report["total_bytes"], pass_all=report["pass_all"],
                   manifest_sha256=sha(proof), physical_disks=disks, verified_at_utc=now())
    write_new(PUBLIC / "archive_receipt.json", receipt)
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path("E:/ResearchData/TCO/research_results/native_lora_decoder_groups_v1"))
    archive(parser.parse_args().destination)
