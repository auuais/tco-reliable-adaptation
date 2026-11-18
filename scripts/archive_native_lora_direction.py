"""Preserve directional-study evidence with an independently verified disk copy."""
import argparse
import hashlib
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import Selection, physical_disks
from native_lora_direction_registration import MANIFEST, RAW, RESULTS, ROOT, verify_registration
from prepare_profiled_inputs import read
from profiled_artifacts import logical_inventory
from run_profiled_d0 import now, write_new
from score_native_lora_direction import SCORES, SEAL, seal_inventory


def archive(destination):
    registration = verify_registration(require_committed=True)
    if read(SEAL) != seal_inventory():
        raise ValueError("Finish and verify the sealed endpoint inventory before archiving")
    selected = Selection(ROOT)
    for path in (MANIFEST, SEAL, SCORES, RESULTS / "independent_replay.json"):
        selected.marker(path.relative_to(ROOT).as_posix())
    selected.provenance(registration)
    for base in (RAW, RESULTS):
        prefix = base.relative_to(ROOT).as_posix()
        for logical, path in logical_inventory(base).items():
            if path.suffix.lower() != ".log":
                selected.add(f"{prefix}/{logical}")
    disks = physical_disks(ROOT, destination)
    report = copy_plan(destination, "verdict-v1", selected.finish(), disks)
    manifest = destination / "snapshots/verdict-v1/manifest.json"
    receipt = dict(snapshot="verdict-v1", files_verified=report["files_verified"],
                   total_bytes=report["total_bytes"], pass_all=report["pass_all"],
                   source_compared=True, physical_disks=disks, archive_manifest=manifest.as_posix(),
                   archive_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   verified_at_utc=now())
    write_new(RESULTS / "archive_receipt.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path("E:/ResearchData/TCO/research_results/native_lora_direction_v1"))
    archive(parser.parse_args().destination)
