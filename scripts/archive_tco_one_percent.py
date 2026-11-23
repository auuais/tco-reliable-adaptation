"""Preserve and verify the finished TCO study on the second local disk."""

from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import Selection, physical_disks
from replay_tco_one_percent import publish_or_replay
from run_profiled_d0 import sha, write_new
from tco_one_percent_protocol import CONFIRM_MANIFEST, MANIFEST, PRIVATE, PUBLIC, ROOT, STUDY, TIME_MANIFEST
from tco_one_percent_protocol import is_success, verify


def snapshot():
    registration = verify()
    publish_or_replay()
    selection = Selection(ROOT)
    for marker in (MANIFEST, TIME_MANIFEST, CONFIRM_MANIFEST, *sorted(PUBLIC.glob("*.json"))):
        if marker.is_file():
            selection.marker(marker.relative_to(ROOT).as_posix())
    for name, checksum in registration["source_sha256_lf"].items():
        selection.add(name, text_expected=checksum)
    for terminal in sorted((PRIVATE / "launches").glob("*/*/terminal.json")):
        relative = terminal.parent.relative_to(ROOT)
        receipt = selection.marker((relative / "terminal.json").as_posix())
        selection.add((relative / "start.json").as_posix())
        output = PRIVATE / "cases" / receipt["job"] / receipt["attempt"]
        if is_success(receipt):
            selection.marker((output / "complete.json").relative_to(ROOT).as_posix())
            for path in sorted(output.iterdir()):
                if path.name != "complete.json":
                    selection.add(path.relative_to(ROOT).as_posix())
    for terminal in sorted((PRIVATE / "allocator_probes").glob("*/terminal.json")):
        relative = terminal.parent.relative_to(ROOT)
        record = selection.marker(terminal.relative_to(ROOT).as_posix())
        selection.add((relative / "start.json").as_posix())
        if record["result"]:
            output = PRIVATE / "cases" / record["result"]["id"] / terminal.parent.name / "complete.json"
            selection.marker(output.relative_to(ROOT).as_posix())
    archive = Path("E:/ResearchData/TCO/research_results") / STUDY
    disks = physical_disks(ROOT, archive)
    result = copy_plan(archive, "study-v1", selection.finish(), disks)
    receipt = dict(files_verified=result["files_verified"], total_bytes=result["total_bytes"],
                   manifest_sha256=sha(archive / "snapshots/study-v1/manifest.json"),
                   physical_disks=disks, pass_all=result["pass_all"], rental_compute_usd=0)
    write_new(PUBLIC / "archive_receipt.json", receipt)
    print(receipt)


if __name__ == "__main__":
    snapshot()
