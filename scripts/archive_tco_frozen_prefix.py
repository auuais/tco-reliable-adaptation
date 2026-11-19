"""Verify a second-disk copy of the registered TCO benchmark evidence."""
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import Selection, physical_disks
from run_profiled_d0 import read, sha, write_new
from tco_frozen_prefix_protocol import MANIFEST, PRIVATE, PUBLIC, ROOT, STUDY, collect, decision, verify


def main():
    contract = verify()
    evidence = read(PUBLIC/"evidence.json")
    if evidence != collect(contract) or decision(contract, evidence) != read(PUBLIC/"decision.json"):
        raise ValueError("Benchmark evidence differs from the registered decision")
    plan = Selection(ROOT)
    for marker in (MANIFEST, PUBLIC/"evidence.json", PUBLIC/"decision.json"):
        plan.marker(marker.relative_to(ROOT).as_posix())
    for row in evidence["rows"]:
        job = row["id"]
        for terminal in (PRIVATE/"launches"/job).glob("*/terminal.json"):
            plan.add(terminal.relative_to(ROOT).as_posix())
            receipt = read(terminal)
            if receipt["exit_code"] != 0 or receipt["error"] or not receipt["stopped"]:
                continue
            folder = PRIVATE/"cases"/job/receipt["attempt"]
            for name in ("prediction.npz", "complete.json", "metrics.json"):
                plan.add((folder/name).relative_to(ROOT).as_posix())
    destination = Path("E:/ResearchData/TCO/research_results")/STUDY
    disks = physical_disks(ROOT, destination)
    snapshot = "development-v1"
    result = copy_plan(destination, snapshot, plan.finish(), disks)
    manifest = destination/"snapshots"/snapshot/"manifest.json"
    receipt = dict(files_verified=result["files_verified"], total_bytes=result["total_bytes"],
                   manifest_sha256=sha(manifest), physical_disks=disks, pass_all=result["pass_all"])
    write_new(PUBLIC/"archive_receipt.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
