"""Verify a second-disk copy of the registered TCO benchmark evidence."""
import argparse
from pathlib import Path

from archive_profiled_compensation import copy_plan
from archive_profiled_evidence import Selection, physical_disks
from run_profiled_d0 import read, sha, write_new
from tco_frozen_prefix_protocol import MANIFEST, PRIVATE, PUBLIC, ROOT, STUDY, collect, decision, verify
from bound_tco_frozen_prefix import bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    contract = verify()
    rows_file, decision_file = (("partial_evidence.json", "partial_bounds.json") if args.partial
                                else ("evidence.json", "decision.json"))
    evidence = read(PUBLIC/rows_file)
    recompute = bound if args.partial else decision
    if evidence != collect(contract) or recompute(contract, evidence) != read(PUBLIC/decision_file):
        raise ValueError("Benchmark evidence differs from the registered decision")
    plan = Selection(ROOT)
    for marker in (MANIFEST, PUBLIC/rows_file, PUBLIC/decision_file, PUBLIC/"engineering.json"):
        plan.marker(marker.relative_to(ROOT).as_posix())
    for name in ("allocator_diagnostics.json", "allocator_sensitivity_decision.json"):
        marker = PUBLIC/name
        if marker.is_file():
            plan.marker(marker.relative_to(ROOT).as_posix())
    for job in (row["id"] for row in contract["jobs"]):
        for terminal in (PRIVATE/"launches"/job).glob("*/terminal.json"):
            plan.add(terminal.relative_to(ROOT).as_posix())
            receipt = read(terminal)
            if receipt["exit_code"] != 0 or receipt["error"] or not receipt["stopped"]:
                continue
            folder = PRIVATE/"cases"/job/receipt["attempt"]
            for name in ("prediction.npz", "complete.json", "metrics.json"):
                plan.add((folder/name).relative_to(ROOT).as_posix())
    for terminal in (PRIVATE/"allocator_diagnostics").glob("*/*/terminal.json"):
        plan.add(terminal.relative_to(ROOT).as_posix())
        receipt = read(terminal)
        if receipt["result"]:
            folder = PRIVATE/"cases"/receipt["job"]/receipt["attempt"]
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
