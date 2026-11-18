"""Copy registered compensation evidence to a second disk and verify the bytes."""
import argparse
import json
from pathlib import Path
import shutil
import traceback

from archive_profiled_evidence import (ROOT, Selection, copy_file, dump_new, no_active_target,
    physical_disks, require, resolve_logical, sha, snapshot_path, verify_snapshot)
from profiled_compensation_protocol import DESIGN, ENGINEERING, EVALUATION, EXECUTION, PUBLIC
from register_profiled_compensation import verify_release
from run_profiled_batch import verify_completed
from run_profiled_d0 import now
from score_profiled_compensation import ACCESS, ATTEMPT, SCORES, SEAL, SAMPLES, build_seal

SELF = "scripts/archive_profiled_compensation.py"
VERDICT_FILES = (SEAL, ACCESS, SCORES, SAMPLES, f"{PUBLIC}/source_replay.json",
                 f"{PUBLIC}/independent_replay.json")


def selection_for(registry, *, chosen, verdict):
    plan = Selection(ROOT)
    jobs = {job["run_id"]: job for job in registry["jobs"]}
    if verdict:
        sealed = plan.marker(SEAL)
        require(sealed == build_seal(registry), "Global prediction seal changed")
        chosen = list(jobs)
        expected = plan.marker(SCORES)["sample_replay"]
        require(expected["file"] == SAMPLES, "Score sample bundle has an unexpected path")
        plan.add(SAMPLES, expected["sha256"])
        for name in VERDICT_FILES:
            plan.add(name)
        for name in ("source_replay.json", "independent_replay.json"):
            proof = plan.marker(f"{PUBLIC}/{name}")
            require(proof["pass_all"] is True, "Score replay did not pass: " + name)
        official = plan.marker(f"{PUBLIC}/source_replay.json")
        independent = plan.marker(f"{PUBLIC}/independent_replay.json")
        require(official["report"]["sha256"] == independent["report_sha256"] == plan.files[SCORES]["sha256"]
                and official["sample_replay"]["sha256"] == independent["samples_sha256"] == expected["sha256"],
                "Replays do not bind this verdict and compact sample bundle")
    require(chosen and len(set(chosen)) == len(chosen) and set(chosen) <= set(jobs),
            "Select explicit registered cases or the sealed verdict")
    no_active_target(ROOT, set(chosen))
    for name in (SELF, DESIGN, ENGINEERING, EXECUTION, EVALUATION,
                 "scripts/archive_profiled_evidence.py", "scripts/profiled_artifacts.py"):
        plan.add(name)
    plan.provenance(registry)
    for name, digest in registry["upstream_sha256_lf"].items():
        plan.add(name, text_expected=digest)

    for run in chosen:
        job = jobs[run]
        complete = verify_completed(ROOT, ROOT / EXECUTION, registry, job, ATTEMPT)
        require(complete is not None, "Case has no verified completion: " + run)
        report = plan.marker(complete["complete"], complete["complete_sha256"])
        plan.tree(f"{job['raw_attempt_parent']}/{ATTEMPT}", report["raw_files"])
        public_files = dict(report["public_files"], **{"complete.json": complete["complete_sha256"]})
        plan.tree(f"{job['public_attempt_parent']}/{ATTEMPT}", public_files)
        launcher = "tco-profiled-compensation-" + run.lower().replace("_", "-")
        receipt_path = f"{PUBLIC}/launches/{launcher}/terminal.json"
        terminal = plan.marker(receipt_path)
        require(terminal["verified_stopped"] is True and terminal["state"]["ExitCode"] == 0,
                "Missing terminal proof of successful shutdown: " + run)
        for name in ("started.json", "ownership.json"):
            plan.add(f"{PUBLIC}/launches/{launcher}/{name}")
        plan.targets.append(dict(run_id=run, attempt=ATTEMPT, complete=complete["complete"],
                                 terminal=receipt_path))
    return plan.finish()


def copy_plan(destination, snapshot, plan, disks):
    target = snapshot_path(ROOT, destination, snapshot)
    require(not target.exists(), "Preserve existing snapshots; choose a fresh ID")
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(shutil.disk_usage(ancestor).free >= plan["total_bytes"] + 16*1024**2,
            "Insufficient archive disk space")
    data = target / "data"
    data.mkdir(parents=True, exist_ok=False)
    intent = dict(schema="profiled_archive_intent_v1", snapshot=snapshot,
                  source_root=str(ROOT), archive_root=str(destination), physical_disks=disks,
                  started_at_utc=now(), selection=plan)
    dump_new(target/"intent.json", intent)
    copied = 0
    try:
        for row in plan["files"]:
            copy_file(resolve_logical(ROOT, row["file"], must_exist=True),
                      resolve_logical(data, row["file"]), row)
            copied += 1
        require(all(sha(resolve_logical(ROOT, name, must_exist=True)) == expected
                    for name, expected in plan["completion_guards"].items()),
                "A completion marker changed during archival")
        report = dict(schema="profiled_evidence_archive_v1", snapshot=snapshot, source_root=str(ROOT),
                      archive_root=str(destination), physical_disks=disks,
                      selection=plan, intent_sha256=sha(target/"intent.json"), files_verified=copied,
                      total_bytes=plan["total_bytes"], completed_at_utc=now(),
                      arrays_deserialized=False, pass_all=True)
        dump_new(target/"manifest.json", report)
        return verify_snapshot(destination, snapshot, source_root=ROOT)
    except Exception as exc:
        dump_new(target/"failure.json", dict(pass_all=False, files_copied=copied, error=str(exc),
                                            traceback=traceback.format_exc(), failed_at_utc=now()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "copy", "verify"))
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--case", action="append")
    p.add_argument("--verdict", action="store_true")
    p.add_argument("--receipt", type=Path)
    a = p.parse_args()
    destination = a.destination.resolve()
    target = snapshot_path(ROOT, destination, a.snapshot)
    if a.action == "verify":
        require(not a.case and not a.verdict and a.receipt is None, "Verify uses only the existing snapshot")
        result = verify_snapshot(destination, a.snapshot, source_root=ROOT)
    else:
        require(bool(a.case) != a.verdict, "Specify --case or --verdict")
        registry, _ = verify_release()
        plan = selection_for(registry, chosen=a.case, verdict=a.verdict)
        disks = physical_disks(ROOT, destination)
        if a.action == "plan":
            require(not target.exists() and a.receipt is None, "Plan requires a fresh snapshot ID")
            result = dict(plan_only=True, cases=len(plan["targets"]), files=len(plan["files"]),
                          total_bytes=plan["total_bytes"], physical_disks=disks)
        else:
            if a.receipt is not None:
                receipt = (ROOT / a.receipt).resolve()
                require(receipt.is_relative_to((ROOT / PUBLIC).resolve()) and not receipt.exists(),
                        "Use a new receipt under the compensation results directory")
            result = copy_plan(destination, a.snapshot, plan, disks)
            if a.receipt is not None:
                dump_new(receipt, dict(**result, cases=len(plan["targets"]), physical_disks=disks,
                         archive_manifest=(target/"manifest.json").as_posix(),
                         archive_manifest_sha256=sha(target/"manifest.json"), verified_at_utc=now()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
