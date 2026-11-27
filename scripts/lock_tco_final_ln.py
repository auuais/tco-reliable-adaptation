"""Seal all prospective five-view Stanford trials and outcome-selected development arms."""
import argparse
from pathlib import Path

from tco_final_ln_protocol import LOCK, MANIFEST, PREP, PUBLIC, RAW, ROOT, SEEDS, STUDY, job, matrix, read, seal, sha


def room_packet(row, dataset):
    out = dict(room=row["scene"],environment=row["environmental_site"],dataset=dataset,
               slug=row["scene"].replace("/","__"),frames=row["frames"],input_sha256=row["input_sha256"],
               reference_sha256=row["reference_sha256"],input_file_sha256=row["model_npz_sha256"],
               reference_file_sha256=row["reference_npz_sha256"])
    for directory,key in (("model","input_file_sha256"),("reference","reference_file_sha256")):
        if sha(PREP/directory/(out["slug"]+".npz")) != out[key]:
            raise ValueError("Packet seal disagreement "+out["room"])
    return out


def lock_primary():
    if not read(RAW/"qualification_pass.json")["passed"]:
        raise ValueError("Numerical qualification is pending")
    data_file = PUBLIC/"stanford_inputs.json"
    dataset = read(data_file)
    if dataset["exposure"]["unresolved_mentions"]:
        raise ValueError("Exposure audit unresolved")
    rooms = [room_packet(r,"stanford12") for _,r in sorted(dataset["rooms"].items())]
    if len(rooms) != len({r["room"] for r in rooms}):
        raise ValueError("Duplicate room")
    files = [MANIFEST,data_file,ROOT/"configs/7scenes_baseline.yaml",ROOT/"scripts/run_experiment.py",
             ROOT/"scripts/run_tco_render_lifetime.py",ROOT/"scripts/tco_render_lifetime.py",
             ROOT/"scripts/run_tco_full40_recompute.py",ROOT/"scripts/tco_frozen_prefix.py",
             ROOT/"third_party/TCO/tco_vggt_lora.py",ROOT/"third_party/TCO/utils/interfaces.py",
             ROOT/"third_party/TCO/evaluation/mv_recon/utils.py"]
    files += [ROOT/"scripts"/name for name in ("run_tco_final_ln_model.py","tco_final_ln_policy.py","tco_final_ln_raster.py",
              "score_tco_final_ln.py","replay_tco_final_ln.py","prepare_tco_final_ln_data.py","tco_final_ln_protocol.py")]
    record = dict(study=STUDY,dataset="stanford12",rooms=rooms,seeds=SEEDS,qualification_sha256=sha(RAW/"qualification_pass.json"),
                  pinned_files={p.relative_to(ROOT).as_posix():sha(p) for p in files},primary=matrix(rooms,["L10","LN10","L40"],"primary"),
                  anchor=[job(r,42,"ZERO","anchor") for r in rooms],input_archive_sha256=dataset["archive_sha256"],
                  exclusions=dataset.get("exclusions",[]),full_room_coverage_eligible=len(rooms)>=6,
                  arms=["L10","LN10","L40"])
    seal(LOCK,record)
    print("Locked rooms",len(rooms),"matched trials",len(record["primary"]),"anchors",len(record["anchor"]),sha(LOCK))


def lock_followup(decision):
    results = read(decision)
    if results["dataset"] != "stanford12" or not results.get("formal_coverage"):
        raise ValueError("Full primary coverage required to select a scientific follow-up")
    if results["joint_verdict"] == "pass":
        branch = "all_pass"
    else:
        grades = results["decisions_on_complete_rooms"]
        passed = all(grades[name] == "pass" for name in ("accuracy_superiority_vs_short","completeness_vs_short"))
        branch = "equal_update_pass_only" if passed else "other"
    initial = read(MANIFEST)
    arms = initial["followups"][branch]
    record = dict(study=STUDY,stage="followup",branch=branch,primary_decision_sha256=sha(decision),arms=arms,
                  jobs=matrix(initial["development_rooms"],arms,"followup"))
    seal(ROOT/"manifests"/(STUDY+"_followup.json"),record)
    print("Locked follow-up",branch,arms,len(record["jobs"]))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("stanford","followup"))
    p.add_argument("--decision", type=Path)
    args=p.parse_args()
    if args.action=="stanford":
        lock_primary()
    elif args.decision:
        lock_followup(args.decision)
    else:
        p.error("followup requires --decision")
