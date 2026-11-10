"""Frozen 39-job 2x2 pose/intrinsics ablation; all settings rerun on one device."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time
from prior_ablation_config import CONDITIONS

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = "prior_ablation_v5"
SCENES = ["heads/seq-01", "chess/seq-03", "fire/seq-03"]
SOURCES = ["scripts/run_prior_ablation.py", "scripts/prior_ablation_config.py",
           "scripts/run_prior_ablation_batch.py", "scripts/aggregate_prior_ablation.py",
           "scripts/prior_corruptions.py", "configs/7scenes_baseline.yaml",
           "configs/diagnostic_profiles.json", "scripts/calibration_audit.py",
           "scripts/test_prior_ablation.py", "scripts/test_calibration_audit.py",
           "scripts/cloud_prior_ablation_session.sh", "scripts/smoke_prior_ablation.py"]


def plan():
    def job(scene, condition, seed, steps, pose=1., intrinsics=1.):
        return dict(run_id=f"prior5_{scene.replace('/', '_')}_{condition}_seed{seed}",
                    scene=scene, condition=condition, seed=seed, steps=steps,
                    learning_rate=.0003, pose_scale=pose, intrinsics_scale=intrinsics,
                    weight=1., profile="clean", prior_seed=1729, size=518,
                    frames=[0, 200, 400, 600, 800])
    jobs = [job(s, "zero", 42, 0) for s in SCENES]
    for seed in [42, 123, 2026]:
        block = [job(s, c, seed, 40, *v) for s in SCENES for c, v in CONDITIONS.items()]
        random.Random(8500 + seed).shuffle(block)
        jobs.extend(block)
    return dict(study_id=STUDY, jobs=jobs, max_minutes=70, per_job_seconds=360,
        purpose="Isolate camera-pose and camera-intrinsics penalty contributions at fixed LR and visual loss",
        scope="Three previously inspected development sequences; no held-out generalization claim",
        calibration="Released nominal loader K and evaluation targets remain unchanged; calibration audit is separate",
        primary_score="Mean of accuracy/base_accuracy and completeness/base_completeness; equal scene weight",
        harm="accuracy OR completeness >1% worse than same-scene fresh zero-step control",
        contrasts="Within each scene/seed: pose-only minus visual; intrinsics-only minus visual; both minus pose-only minus intrinsics-only plus visual",
        uncertainty="Three optimizer seeds coexist with unresolved execution variability; no significance or population failure-rate claim",
        reporting="All 39 planned jobs, per-seed metrics, observed ranges, factorial contrasts, runtime and memory",
        budget_cap_usd=3, sources={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in SOURCES})


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assets", type=pathlib.Path, default=pathlib.Path("/workspace/tco-assets"))
    args=parser.parse_args()
    p=plan(); manifest=ROOT/f"manifests/{STUDY}.json"
    if args.freeze and not manifest.exists(): dump(manifest,p)
    assert manifest.exists() and json.loads(manifest.read_text()) == p, "Freeze/source mismatch; preserve frozen studies"
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(p["jobs"]) == len({j["run_id"] for j in p["jobs"]}) == 39
    if args.freeze or args.dry_run:
        print(json.dumps(dict(jobs=39, updates=1440, manifest_sha256=digest))); return
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(), "Commit before execution"
    report=ROOT/"reports"/STUDY
    assert not (report/"batch_status.json").exists(), "Never overwrite an attempt"
    (ROOT/"runs/logs").mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+p["max_minutes"]*60; status=[]
    for j in p["jobs"]:
        assert not (ROOT/"runs"/j["run_id"]).exists()
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError("Batch deadline reached")
        cmd=[sys.executable,"-u","scripts/run_prior_ablation.py","--run-id",j["run_id"],
             "--study-id",STUDY,"--manifest-sha256",digest,"--scene",j["scene"],
             "--size",str(j["size"]),"--steps",str(j["steps"]),"--seed",str(j["seed"]),
             "--learning-rate",str(j["learning_rate"]),"--pose-weight-scale",str(j["pose_scale"]),
             "--intrinsics-weight-scale",str(j["intrinsics_scale"]),"--frames",*map(str,j["frames"]),
             "--data-root",str(args.assets/"data/7scenes"),
             "--dataset-cache",str(args.assets/"data/7scenes/full_sequence_metadata.npy"),
             "--checkpoint",str(args.assets/"checkpoints/VGGT-1B")]
        entry=dict(run_id=j["run_id"],status="running",utc_start=datetime.datetime.now(datetime.timezone.utc).isoformat())
        status.append(entry);dump(report/"batch_status.json",status)
        print("Starting",len(status),"/39",j["run_id"],flush=True)
        try:
            with (ROOT/"runs/logs"/(j["run_id"]+".log")).open("x") as log:
                r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining,p["per_job_seconds"]))
            assert r.returncode==0,f"Job failed: {j['run_id']}"
            record=json.loads((ROOT/"runs"/j["run_id"]/"run.json").read_text())
            assert record["status"]=="complete" and not record["research_dirty"]
            assert record["runner_sha256"]==p["sources"]["scripts/run_prior_ablation.py"]
            entry["status"]="complete"
        except BaseException as exc:
            entry.update(status="failed_or_timed_out",error=repr(exc));raise
        finally: dump(report/"batch_status.json",status)
        print("Completed",len(status),"/39",j["run_id"],flush=True)


if __name__=="__main__": main()
