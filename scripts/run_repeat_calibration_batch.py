"""Frozen 40-job common-reference calibration and identical-seed repeat study."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from evaluation_stability import dump, sha

ROOT=Path(__file__).resolve().parents[1]
STUDY="repeat_calibration_v6"
SCENES=["heads/seq-01","chess/seq-03","fire/seq-03"]
SEEDS=[42,123,2026]
SOURCES=["scripts/run_repeat_calibration.py","scripts/run_repeat_calibration_batch.py",
         "scripts/repeatability_capture.py","scripts/calibration_variants.py","scripts/calibration_audit.py",
         "scripts/prior_ablation_config.py","scripts/prior_corruptions.py","scripts/evaluation_stability.py",
         "scripts/aggregate_repeat_calibration.py","scripts/cloud_repeat_calibration_session.sh",
         "scripts/calibration_reference_sensitivity.py",
         "scripts/smoke_repeat_calibration.py","configs/7scenes_baseline.yaml","configs/diagnostic_profiles.json"]


def plan():
    def job(scene,condition,calibration,seed,repeat=0):
        return dict(run_id=f"repeat6_{scene.replace('/','_')}_{condition}_{calibration}_seed{seed}_r{repeat}",
            scene=scene,condition=condition,calibration=calibration,seed=seed,execution_repeat=repeat,
            steps=0 if condition=="zero" else 40,pose_scale=0. if condition=="visual" else 1.,
            intrinsics_scale=0. if condition=="visual" else 1.,learning_rate=.0003,size=518,frames=[0,200,400,600,800])
    jobs=[job(s,"zero","released",42) for s in SCENES]
    for seed in SEEDS:
        block=[job(s,"both",k,seed) for s in SCENES for k in ["released","axis_nominal","axis_registration"]]
        if seed==42:
            block += [job(s,"visual","released",42,r) for s in [SCENES[0],SCENES[2]] for r in range(3)]
            block += [job(s,"both","released",42,r) for s in [SCENES[0],SCENES[2]] for r in [1,2]]
        random.Random(9600+seed).shuffle(block)
        jobs.extend(block)
    assert len(jobs)==40 and sum(j["steps"] for j in jobs)==1480
    return dict(study_id=STUDY,jobs=jobs,max_minutes=80,per_job_seconds=420,budget_cap_usd=3,
        purpose="Separate same-seed adaptation variability from prior-only calibration interventions",
        scope="Three already inspected development sequences; not held-out, not an independent replication",
        calibration_reference="All primary metrics use unchanged released nominal/uniform-resize point maps",
        secondary_reference="CPU-only scoring of every saved prediction against ONE common axis-aware 525-focal reference, including fresh controls. Recover depth with exact inverse of actual backprojection transform; verify released-reference reconstruction within 1e-6 m. Do not pool primary and sensitivity protocols.",
        calibration_comparison="Use execution_repeat=0 ONLY for all 27 paired scene/seed/calibration cells; repeats diagnose noise, never best-run selection",
        repeatability_comparison="Three fresh same-seed executions of both and visual on Heads and Fire, on one physical GPU",
        primary_score="Mean accuracy/base_accuracy and completeness/base_completeness; equal scene and seed weights",
        harm="Accuracy OR completeness exceeds 1.01 times same-scene fresh zero-step control",
        diagnostic_improvement_gate=dict(minimum_relative_score_gain=.05,no_increase_in_harmful_cells=True,
            maximum_per_scene_score_regression=.01,aggregate_gain_must_exceed_maximum_both_same_seed_score_range=True,
            scope="Conservative development-screen criterion only; does not establish publication readiness or significance"),
        later_spending="Requires redesigned method with reliable evidence; calibration repair alone is not a novel method. No later-stage spending after an unreliable diagnostic result.",
        precision="Unchanged upstream mixed precision; no determinism or precision changes mixed into these conditions",
        observation="Capture initial prediction components, initial/final LoRA tensors and Python/NumPy/Torch CPU/CUDA RNG states. Include capture overhead in timing.",
        reporting="All 40 jobs, paired metrics, within-seed raw prediction differences, RNG/adapter/prediction hashes, loss traces, runtime/memory and failures",
        sources={f:sha(ROOT/f) for f in SOURCES})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--assets",type=Path,default=Path("/workspace/tco-assets"))
    args=parser.parse_args()
    p=plan();manifest=ROOT/"manifests"/(STUDY+".json")
    if args.freeze:
        assert not manifest.exists();dump(manifest,p)
    assert json.loads(manifest.read_text())==p,"Frozen source/design mismatch"
    digest=sha(manifest)
    assert len({j["run_id"] for j in p["jobs"]})==40
    if args.freeze or args.dry_run:
        print(json.dumps(dict(jobs=40,updates=1480,manifest_sha256=digest)));return
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"Commit first"
    directory=ROOT/"reports"/STUDY;status_path=directory/"batch_status.json"
    assert not status_path.exists(),"Preserve attempts"
    deadline=time.monotonic()+p["max_minutes"]*60
    status=[];identities=set()
    (ROOT/"runs/logs").mkdir(parents=True,exist_ok=True)
    for j in p["jobs"]:
        remaining=deadline-time.monotonic()
        assert remaining>0,"Batch deadline reached"
        cmd=[sys.executable,"-u","scripts/run_repeat_calibration.py","--run-id",j["run_id"],
             "--study-id",STUDY,"--manifest-sha256",digest,"--scene",j["scene"],
             "--calibration",j["calibration"],"--execution-repeat",str(j["execution_repeat"]),
             "--size",str(j["size"]),"--steps",str(j["steps"]),"--seed",str(j["seed"]),
             "--learning-rate",str(j["learning_rate"]),"--pose-weight-scale",str(j["pose_scale"]),
             "--intrinsics-weight-scale",str(j["intrinsics_scale"]),"--frames",*map(str,j["frames"]),
             "--data-root",str(args.assets/"data/7scenes"),
             "--dataset-cache",str(args.assets/"data/7scenes/full_sequence_metadata.npy"),
             "--checkpoint",str(args.assets/"checkpoints/VGGT-1B")]
        entry=dict(run_id=j["run_id"],status="running");status.append(entry);dump(status_path,status)
        print("Starting",len(status),"/40",j["run_id"],flush=True)
        started=time.monotonic()
        try:
            with (ROOT/"runs/logs"/(j["run_id"]+".log")).open("x") as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                               timeout=min(remaining,p["per_job_seconds"]))
            record=json.loads((ROOT/"runs"/j["run_id"]/"run.json").read_text())
            assert record["status"]=="complete" and not record["research_dirty"]
            assert record["runner_sha256"]==p["sources"]["scripts/run_repeat_calibration.py"]
            identities.add(record["gpu_identity"])
            assert len(identities)==1,"Physical GPU changed within study"
            entry["status"]="complete"
        except BaseException as error:
            entry.update(status="failed_or_timed_out",error=repr(error));raise
        finally:
            entry["seconds"]=time.monotonic()-started;dump(status_path,status)
        print("Completed",len(status),"/40",flush=True)


if __name__=="__main__":main()
