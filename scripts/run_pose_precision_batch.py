"""Freeze and execute a bounded local development study; no cloud spending."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluation_stability import dump, sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = "pose_precision_v7_local"
SCENES = ["heads/seq-01", "chess/seq-03", "fire/seq-03"]
SOURCES = ["scripts/run_pose_precision_batch.py", "scripts/run_pose_precision_study.py",
           "scripts/run_pose_precision.py", "scripts/run_memory_probe.py", "scripts/run_repeat_calibration.py",
           "scripts/repeatability_capture.py", "scripts/prior_ablation_config.py", "scripts/calibration_variants.py",
           "scripts/prior_corruptions.py", "scripts/aggregate_pose_precision.py", "configs/7scenes_baseline.yaml"]


def make_plan():
    jobs = []
    def add(scene, mode, seed, repeat, steps):
        jobs.append(dict(run_id=f"precision7_{scene.replace('/', '_')}_{mode}_seed{seed}_r{repeat}_s{steps}",
                         scene=scene, mode=mode, seed=seed, execution_repeat=repeat, steps=steps))
    for scene in SCENES:
        add(scene, "released", 42, 0, 0)
    for scene in SCENES:
        for seed in [42, 123, 2026]:
            for mode in ["released", "float32"]:
                add(scene, mode, seed, 0, 40)
    for scene in [SCENES[0], SCENES[2]]:
        for repeat in [1, 2]:
            for mode in ["released", "float32"]:
                add(scene, mode, 42, repeat, 40)
    return dict(study_id=STUDY, jobs=jobs, frames=[0,200,400,600,800], size=518, learning_rate=.0003,
                calibration="released", pose_scale=1., intrinsics_scale=1., local_gpu="NVIDIA GeForce RTX 3080",
                sources={name:sha(ROOT/name) for name in SOURCES},
                dataset_cache_sha256="fdd04a89625cc7e2d2fba3727dba21cd2a108e26eb31e7fc43cd66eecf9cc2c5",
                timeout_seconds_per_job=900, batch_timeout_seconds=14400,
                selection="Use repeat 0 for all three seeds; never select best execution. Repeats are a separate diagnostic.",
                gate=dict(min_relative_gain=.05, harm_not_increased=True, max_scene_relative_regression=.01,
                          absolute_gain_exceeds_both_modes_max_repeat_range=True),
                purpose="Engineering precision control before any novel method; same local offload/capture path for both modes",
                later_allocations_automatically_released=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args();plan=make_plan();manifest=ROOT/"manifests"/(STUDY+".json")
    if args.freeze:
        assert not manifest.exists();dump(manifest,plan)
    assert json.loads(manifest.read_text())==plan
    if args.freeze or args.dry_run:
        print(json.dumps(dict(jobs=len(plan["jobs"]),updates=sum(j["steps"] for j in plan["jobs"]),manifest_sha256=sha(manifest))));return
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    assert sha(ROOT/"environment/validation_dataset_cache.npy")==plan["dataset_cache_sha256"]
    out=ROOT/"reports"/STUDY;out.mkdir(parents=True,exist_ok=False)
    statuses=[];started=time.monotonic()
    for index,job in enumerate(plan["jobs"]):
        remaining=plan["batch_timeout_seconds"]-(time.monotonic()-started)
        assert remaining>0,"Local batch deadline reached"
        env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED=str(job["seed"]))
        command=[sys.executable,"-u","scripts/run_pose_precision_study.py","--pose-precision",job["mode"],
            "--study-id",STUDY,"--manifest-sha256",sha(manifest),"--run-id",job["run_id"],"--scene",job["scene"],
            "--frames",*map(str,plan["frames"]),"--size","518","--seed",str(job["seed"]),"--steps",str(job["steps"]),
            "--execution-repeat",str(job["execution_repeat"]),"--calibration","released","--learning-rate",".0003",
            "--dataset-cache",str(ROOT/"environment/validation_dataset_cache.npy")]
        print("Local precision",index+1,"/",len(plan["jobs"]),job["run_id"],flush=True)
        item=dict(run_id=job["run_id"],status="running",command=command);statuses.append(item);dump(out/"batch_status.json",statuses)
        try:
            with (out/(job["run_id"]+".log")).open("x") as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
                               timeout=min(plan["timeout_seconds_per_job"],remaining))
            item["status"]="complete"
        except Exception as exc:
            item.update(status="failed",error=repr(exc));raise
        finally:
            dump(out/"batch_status.json",statuses)


if __name__=="__main__":main()
