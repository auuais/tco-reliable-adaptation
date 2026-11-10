"""Extend the existing local offload feasibility check to forty full-size updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]
STUDY="local_memory_v3_full40"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    plan=dict(study_id=STUDY,jobs=[dict(run_id="memory3_offload_s0",steps=0),dict(run_id="memory3_offload_s40",steps=40)],
        scene="heads/seq-01",frames=[0,200,400,600,800],size=518,seed=42,learning_rate=.0003,
        purpose="RTX 3080 engineering feasibility, no adaptation-accuracy or speed-equivalence claim",
        sources={name:sha(ROOT/name) for name in ["scripts/run_memory_full40.py","scripts/run_memory_probe.py","scripts/run_experiment.py","configs/7scenes_baseline.yaml"]},
        dataset_cache_sha256=sha(ROOT/"environment/validation_dataset_cache.npy"),timeout_seconds_per_job=900)
    manifest=ROOT/"manifests"/(STUDY+".json")
    if args.freeze:
        assert not manifest.exists();dump(manifest,plan)
    assert json.loads(manifest.read_text())==plan
    if args.freeze or args.dry_run:print(json.dumps(dict(jobs=2,updates=40,manifest_sha256=sha(manifest))));return
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    out=ROOT/"reports"/STUDY;out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42")
    for job in plan["jobs"]:
        command=[sys.executable,"-u","scripts/run_memory_probe.py","--study-id",STUDY,"--manifest-sha256",sha(manifest),
            "--run-id",job["run_id"],"--scene",plan["scene"],"--frames",*map(str,plan["frames"]),"--size","518","--seed","42",
            "--steps",str(job["steps"]),"--learning-rate",".0003","--dataset-cache",str(ROOT/"environment/validation_dataset_cache.npy"),"--skip-evaluation"]
        print("Local full-size feasibility",job["run_id"],flush=True)
        with (out/(job["run_id"]+".log")).open("x") as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=plan["timeout_seconds_per_job"])


if __name__=="__main__":main()
