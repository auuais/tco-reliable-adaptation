"""Run a paired camera-frame intervention inside the original TCO objective."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluation_stability import dump, sha
from fit_sensor_extrinsic import homogeneous

ROOT = Path(__file__).resolve().parents[1]
NAME = "sensor_tco_v17"
MANIFEST = ROOT/"manifests"/(NAME+".json")
RAW = ROOT/"reports"/NAME
SCENES = ("heads/seq-01","chess/seq-03","fire/seq-03")
CONDITIONS = ("released","frame_only","intrinsics_only","fixed_rgb")


def plan():
    jobs = []
    for turn in range(4):
        for index,scene in enumerate(SCENES):
            condition = CONDITIONS[(turn+index)%4]
            jobs.append(dict(scene=scene,condition=condition,run_id="sensor17_"+scene.replace("/","_")+"_"+condition))
    sources = ["scripts/run_sensor_tco.py","scripts/run_repeat_calibration.py","scripts/run_memory_probe.py",
               "scripts/prior_corruptions.py","scripts/fit_sensor_extrinsic.py","scripts/calibration_variants.py",
               "scripts/pointmap_camera_diagnostic.py","scripts/depth_trust.py","scripts/screen_sensor_extrinsic.py",
               "manifests/sensor_frame_v14.json"]
    return dict(study=NAME,sources={p:sha(ROOT/p) for p in sources},jobs=jobs,frames=[50,250,450,650,850],
                seed=42,steps=40,learning_rate=.0003,size=518,camera_pose_weight=1,intrinsic_weight_scale=1,
                upstream_commit="65387333877723cf99086f7f29089379dd50c59a",
                dataset_cache_sha256="ce8f28f03d960fc4fc8be5624823715822a41550bddd421af9ca1942295afb8a",
                method="Use original full TCO with the local inactive-module offload. Factor 1: left-compose the physical depth-to-RGB transform with world-to-sensor extrinsics before optimization. Factor 2: supplied K is axis-aware registration focal 525 rather than released nominal. All conditions use the same frozen base checkpoint, adapters initialized from the same seed, objective, 40 Adam updates, image data and evaluation targets.",
                scope="New five-frame offset-50 window per development sequence, already allocated to v16 image consistency. No v16 per-window scores used for design. No held-out scene, no scene-level inference from repeated frames. This is a new mechanism pilot, not a replication of old frame-index results.",
                analysis="Evaluate released and registered RGB reference separately. Report per-scene accuracy and completeness, mean normals, and mean ratios to matched FP32 initial reconstruction. For 4 adapted cells per scene compute main contrasts and interaction. Compare full correction to unchanged TCO and to post-hoc half-depth for all conditions, direct calibrated and released priors, and initial. Require >=5% lower error vs strongest comparison, no scene >1% worse and no increased harmful scenes at BOTH refs to motivate additional seeds and a noisy matched test. One pilot seed cannot release funds.",
                runtime="Separate fresh processes, disk evidence after every job, 30-minute active job limit. Preserve failures explicitly. A technically interrupted job may be relaunched once under a distinct attempt ID after diagnosis, retaining its failed trace. No repeat chosen by outcome.",
                additional_runpod_usd=0)


def run_one(job, attempt):
    import run_repeat_calibration as base
    import run_memory_probe as memory
    physical = np.array(json.loads((ROOT/"manifests/sensor_frame_v14.json").read_text())["calibration"])
    original_priors, original_dump = base.corrupt_priors, base.dump
    condition = job["condition"]
    def priors(*args):
        e,k,metadata = original_priors(*args)
        metadata["original_extrinsics_sha256"] = base.tensor_hash(e)
        if condition in ("frame_only","fixed_rgb"):
            e = (physical @ homogeneous(e))[:,:3].astype(e.dtype)
        metadata.update(sensor_condition=condition,extrinsics_sha256=base.tensor_hash(e))
        return e,k,metadata
    def save(path,content):
        if Path(path).name == "run.json":
            content.update(scope="new_development_window_sensor_frame_factorial",sensor_condition=condition,attempt=attempt)
        original_dump(path,content)
    base.corrupt_priors, base.dump = priors, save
    memory.base = base
    p = plan()
    run_id = job["run_id"]+f"_a{attempt}"
    calibration = "axis_registration" if condition in ("intrinsics_only","fixed_rgb") else "released"
    sys.argv = [__file__,"--run-id",run_id,"--study-id",NAME,"--manifest-sha256",sha(MANIFEST),"--calibration",calibration,
                "--scene",job["scene"],"--frames",*map(str,p["frames"]),"--size",str(p["size"]),"--steps",str(p["steps"]),
                "--seed",str(p["seed"]),"--learning-rate",str(p["learning_rate"]),
                "--dataset-cache",str(ROOT/"environment/development_dataset_cache_v7.npy"),"--skip-evaluation"]
    memory.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","run","worker"))
    parser.add_argument("--job",type=int)
    parser.add_argument("--attempt",type=int,choices=(1,2),default=1)
    args = parser.parse_args()
    p = plan()
    if args.action=="freeze":
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        dump(MANIFEST,p)
        return
    if p != json.loads(MANIFEST.read_text()) or sha(ROOT/"environment/development_dataset_cache_v7.npy")!=p["dataset_cache_sha256"]:
        raise ValueError("Study or data cache changed")
    upstream = subprocess.check_output(["git","-C",str(ROOT/"third_party/TCO"),"rev-parse","HEAD"],text=True).strip()
    if upstream != p["upstream_commit"]:
        raise ValueError("Unexpected TCO checkout")
    if args.action=="worker":
        run_one(p["jobs"][args.job],args.attempt)
        return
    RAW.mkdir(parents=True,exist_ok=True)
    for i,job in enumerate(p["jobs"]):
        if args.job is not None and args.job!=i:
            continue
        fn = RAW/(job["run_id"]+f"_a{args.attempt}")
        record = ROOT/"runs"/fn.name/"run.json"
        if record.exists() and json.loads(record.read_text()).get("status")=="complete":
            continue
        with fn.with_suffix(".log").open("x") as log:
            print("Starting",fn.name,flush=True)
            try:
                subprocess.run([sys.executable,"-u",__file__,"worker","--job",str(i),"--attempt",str(args.attempt)],
                               env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
                                        NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED=str(p["seed"])),
                               stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
                dump(fn.with_suffix(".json"),dict(**job,attempt=args.attempt,status="complete",record_sha256=sha(record)))
            except (OSError,subprocess.SubprocessError) as exc:
                dump(fn.with_suffix(".json"),dict(**job,attempt=args.attempt,status="interrupted",reason=str(exc)))
                raise


if __name__=="__main__":
    main()
