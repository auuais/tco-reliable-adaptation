"""CPU-only common-reference sensitivity; never feeds reference depth to adaptation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from calibration_variants import calibration_prior
from evaluation_stability import dump, sha
from repeatability_capture import array_digest

ROOT=Path(__file__).resolve().parents[1]
STUDY="repeat_calibration_v6"


def alternate_points(arrays):
    """Recover stored camera depth with the exact inverse of its actual transform.

    The released loader's transpose inverse is intentionally retained for the
    final camera-to-world transform; no pose orthogonalization is introduced.
    """
    points=arrays["reference_points"]
    _,height,width,_=points.shape
    original_k=arrays["intrinsics"]
    new_k=calibration_prior(original_k,width,height,"axis_registration")
    u,v=np.meshgrid(np.arange(width),np.arange(height))
    alternate=[];errors=[]
    for old,e,k,new in zip(points,arrays["extrinsics"],original_k,new_k):
        # Exactly the transform used by upstream closed_form_inverse_se3.
        c2w=np.eye(4);c2w[:3,:3]=e[:3,:3].T;c2w[:3,3]=-e[:3,:3].T@e[:3,3]
        exact_w2c=np.linalg.inv(c2w)
        camera=old@exact_w2c[:3,:3].T+exact_w2c[:3,3]
        depth=camera[...,2].astype(np.float32)
        def unproject(intrinsic):
            camera_points=np.stack([(u-intrinsic[0,2])*depth/intrinsic[0,0],
                                    (v-intrinsic[1,2])*depth/intrinsic[1,1],depth],axis=-1).astype(np.float32)
            return camera_points@c2w[:3,:3].T+c2w[:3,3]
        restored=unproject(k)
        error=float(np.max(np.abs(restored-old)))
        assert error<1e-6,"Reference depth recovery mismatch"
        errors.append(error);alternate.append(unproject(new))
    return np.stack(alternate),new_k,errors


def worker(evidence,job,output):
    import random
    import torch
    import open3d as o3d
    from run_prior_ablation import evaluate
    folder=evidence/"runs"/job["run_id"]
    record=json.loads((folder/"run.json").read_text())
    arrays=dict(np.load(evidence/record["evaluation_inputs_relative"],allow_pickle=False))
    assert array_digest(arrays)==record["evaluation_tensor_sha256"]
    points,k,errors=alternate_points(arrays)
    seed=job["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);o3d.utility.random.seed(seed)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    metrics=evaluate(np.load(folder/"predictions.npy",allow_pickle=False),dict(
        pointclouds=points,valid_mask=arrays["valid_mask"],images=torch.from_numpy(arrays["images"])),output)
    dump(output/"sensitivity.json",dict(run_id=job["run_id"],metrics=metrics,
        original_prediction_sha256=sha(folder/"predictions.npy"),original_reference_tensor_sha256=record["evaluation_tensor_sha256"],
        alternate_reference_tensor_sha256=array_digest(dict(points=points,intrinsics=k,mask=arrays["valid_mask"])),
        released_reference_recovery_max_absolute_m=errors,common_reference="axis_aware_registration_RGB_focal_525",
        warning="Assumed calibration sensitivity, not independently measured physical calibration; all conditions use this same alternate reference"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/calibration_reference_sensitivity_v6")
    args=parser.parse_args()
    manifest=json.loads((ROOT/"manifests"/(STUDY+".json")).read_text())
    assert sha(Path(__file__))==manifest["sources"]["scripts/calibration_reference_sensitivity.py"]
    if args.worker is not None:
        job=manifest["jobs"][args.worker];worker(args.evidence_root,job,args.output/job["run_id"]);return
    args.output.mkdir(parents=True,exist_ok=False)
    for index,job in enumerate(manifest["jobs"]):
        env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
        env["PYTHONHASHSEED"]=str(job["seed"])
        print("Scoring common alternate reference",index+1,"/40",job["run_id"],flush=True)
        with (args.output/(job["run_id"]+".log")).open("x") as log:
            subprocess.run([sys.executable,"scripts/calibration_reference_sensitivity.py","--evidence-root",str(args.evidence_root),
                            "--worker",str(index),"--output",str(args.output)],cwd=ROOT,env=env,check=True,timeout=600,
                           stdout=log,stderr=subprocess.STDOUT)
    rows=[];zero={};signatures={}
    for job in manifest["jobs"]:
        result=json.loads((args.output/job["run_id"]/"sensitivity.json").read_text())
        sig=result["alternate_reference_tensor_sha256"]
        assert signatures.setdefault(job["scene"],sig)==sig
        row={**job,**result};rows.append(row)
        if job["condition"]=="zero":zero[job["scene"]]=row["metrics"]
    for row in rows:
        base=zero[row["scene"]];keys=["accuracy_mean_m","completeness_mean_m"]
        row["score"]=sum(row["metrics"][k]/base[k] for k in keys)/2
        row["harmful"]=any(row["metrics"][k]>1.01*base[k] for k in keys)
    conditions={k:dict(score=float(np.mean([r["score"] for r in rows if r["condition"]=="both" and r["calibration"]==k and r["execution_repeat"]==0])),
                       harmful_cells=sum(r["harmful"] for r in rows if r["condition"]=="both" and r["calibration"]==k and r["execution_repeat"]==0))
                for k in ["released","axis_nominal","axis_registration"]}
    dump(ROOT/"results"/STUDY/"reference_sensitivity.json",dict(completed_evaluations=len(rows),rows=rows,conditions=conditions,
        common_reference_verified=True,manifest_sha256=sha(ROOT/"manifests"/(STUDY+".json")),
        interpretation="A separate common-reference sensitivity analysis. Do not pool with primary released-reference metrics."))


if __name__=="__main__":main()
