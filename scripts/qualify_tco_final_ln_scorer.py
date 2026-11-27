"""Validate the original CPU scorer against known similarity geometry on Stanford development pixels."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree
import torch

from run_experiment import evaluate
from run_tco_courtyard_camera import seed_model
from score_tco_final_ln import digest, require, PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    scene=json.loads(args.manifest.read_text())["scene"]
    require(scene["scene"]=="apt1/kitchen" and not torch.cuda.is_available(),"Use CPU and the old development room")
    require(digest(args.model)==scene["model_npz_sha256"] and digest(args.reference)==scene["reference_npz_sha256"],
            "Unexpected development packet")
    with np.load(args.model,allow_pickle=False) as r:
        images=torch.from_numpy(r["images"])
    with np.load(args.reference,allow_pickle=False) as r:
        target=r["pointclouds"]
        mask=r["valid_mask"]
    # Fixed lattice keeps this intake check inexpensive. Production scoring uses all valid pixels.
    sparse=np.zeros_like(mask)
    sparse[:,::7,::7]=mask[:,::7,::7]
    require(sparse.sum()>1000 and all(m.sum()>100 for m in sparse),"Insufficient calibration support")
    angle=.41
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    world_shift=np.array([.31,-.19,.27])
    scale=1.13
    generated=np.zeros_like(target,dtype=np.float32)
    generated[sparse]=(scale*(target[sparse]@rotation.T)+world_shift).astype(np.float32)
    reference=dict(pointclouds=target,valid_mask=sparse,images=images)
    evidence={}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("affine","localized_deformation"):
            pred=generated.copy()
            if name=="localized_deformation":
                hit=np.zeros_like(sparse)
                hit[1,:,::14]=sparse[1,:,::14]
                pred[hit]+=np.array([.03,-.015,.04],dtype=np.float32)
            folder=Path(tmp)/name
            folder.mkdir()
            seed_model(42)
            metrics=evaluate(pred,reference,folder)
            require(metrics["protocol"]==PROTOCOL and int(sparse.sum())==metrics["valid_points"],
                    "Wrong evaluation path")
            acc, comp = metrics["accuracy_mean_m"], metrics["completeness_mean_m"]
            rr=np.array(metrics["similarity_rotation"])
            tt=np.array(metrics["similarity_translation"]).reshape(3)
            registered=metrics["similarity_scale"]*pred[sparse].astype(np.float64)@rr.T+tt
            icp=np.array(metrics["icp_transformation"])
            registered=registered@icp[:3,:3].T+icp[:3,3]
            gt=target[sparse].astype(np.float64)
            kd_acc=float(cKDTree(gt).query(registered)[0].mean())
            kd_comp=float(cKDTree(registered).query(gt)[0].mean())
            require(abs(acc-kd_acc)<2e-7 and abs(comp-kd_comp)<2e-7,"Directed cKDTree oracle mismatch")
            if name=="affine":
                require(max(acc,comp)<1e-5 and abs(metrics["similarity_scale"]-1./scale)<1e-5
                        and np.max(np.abs(rr-rotation.T))<1e-5,"Known similarity not recovered")
            else:
                require(min(acc,comp)>5e-5 and abs(acc-comp)>1e-6,"Localized distortion sanity check failed")
            evidence[name]=dict(accuracy_mm=1000*acc,completeness_mm=1000*comp,
                                independent_accuracy_mm=1000*kd_acc,independent_completeness_mm=1000*kd_comp,
                                recovered_scale=metrics["similarity_scale"])
    result=dict(room=scene["scene"],synthetic=True,model_predictions=False,evaluator=PROTOCOL,
                mask_description="known development valid pixels on fixed 7-pixel 2-D lattice",
                pixels=int(sparse.sum()),input_file_sha256=digest(args.model),reference_file_sha256=digest(args.reference),
                reference_manifest_sha256=digest(args.manifest),data=evidence,passed=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print("Verified signed known similarity and both directional distances",flush=True)


if __name__=="__main__":
    main()
