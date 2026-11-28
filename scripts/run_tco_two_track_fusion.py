"""Isolated CPU prediction and surface scoring for Track B's three-run development test."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from tco_final_ln_protocol import read, seal, sha
from tco_two_track_fusion import fuse,unproject


SEEDS = (42,43,44)
STUDY = "tco_two_track_v1"


def require(condition,message):
    if not condition:
        raise ValueError(message)


def signature(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        digest.update(repr((name,a.shape,a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def model_inputs(room):
    path = Path("/model")/(room["slug"]+".npz")
    require(sha(path)==room["input_file_sha256"],"Model input bytes changed")
    with np.load(path,allow_pickle=False) as f:
        images,poses,k=(f[key] for key in ("images","extrs","intrs"))
        require(set(f.files)=={"images","extrs","intrs"},"Unexpected input array")
    require(signature(dict(rgb=images,poses=poses,intrinsics=k))==room["input_sha256"],"Model input signature changed")
    return images,poses


def prediction(path):
    with np.load(path,allow_pickle=False) as f:
        return dict(depth=f["depth"][0,...,0],extrinsics=f["extrinsics"][0],intrinsics=f["intrinsics"][0])


def persist_array(path,value):
    if path.is_file():
        require(np.array_equal(value,np.load(path,allow_pickle=False)),f"Changed prediction {path.name}")
    else:
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.save(stream,value,allow_pickle=False)
        tmp.replace(path)
    return sha(path)


def infer():
    require(not Path("/reference").exists() and not Path("/data").exists(),"Reference volume exposed to fusion")
    require(os.getenv("CUDA_VISIBLE_DEVICES")=="","Expected CPU-only isolated process")
    plan=read("/sources/inputs.json")
    outputs=Path("/outputs")
    for room in plan["rooms"]:
        target=outputs/room["slug"]
        target.mkdir(exist_ok=True,parents=True)
        if (target/"prediction.json").exists():
            receipt=read(target/"prediction.json")
            require(receipt["source_lock_sha256"]==sha("/sources/inputs.json") and
                    all(sha(target/(k+".npy"))==v for k,v in receipt["points_sha256"].items()),
                    "Changed sealed prediction or sources")
            continue
        _,poses=model_inputs(room)
        sources=[]
        native_check=[]
        for arch in room["archives"]:
            path=Path("/sources")/arch["file"]
            require(sha(path)==arch["digest"],"Changed staged source")
            src=prediction(path)
            with np.load(path,allow_pickle=False) as f:
                canonical=f["canonical"]
            max_err=0.
            for z,k,e,p in zip(src["depth"],src["intrinsics"],src["extrinsics"],canonical):
                offset=-(e[:3,:3].T @ e[:3,3])
                measured=unproject(z.astype(float),k,e[:3,:3],offset)
                err=float(np.max(np.abs(measured-p)))
                require(np.allclose(measured,p,atol=5e-6,rtol=1e-5),"Native FP32 unprojection bridge failed")
                max_err=max(max_err,err)
            native_check.append(max_err)
            sources.append(src)
        start=time.perf_counter()
        points,geometry=fuse(sources,poses)
        fusion_seconds=time.perf_counter()-start
        control,sham_receipt=fuse([sources[0]]*3,poses)
        difference=control["fusion"].astype(float)-points["project0"].astype(float)
        sham_abs=float(np.abs(difference).max())
        require(np.allclose(control["fusion"],points["project0"],atol=2e-6,rtol=1e-6),"Identity triplet failed")
        points["triplicate_anchor"]=control["fusion"]
        hashes={k:persist_array(target/(k+".npy"),a) for k,a in points.items()}
        seal(target/"prediction.json",dict(room=room["room"],input_sha256=room["input_sha256"],seeds=list(SEEDS),
            anchor_seed=SEEDS[0],source_lock_sha256=sha("/sources/inputs.json"),geometry=geometry,
            sham_geometry=sham_receipt,sham_max_abs_m=sham_abs,native_unprojection_max_abs=native_check,
            points_sha256=hashes,fusion_cpu_seconds=fusion_seconds,emitted_points=int(np.prod(points["fusion"].shape[:3])),
            inference_reference_mounted=False,rental_compute_usd=0))
        print("Sealed",room["room"],hashes["fusion"],flush=True)


def score(independent):
    import torch
    from run_experiment import evaluate
    from run_tco_courtyard_camera import seed_model
    require(not torch.cuda.is_available(),"CPU scoring requires no GPU")
    plan=read("/sources/inputs.json")
    for room in plan["rooms"]:
        images,_=model_inputs(room)
        folder=Path("/points")/room["slug"]
        receipt=read(folder/"prediction.json")
        require(receipt["source_lock_sha256"]==sha("/sources/inputs.json"),"Prediction intake changed")
        refpath=Path("/reference")/(room["slug"]+".npz")
        require(sha(refpath)==room["reference_file_sha256"],"Baseline reference bytes changed")
        with np.load(refpath,allow_pickle=False) as f:
            mask,truth=f["valid_mask"],f["pointclouds"]
        require(signature(dict(mask=mask,surface=truth))==room["reference_sha256"],"Wrong truth or validity mask")
        data=dict(images=torch.from_numpy(images),valid_mask=mask,pointclouds=truth)
        archive={i:a for i,a in zip(SEEDS,room["archives"])}
        specs=[("native",s) for s in SEEDS]+[("reprojected",s) for s in SEEDS]+[("triplicate_anchor",42),("fusion",None)]
        if independent:
            specs=[("native",42),("reprojected",43),("triplicate_anchor",42),("fusion",None)] if room["room"]=="chess/seq-03" else []
        for arm,seed in specs:
            root=Path("/scores")/("independent" if independent else "first")/room["slug"]
            name=arm+("_"+str(seed) if seed is not None else "")
            dest=root/(name+".json")
            if dest.exists():
                continue
            if arm=="native":
                path=Path("/sources")/archive[seed]["file"]
                pred_sha=sha(path)
                require(pred_sha==archive[seed]["digest"],"Native prediction changed")
                with np.load(path,allow_pickle=False) as f:
                    points=f["canonical"]
            else:
                key="project"+str(SEEDS.index(seed)) if arm=="reprojected" else arm
                path=folder/(key+".npy")
                pred_sha=sha(path)
                require(pred_sha==receipt["points_sha256"][key],"Prediction digest changed")
                points=np.load(path,allow_pickle=False)
            require(points.dtype==np.float32 and points.shape==truth.shape and np.isfinite(points).all(),"Unexpected output support")
            start=time.perf_counter()
            seed_model(seed or SEEDS[0])
            with tempfile.TemporaryDirectory() as tmp:
                metrics=evaluate(points,data,Path(tmp))
            require(metrics["valid_points"]==int(mask.sum()) and metrics["protocol"]=="upstream_Umeyama_then_rigid_ICP_0.1m_then_nearest_neighbor","Scorer or support changed")
            row=dict(room=room["room"],arm=arm,seed=seed,seeds=list(SEEDS) if arm=="fusion" else [seed],
                     accuracy_mm=metrics["accuracy_mean_m"]*1000,completeness_mm=metrics["completeness_mean_m"]*1000,
                     valid_points=metrics["valid_points"],emitted_points=receipt["emitted_points"],
                     input_sha256=room["input_sha256"],reference_sha256=room["reference_sha256"],
                     predictions_sha256=pred_sha,source_lock_sha256=receipt["source_lock_sha256"],
                     evaluator_sha256=sha(Path(__file__).parent/"run_experiment.py"),evaluator=metrics["protocol"],
                     cpu_score_seconds=time.perf_counter()-start,independent=independent,rental_compute_usd=0)
            dest.parent.mkdir(exist_ok=True,parents=True)
            seal(dest,row)
            print("Scored",room["room"],name,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("infer","score","independent"))
    action=parser.parse_args().stage
    if action=="infer":
        infer()
    else:
        score(action=="independent")
