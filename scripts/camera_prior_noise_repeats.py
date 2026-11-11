"""Repeat the archived half-weight camera-depth experiment with new prior directions."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import camera_tangent_depth as tangent
from evaluation_stability import dump,sha
from focal_zoom_gradient import noisy_poses
from prepare_source_spacing import FRAMES,INDEX,PUBLIC,RAW,ROOT,SCENES
from score_depth_scale_study import aggregate,HISTORICAL
from score_saved_geometry import read

fusion=tangent.fusion
SEEDS=(211,307,401)
CAL=ROOT/"manifests/sensor_frame_v14.json"
DATA=RAW/"prior_noise_repeats"
DESIGN=ROOT/"docs/CAMERA_PRIOR_NOISE_REPEATS.md"
SEAL=PUBLIC/"prior_noise_predictions.json"
RESULT=PUBLIC/"prior_noise_scores.json"


def clean_rig(data,scene):
    poses=[]
    for frame in FRAMES:
        path=data/scene/f"frame-{frame:06d}.pose.txt"
        expected=read(INDEX)["frames"][scene][str(frame)]["original"]["pose.txt"]
        if sha(path)!=expected: raise ValueError("Public camera file differs")
        poses.append(np.loadtxt(path))
    source=np.stack(poses)
    rig=np.tile(np.eye(4),(len(source),1,1))
    r=source[:,:3,:3].transpose(0,2,1)
    rig[:,:3,:3]=r
    rig[:,:3,3:]=-r@source[:,:3,3:]
    return np.array(read(CAL)["calibration"])@rig


def infer(data):
    records={}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        parent=read(fusion.model.SEAL)["rows"][key]
        pose=np.array(parent["fused"])
        rot,c=pose[:,:3,:3],fusion.centers(pose)
        points=fusion.depth.cross.temporal.load_points(fusion.model.DATA,parent["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",points,pose[:,2,:3])+pose[:,2,3,None,None]
        spec=read(fusion.prior.PLAN)["priors"][slug+"_wide"]
        path=fusion.depth.capture.original.INPUT/spec["file"]
        if sha(path)!=spec["sha256"]: raise ValueError("Seed 137 supplied camera changed")
        with np.load(path) as f:
            old=f["extrinsics"]
        clean=clean_rig(data,scene)
        records[key]={}
        for seed in (137,*SEEDS):
            rig=noisy_poses(clean,scene,seed,1.,.02)[:,:3].astype(np.float32)
            if seed==137: np.testing.assert_array_equal(rig,old)
            rs,cs,_=fusion.align_sensor(pose,rig.astype(float))
            w=np.full(len(c),.5)
            fixed=fusion.blend(points,rot,c,rs,cs,w,w)
            coupled=np.stack([tangent.intersect(p,d,q,target)[0] for p,d,q,target in zip(points,z,fixed,c+.5*(cs-c))])
            hashes={}
            for arm,cloud,study in (("fixed",fixed,fusion),("coupled",coupled,tangent)):
                h=fusion.prior.tensor_digest(cloud)
                if seed==137:
                    if h!=read(study.SEAL)["rows"][key]["hashes"]["equal"]: raise ValueError("Original point map differs")
                else:
                    fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=cloud))
                hashes[arm]=h
            records[key][str(seed)]=dict(hashes=hashes,supplied=rig.tolist())
    return records


def score(item):
    key,rows=item
    hashes={h for s in SEEDS for h in rows[str(s)]["hashes"].values()}
    _,metrics=fusion.prior.evaluate_points(key,hashes,DATA)
    return key,{str(s):{arm:metrics[h] for arm,h in rows[str(s)]["hashes"].items()} for s in SEEDS}


def summarize(obs):
    panel=read(tangent.RESULT)["results"]["references"]
    result={}
    for seed in SEEDS:
        rows=[]
        for scene in SCENES:
            key=scene.split("/")[0]+"_base"
            metrics={}
            for ref,g in panel.items():
                arms={a:v["raw"] for a,v in g["scenes"][scene].items()}
                arms["old_tangent_adaptive"]=arms.pop("guarded")
                arms["repeat_fixed"]=obs[key][str(seed)]["fixed"][ref]
                arms["guarded"]=obs[key][str(seed)]["coupled"][ref]
                metrics[ref]=arms
            rows.append(dict(scene=scene,input_key=key,case="base",metrics=metrics))
        comp=tuple(a for a in arms if a not in (*HISTORICAL,"guarded"))
        full=aggregate(rows,comp+("guarded",),comp)
        comparison={}
        for ref,g in full["references"].items():
            def select(scores):
                return {k:scores[k] for k in ("joint","repeat_fixed","guarded")}
            comparison[ref]=dict(overall=select(g["overall"]),scenes={s:select(v) for s,v in g["scenes"].items()})
        result[str(seed)]=dict(checks=full["checks"],advance=full["advance"],local_gain=full["local_gain"],
                              historical_gain=full["historical_gain"],comparison=comparison)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("infer","evaluate","verify"))
    parser.add_argument("--data",type=Path)
    args=parser.parse_args()
    deps=[sha(x) for x in (fusion.model.SEAL,fusion.SEAL,tangent.SEAL,INDEX,CAL,fusion.prior.PLAN)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if args.action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer(args.data)))
    else:
        seal=read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Input identity changed")
        deps=[sha(SEAL),sha(tangent.RESULT)]
        if args.action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report=read(RESULT)
        if report["inputs"]!=deps or summarize(report["observed"])!=report["results"]: raise ValueError("Score replay differs")
        if args.action=="verify":
            hashes={k:{f"{s}_{a}":h for s in SEEDS for a,h in r[str(s)]["hashes"].items()} for k,r in seal["rows"].items()}
            fusion.prior.old.verify_map_archive(PUBLIC/"prior_noise_archive.json",hashes)
        print({s:v["advance"] for s,v in report["results"].items()})
