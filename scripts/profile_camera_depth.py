"""Profile pose shifts during joint fitting and test retaining only the inferred depth."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

import joint_rgb_reprojection as fitted
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

DATA=RAW/"profile_camera_depth"
SEAL=PUBLIC/"profile_camera_predictions.json"
RESULT=PUBLIC/"profile_camera_scores.json"
DESIGN=ROOT/"docs/PROFILE_CAMERA_FOR_DEPTH.md"
ARMS=("joint_depth","joint_pose","joint_prior_depth","joint_prior_pose")


def camera_frame(cloud,rot,center):
    return np.einsum("nij,nhwj->nhwi",rot,cloud-center[:,None,None,:])


def infer():
    rows={}
    fusion=fitted.fusion
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        original=read(fusion.model.SEAL)["rows"][key]
        e=np.array(original["fused"])
        c=fusion.centers(e)
        base=fusion.depth.cross.temporal.load_points(fusion.model.DATA,original["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",base,e[:,2,:3])+e[:,2,3,None,None]
        scale=np.median(z)
        xyz=camera_frame(base,e[:,:3,:3],c)
        rows[key]={}
        for method in ("joint","joint_prior"):
            old=read(fitted.SEAL)["rows"][key]["arms"][method]
            p=np.array(old["proposed"]) if old["accepted"] else np.zeros((len(c),8))
            complete=fitted.apply_update(base,c,z,scale,p)
            if fusion.prior.tensor_digest(complete)!=old["hash"]: raise ValueError("Joint reconstruction differs")
            r=e[:,:3,:3]@Rotation.from_rotvec(p[:,:3]).as_matrix().transpose(0,2,1)
            center=c+scale*p[:,3:6]
            updated=camera_frame(complete,r,center)
            for name in ("depth","pose"):
                params=p.copy()
                if name=="depth": params[:,:6]=0
                else: params[:,6:]=0
                points=fitted.apply_update(base,c,z,scale,params)
                observed=camera_frame(points,e[:,:3,:3],c) if name=="depth" else camera_frame(points,r,center)
                expected=updated if name=="depth" else xyz
                np.testing.assert_allclose(observed,expected,atol=1e-6,rtol=1e-6)
                h=fusion.prior.tensor_digest(points)
                fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=points))
                rows[key][method+"_"+name]=dict(hash=h,max_camera_frame_difference=float(np.abs(observed-expected).max()))
    return rows


def score(item):
    key,row=item
    _,vals=fitted.fusion.prior.evaluate_points(key,{r["hash"] for r in row.values()},DATA)
    return key,{a:vals[r["hash"]] for a,r in row.items()}


def summarize(obs):
    from score_depth_scale_study import aggregate,HISTORICAL
    panel=read(fitted.RESULT)["results"]["references"]
    rows=[]
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        metrics={}
        for ref,g in panel.items():
            arms={a:v["raw"] for a,v in g["scenes"][scene].items()}
            arms["reproject_joint_prior"]=arms.pop("guarded")
            for a in ARMS:
                arms["guarded" if a=="joint_prior_depth" else "profile_"+a]=obs[key][a][ref]
            metrics[ref]=arms
        rows.append(dict(scene=scene,input_key=key,case="base",metrics=metrics))
    comp=tuple(a for a in arms if a not in (*HISTORICAL,"guarded"))
    return aggregate(rows,comp+("guarded",),comp)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    act=p.parse_args().action
    deps=[sha(fitted.SEAL),sha(fitted.fusion.model.SEAL),hashlib.sha256(DESIGN.read_text().encode()).hexdigest()]
    if act=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Sealed inputs changed")
        deps=[sha(SEAL),sha(fitted.RESULT)]
        if act=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        data=read(RESULT)
        if data["inputs"]!=deps or summarize(data["observed"])!=data["results"]: raise ValueError("Profile comparison differs")
        if act=="verify":
            hashes={k:{a:v["hash"] for a,v in r.items()} for k,r in seal["rows"].items()}
            fitted.fusion.prior.old.verify_map_archive(PUBLIC/"profile_camera_archive.json",hashes)
        print(data["results"]["checks"])
