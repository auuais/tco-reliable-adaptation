"""Measure camera-frame depth and normals after RGB reprojection fitting."""
import argparse
from statistics import mean
import numpy as np
from scipy.spatial.transform import Rotation
import joint_rgb_reprojection as study
import tangent_paired_geometry as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import FRAMES,INDEX,PUBLIC,RAW,SCENES
from score_saved_geometry import read

RESULT=PUBLIC/"joint_reprojection_paired.json"


def measure():
    result=[]
    fusion=study.fusion
    load=fusion.depth.cross.temporal.load_points
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        parent=read(fusion.model.SEAL)["rows"][key]
        e=np.array(parent["fused"])
        c=fusion.centers(e)
        base=load(fusion.model.DATA,parent["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",base,e[:,2,:3])+e[:,2,3,None,None]
        gauge=np.median(z)
        metric_scale=read(paired.axial.SCALES)["batches"][key]["scale_to_metres"]
        cameras={"baseline":e}
        maps={"baseline":base}
        for arm,s in read(study.SEAL)["rows"][key]["arms"].items():
            p=np.array(s["proposed"]) if s["accepted"] else np.zeros((5,8))
            r=e[:,:3,:3]@Rotation.from_rotvec(p[:,:3]).as_matrix().transpose(0,2,1)
            cameras[arm]=np.concatenate((r,-r@(c+gauge*p[:,3:6])[...,None]),axis=2)
            maps[arm]=load(study.DATA,s["hash"])
            actual=np.einsum("nhwj,nj->nhw",maps[arm],r[:,2,:])+cameras[arm][:,2,3,None,None]
            np.testing.assert_allclose(actual,z*np.exp(p[:,6,None,None]+p[:,7,None,None]*np.clip(gauge/z-1,-1,1)),rtol=2e-6,atol=1e-6)
        for i,frame in enumerate(FRAMES):
            path=RAW/"reference"/slug/f"{frame:06d}.npz"
            if sha(path)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]: raise ValueError("Reference changed")
            xyz={a:cloud[i]@cameras[a][i,:3,:3].T+cameras[a][i,:3,3] for a,cloud in maps.items()}
            with np.load(path) as f:
                obs=paired.frame_score(f["depth"].astype(float),f["k"],f["valid"],xyz,metric_scale,parent="baseline",candidate="joint_prior")
            result.append(dict(scene=scene,frame=frame,**obs))
    return result


def summarize(rows):
    history=read(paired.RESULT)["rows"]
    if [(r["scene"],r["frame"]) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]: raise ValueError("Missing targets")
    for row,old in zip(rows,history):
        if row["metrics"]["baseline"]!=old["metrics"]["joint"]: raise ValueError("Baseline differs")
    def group(items):
        return {a:dict(depth_mse=mean(r["metrics"][a]["depth_sse"]/r["depth_count"] for r in items),
                       angle_deg=mean(r["metrics"][a]["angular_sum"]/r["normal_count"] for r in items)) for a in ("baseline",*study.ARMS)}
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r["scene"]==s]) for s in SCENES})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (study.SEAL,paired.RESULT,INDEX,paired.axial.SCALES)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if data["inputs"]!=deps or summarize(data["rows"])!=data["results"]: raise ValueError("Paired replay differs")
    print(data["results"]["overall"])
