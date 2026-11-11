"""Recover the applied rigid transforms and check camera error against registered poses."""
import argparse
from statistics import mean

import numpy as np
from scipy.spatial.transform import Rotation

import anchor_prior_fusion as fusion
from camera_shape_v10 import relative_rays
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, SCENES
from score_saved_geometry import read

RESULT = PUBLIC/"anchor_prior_pose_errors.json"
ARMS = ("joint",*fusion.ARMS)


def recover_pose(source,target,original):
    p = source[::8,::8].reshape(-1,3).astype(float)
    q = target[::8,::8].reshape(-1,3).astype(float)
    x,y = p.mean(0),q.mean(0)
    rot = fusion.model.proper((p-x).T@(q-y))
    t = y-x@rot
    np.testing.assert_allclose(p@rot+t,q,atol=2e-6,rtol=1e-6)
    center = (-original[:3,3]@original[:3,:3])@rot+t
    new_rot = original[:3,:3]@rot
    return np.column_stack((new_rot,-new_rot@center))


def measure():
    rows = {}
    for scene in SCENES:
        slug = scene.split("/")[0]
        key = slug+"_base"
        source = read(fusion.model.SEAL)["rows"][key]
        initial = np.array(source["fused"])
        base = fusion.depth.cross.temporal.load_points(fusion.model.DATA,source["hashes"]["joint"])
        rigs = {"joint":initial}
        for arm,h in read(fusion.SEAL)["rows"][key]["hashes"].items():
            points = fusion.depth.cross.temporal.load_points(fusion.DATA,h)
            rigs[arm] = np.stack([recover_pose(p,q,e) for p,q,e in zip(base,points,initial)])
        refs = []
        for frame in FRAMES:
            ref = RAW/"reference"/slug/f"{frame:06d}.npz"
            if sha(ref)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]:
                raise ValueError("Reference camera differs")
            with np.load(ref) as z:
                refs.append(z["e"][:3])
        rows[scene] = {k:v.tolist() for k,v in dict(rigs,reference=np.stack(refs)).items()}
    return rows


def summarize(rows):
    out = {}
    for scene in SCENES:
        orig_rot,orig_delta = relative_rays(np.array(rows[scene]["reference"]))
        arms = {}
        for arm in ARMS:
            rot,delta = relative_rays(np.array(rows[scene][arm]))
            angles = Rotation.from_matrix(rot@orig_rot.transpose(0,2,1)).magnitude()
            scale = max(0.,np.sum(delta*orig_delta)/np.sum(delta**2))
            arms[arm] = dict(rotation_deg=float(np.rad2deg(angles).mean()),
                             center_relative_rmse=float(np.linalg.norm(scale*delta-orig_delta)/np.linalg.norm(orig_delta)))
        out[scene] = arms
    return dict(scenes=out,overall={a:{m:mean(v[a][m] for v in out.values()) for m in out[SCENES[0]][a]} for a in ARMS})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (fusion.SEAL,fusion.model.SEAL,INDEX)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if deps!=data["inputs"]: raise ValueError("Pose provenance differs")
    analysis=summarize(data["rows"])
    for scene in SCENES:
        for arm in ARMS:
            for metric in analysis["scenes"][scene][arm]:
                for a,b in ((analysis["scenes"][scene],data["results"]["scenes"][scene]),(analysis["overall"],data["results"]["overall"])):
                    np.testing.assert_allclose(a[arm][metric],b[arm][metric],rtol=1e-9,atol=1e-9)
    print(analysis["overall"])
