"""Fuse camera predictions across input orders in one world coordinate system."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import anchor_ensemble_surface as depth_model
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

prior = depth_model.prior
DATA = RAW/"anchor_pose"
SEAL = PUBLIC/"anchor_pose_predictions.json"
RESULT = PUBLIC/"anchor_pose_scores.json"
DESIGN = ROOT/"docs/ANCHOR_ORDER_POSES.md"


def proper(m):
    u,_,vt = np.linalg.svd(m)
    u[...,:,-1] *= np.linalg.det(u@vt)[...,None]
    return u@vt


def align(reference, alternative, factor):
    # Column vector world transform Q makes R_alt Q^T agree with R_ref.
    ref, alt = reference[:,:3,:3], alternative[:,:3,:3]
    q = proper((ref.transpose(0,2,1)@alt).sum(axis=0))
    c = centers(alternative)
    return alt@q.T, factor*(c-c.mean(axis=0))@q.T+centers(reference).mean(axis=0)


def fuse(reference, poses, factors):
    aligned = [align(reference,p,s) for p,s in zip(poses,factors)]
    rs,cs = map(np.stack,zip(*aligned))
    rot = proper(rs.sum(axis=0))
    center = cs.mean(axis=0)
    output = np.concatenate((rot,(-rot@center[...,None])),axis=2)
    angles = np.arccos(np.clip((np.trace(proper(rs)@rot.transpose(0,2,1),axis1=-2,axis2=-1)-1)/2,-1,1))
    return output,float(np.sqrt(np.mean(np.sum((cs-center)**2,axis=-1)))),float(np.rad2deg(np.sqrt(np.mean(angles**2))))


def infer():
    capture = depth_model.capture
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    records = {}
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        _,depth,k,e,_ = prior.inputs(key,read(prior.PARENT/"predictions.json")["rows"],read(prior.PLAN))
        r = read(depth_model.SEAL)["rows"][key]
        offsets = r["log_offsets"]
        poses,shifts = [],[]
        for i,offset in enumerate(offsets):
            name = f"{key}_{i}"
            path = capture.DATA/(name+".npz")
            if sha(path)!=read(capture.RESULT)["rows"][name]["sha256"]: raise ValueError("Anchor archive differs")
            with np.load(path) as f:
                poses.append(f["extrinsics"].astype(float))
                shifts.append(np.log(f["depth"][...,0].astype(float)/depth)-offset)
        np.testing.assert_array_equal(poses[0],e)
        fused,disp,angle = fuse(e,poses,np.exp(-np.array(offsets)))
        median_z = depth*np.exp(np.median(shifts,axis=0))
        replay = prior.tensor_digest(prior.point_map(median_z,k,e,np.ones(5)))
        if replay!=r["hashes"]["median"]: raise ValueError("Depth ensemble differs")
        hashes = {}
        for arm,z in (("pose",depth),("joint",median_z)):
            points = prior.point_map(z,k,fused,np.ones(5))
            h = prior.tensor_digest(points)
            prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=points))
            hashes[arm] = h
        records[key] = dict(hashes=hashes,fused=fused.tolist(),center_rms_over_median_depth=disp/np.median(depth),
                            rotation_rms_degrees=angle)
    return records


def score(item):
    key,row = item
    _,values = prior.evaluate_points(key,set(row["hashes"].values()),DATA)
    return key,{a:values[h] for a,h in row["hashes"].items()}


def summarize(obs):
    cross = depth_model.cross
    controls = {"diffused":read(cross.diffusion.RESULT)["observed"],"rgb_rigid":read(cross.rgb.RESULT)["observed"],
                "two_planes":read(cross.RESULT)["observed"],"pose_only":{k:v["pose"] for k,v in obs.items()}}
    earlier = read(depth_model.RESULT)["observed"]
    controls.update({arm:{k:v[arm] for k,v in earlier.items()} for arm in depth_model.ARMS})
    return prior.summarize(read(prior.RESULT)["observed"],{k:v["joint"] for k,v in obs.items()},extra=controls)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action = p.parse_args().action
    deps = [sha(depth_model.SEAL),sha(depth_model.capture.RESULT),hashlib.sha256(DESIGN.read_text().encode()).hexdigest()]
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Changed sealed input")
        deps=[sha(p) for p in (SEAL,depth_model.RESULT,prior.RESULT,depth_model.cross.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report=read(RESULT)
        if report["inputs"]!=deps or summarize(report["observed"])!=report["results"]: raise ValueError("Pose score differs")
        if action=="verify":
            prior.old.verify_map_archive(PUBLIC/"anchor_pose_archive.json",{k:v["hashes"] for k,v in seal["rows"].items()})
        print(report["results"]["checks"])
