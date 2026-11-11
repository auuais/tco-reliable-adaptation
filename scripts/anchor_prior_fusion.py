"""Fuse a supplied camera rig with empirical variation across frozen model orders."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.transform import Rotation

import anchor_pose_surface as model
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

depth = model.depth_model
prior = model.prior
DATA = RAW/"anchor_prior_fusion"
SEAL = PUBLIC/"anchor_prior_predictions.json"
RESULT = PUBLIC/"anchor_prior_scores.json"
DESIGN = ROOT/"docs/ANCHOR_PRIOR_FUSION.md"
ARMS = ("equal","full_prior","adaptive")


def align_sensor(fused,sensor):
    rotations,c = model.align(fused,sensor,1.)
    p = c-c.mean(0)
    q = centers(fused)
    factor = np.sum(p*(q-q.mean(0)))/np.sum(p*p)
    if not np.isfinite(factor) or factor<=0: raise ValueError("Unidentifiable prior scale")
    return model.proper(rotations),q.mean(0)+factor*p,float(factor)


def blend(points,rot,c,sensor_rot,sensor_c,angular_gain,center_gain):
    relative = Rotation.from_matrix(sensor_rot@rot.transpose(0,2,1)).as_rotvec()
    new_rot = Rotation.from_rotvec(angular_gain[:,None]*relative).as_matrix()@rot
    new_c = c + center_gain[:,None]*(sensor_c-c)
    return np.stack([((p-o)@r.T@s+n).astype(np.float32)
                     for p,o,r,s,n in zip(points,c,rot,new_rot,new_c)])


def infer():
    plan = read(prior.PLAN)
    captures = read(depth.capture.RESULT)["rows"]
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    records = {}
    for scene in SCENES:
        slug = scene.split("/")[0]
        key = slug+"_base"
        primary = read(model.SEAL)["rows"][key]
        fused = np.array(primary["fused"])
        rot,c = fused[:,:3,:3],centers(fused)
        points = depth.cross.temporal.load_points(model.DATA,primary["hashes"]["joint"])
        offsets = read(depth.SEAL)["rows"][key]["log_offsets"]
        aligned = []
        for i,offset in enumerate(offsets):
            name = f"{key}_{i}"
            path = depth.capture.DATA/(name+".npz")
            if sha(path)!=captures[name]["sha256"]: raise ValueError("Capture changed")
            with np.load(path) as f:
                e = f["extrinsics"].astype(float)
            if i==0: base = e
            aligned.append(model.align(base,e,np.exp(-offset)))
        rs,cs = map(np.stack,zip(*aligned))
        np.testing.assert_allclose(model.proper(rs.sum(0)),rot,atol=1e-10)
        np.testing.assert_allclose(cs.mean(0),c,atol=1e-10)
        relative = model.proper(rs)@rot.transpose(0,2,1)
        angles = Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(len(rs),len(c))
        angles_var = (angles**2).mean(0)
        center_var = ((cs-c)**2).sum(-1).mean(0)
        specs = [plan["priors"][slug+"_"+v] for v in ("wide","tight")]
        rigs = []
        for spec in specs:
            path = depth.capture.original.INPUT/spec["file"]
            if sha(path)!=spec["sha256"]: raise ValueError("Supplied prior differs")
            with np.load(path) as f:
                rigs.append(f["extrinsics"].astype(float))
        np.testing.assert_array_equal(*rigs)
        sensor_rot,sensor_c,scale = align_sensor(fused,rigs[0])
        diameter = np.linalg.norm(sensor_c[:,None]-sensor_c[None,:],axis=-1).max()
        ar = angles_var/(angles_var+np.deg2rad(1.)**2)
        at = center_var/(center_var+(.02*diameter)**2)
        hashes = {}
        for name in ARMS:
            ga,gt = (ar,at) if name=="adaptive" else (np.full(len(c),.5 if name=="equal" else 1.),)*2
            prediction = blend(points,rot,c,sensor_rot,sensor_c,ga,gt)
            h = prior.tensor_digest(prediction)
            prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=prediction))
            hashes[name] = h
        records[key] = dict(hashes=hashes,angular_gain=ar.tolist(),center_gain=at.tolist(),prior_scale=scale)
    return records


def score(item):
    key,row = item
    _,values = prior.evaluate_points(key,set(row["hashes"].values()),DATA)
    return key,{a:values[h] for a,h in row["hashes"].items()}


def summarize(obs):
    cross = depth.cross
    poses = read(model.RESULT)["observed"]
    earlier = read(depth.RESULT)["observed"]
    controls = {"diffused":read(cross.diffusion.RESULT)["observed"],"rgb_rigid":read(cross.rgb.RESULT)["observed"],
                "two_planes":read(cross.RESULT)["observed"]}
    for source,names in ((earlier,depth.ARMS),(poses,("pose","joint")),(obs,("equal","full_prior"))):
        controls.update({a:{k:v[a] for k,v in source.items()} for a in names})
    return prior.summarize(read(prior.RESULT)["observed"],{k:v["adaptive"] for k,v in obs.items()},extra=controls)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action = p.parse_args().action
    deps = [sha(p) for p in (model.SEAL,depth.SEAL,depth.capture.RESULT,prior.PLAN)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal = read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Seal differs")
        deps = [sha(p) for p in (SEAL,model.RESULT,depth.RESULT,prior.RESULT,depth.cross.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs = dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report = read(RESULT)
        if deps!=report["inputs"] or summarize(report["observed"])!=report["results"]:
            raise ValueError("Surface scores differ")
        if action=="verify":
            prior.old.verify_map_archive(PUBLIC/"anchor_prior_archive.json",{k:v["hashes"] for k,v in seal["rows"].items()})
        print(report["results"]["checks"])
