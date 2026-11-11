"""Couple externally updated cameras with depth that preserves local tangent planes."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import anchor_prior_gain_controls as controls
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

fusion = controls.fusion
DATA = RAW/"camera_tangent"
SEAL = PUBLIC/"camera_tangent_predictions.json"
RESULT = PUBLIC/"camera_tangent_scores.json"
DESIGN = ROOT/"docs/CAMERA_TANGENT_DEPTH.md"


def intersect(original,depth,moved,center,normal_field=None):
    base = original[1:-1,1:-1].astype(float)
    if normal_field is None:
        n = np.cross(original[1:-1,2:].astype(float)-original[1:-1,:-2],
                     original[2:,1:-1].astype(float)-original[:-2,1:-1])
    else:
        n = normal_field[1:-1,1:-1].copy()
    norm = np.linalg.norm(n,axis=-1)
    n /= np.maximum(norm,1e-15)[...,None]
    offset = moved[1:-1,1:-1].astype(float)-center
    size = np.linalg.norm(offset,axis=-1)
    ray = offset/size[...,None]
    cosine = np.sum(n*ray,axis=-1)
    length = np.divide(np.sum(n*(base-center),axis=-1),cosine,out=np.zeros_like(size),where=np.abs(cosine)>=.4)
    z = depth[1:-1,1:-1]
    smooth = np.maximum.reduce([np.abs(a-z) for a in (depth[2:,1:-1],depth[:-2,1:-1],depth[1:-1,2:],depth[1:-1,:-2])])<.05*z
    frac = length/size-1
    use = smooth & (norm>1e-15) & (np.abs(cosine)>=.4) & (np.abs(frac)<=.05) & (length>0)
    out = moved.copy()
    out[1:-1,1:-1][use] = center+length[use,None]*ray[use]
    return out,dict(pixels=int(depth.size),eligible=int(use.sum()),abs_fraction_sum=float(np.abs(frac[use]).sum()))


def plane_check():
    y,x = np.indices((13,17),dtype=float)
    directions = np.stack(((x-8)/100,(y-6)/100,np.ones_like(x)),axis=-1)
    original = (directions*2).astype(np.float32)
    z = original[...,2]
    center = np.array([.01,0,.04])
    for c in (np.zeros(3),center):
        moved = (original+c).astype(np.float32)
        output,stats = intersect(original,z,moved,c)
        if stats["eligible"]!=(13-2)*(17-2): raise ValueError("Plane mask incorrect")
        np.testing.assert_allclose(output[1:-1,1:-1],(c+(2-c[2])*directions)[1:-1,1:-1],atol=1e-6)
        np.testing.assert_array_equal(output[0],moved[0])


def infer():
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    rows = {}
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        parent=read(fusion.model.SEAL)["rows"][key]
        pose=np.array(parent["fused"])
        c=fusion.centers(pose)
        original=fusion.depth.cross.temporal.load_points(fusion.model.DATA,parent["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",original,pose[:,2,:3])+pose[:,2,3,None,None]
        row=read(fusion.SEAL)["rows"][key]
        spec=read(fusion.prior.PLAN)["priors"][slug+"_wide"]
        path=fusion.depth.capture.original.INPUT/spec["file"]
        if sha(path)!=spec["sha256"]: raise ValueError("Prior changed")
        with np.load(path) as f:
            _,sensor_c,_=fusion.align_sensor(pose,f["extrinsics"].astype(float))
        hashes,statistics={},{}
        for arm in fusion.ARMS:
            gain=np.array(row["center_gain"]) if arm=="adaptive" else np.full(len(c),.5 if arm=="equal" else 1.)
            target=c+gain[:,None]*(sensor_c-c)
            moved=fusion.depth.cross.temporal.load_points(fusion.DATA,row["hashes"][arm])
            views,stats=zip(*(intersect(p,d,q,t) for p,d,q,t in zip(original,z,moved,target)))
            points=np.stack(views)
            h=fusion.prior.tensor_digest(points)
            fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=points))
            hashes[arm],statistics[arm]=h,stats
        rows[key]=dict(hashes=hashes,statistics=statistics)
    return rows


def score(item):
    key,row=item
    _,values=fusion.prior.evaluate_points(key,set(row["hashes"].values()),DATA)
    return key,{a:values[h] for a,h in row["hashes"].items()}


def summarize(obs):
    earlier=read(fusion.RESULT)["observed"]
    base=fusion.summarize(earlier)["references"]
    gain_controls=read(controls.RESULT)["observed"]
    rows=[]
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        metrics={}
        for ref,g in base.items():
            previous=g["scenes"][scene]
            v={a:d["raw"] for a,d in previous.items()}
            v["prior_adaptive"]=v.pop("guarded")
            v.update({"gain_"+a:gain_controls[key][a][ref] for a in controls.ARMS})
            v.update({"surface_"+a:obs[key][a][ref] for a in ("equal","full_prior")})
            v["guarded"]=obs[key]["adaptive"][ref]
            metrics[ref]=v
        rows.append(dict(scene=scene,input_key=key,case="base",metrics=metrics))
    from score_depth_scale_study import HISTORICAL,aggregate
    comps=tuple(a for a in v if a not in (*HISTORICAL,"guarded"))
    return aggregate(rows,comps+("guarded",),comps)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("infer","evaluate","verify"))
    action=parser.parse_args().action
    plane_check()
    deps=[sha(x) for x in (fusion.SEAL,fusion.model.SEAL,fusion.prior.PLAN)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Seal differs")
        deps=[sha(x) for x in (SEAL,controls.RESULT,fusion.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        data=read(RESULT)
        if data["inputs"]!=deps or summarize(data["observed"])!=data["results"]: raise ValueError("Replay differs")
        if action=="verify":
            fusion.prior.old.verify_map_archive(PUBLIC/"camera_tangent_archive.json",{k:v["hashes"] for k,v in seal["rows"].items()})
        print(data["results"]["checks"])
