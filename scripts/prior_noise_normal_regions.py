"""Locate corresponding-normal loss inside corrected regions and across their boundaries."""
import argparse
from statistics import mean

import numpy as np

import prior_noise_paired as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import FRAMES,INDEX,PUBLIC,RAW,SCENES
from score_saved_geometry import read

repeat=paired.repeat
RESULT=PUBLIC/"prior_noise_normal_regions.json"
SEEDS=(137,*repeat.SEEDS)


def region_sums(z,k,valid,fixed,coupled,changed):
    z=np.where(valid & np.isfinite(z) & (z>0),z,0.)
    y,x=np.indices(z.shape,dtype=float)
    points=(np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(k).T)*z[...,None]
    ref,mask=paired.paired.normals(points)
    center=z[1:-1,1:-1]
    mask &= center>0
    for neighbor in (z[:-2,1:-1],z[2:,1:-1],z[1:-1,:-2],z[1:-1,2:]):
        mask &= (neighbor>0)&(np.abs(neighbor-center)<.05*center)
    n=np.sum(np.stack([changed[1:-1,1:-1],changed[:-2,1:-1],changed[2:,1:-1],changed[1:-1,:-2],changed[1:-1,2:]]),axis=0)
    groups=(mask & (n==0),mask & (n>0)&(n<5),mask & (n==5))
    stats={a:{"count":int(g.sum())} for a,g in zip(("unchanged","mixed","interior"),groups)}
    for a,cloud in (("fixed",fixed),("coupled",coupled)):
        normals,ok=paired.paired.normals(cloud)
        angles=np.where(ok,np.rad2deg(np.arccos(np.clip(np.abs(np.sum(ref*normals,axis=-1)),0,1))),90.)
        for (_,cell),g in zip(stats.items(),groups): cell[a]=float(angles[g].sum())
    return stats


def measure():
    rows=[]
    fusion=repeat.fusion
    load=fusion.depth.cross.temporal.load_points
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        source=read(fusion.model.SEAL)["rows"][key]
        base=load(fusion.model.DATA,source["hashes"]["joint"])
        original=np.array(source["fused"])
        for seed in SEEDS:
            hashes=read(repeat.SEAL)["rows"][key][str(seed)]["hashes"]
            fixed=load(fusion.DATA if seed==137 else repeat.DATA,hashes["fixed"])
            coupled=load(repeat.tangent.DATA if seed==137 else repeat.DATA,hashes["coupled"])
            for i,frame in enumerate(FRAMES):
                path=RAW/"reference"/slug/f"{frame:06d}.npz"
                if sha(path)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]: raise ValueError("Reference changed")
                e=paired.paired.poses.recover_pose(base[i],fixed[i],original[i])
                xyz=[p[i]@e[:,:3].T+e[:,3] for p in (fixed,coupled)]
                changed=np.any(fixed[i]!=coupled[i],axis=-1)
                with np.load(path) as f:
                    stats=region_sums(f["depth"].astype(float),f["k"],f["valid"],*xyz,changed)
                rows.append(dict(scene=scene,frame=frame,seed=seed,regions=stats))
    return rows


def summarize(rows):
    history={(r["scene"],r["frame"],137):r for r in read(paired.paired.RESULT)["rows"]}
    history.update({(r["scene"],r["frame"],r["seed"]):r for r in read(paired.RESULT)["rows"]})
    if len(rows)!=len(history) or len({(r["scene"],r["frame"],r["seed"]) for r in rows})!=len(history):
        raise ValueError("Incomplete target set")
    result={}
    for seed in SEEDS:
        group=[r for r in rows if r["seed"]==seed]
        for r in group:
            prior=history[r["scene"],r["frame"],seed]
            if sum(v["count"] for v in r["regions"].values())!=prior["normal_count"]: raise ValueError("Mask differs")
            for a,b in (("fixed","equal"),("coupled","coupled")):
                np.testing.assert_allclose(sum(v[a] for v in r["regions"].values()),prior["metrics"][b]["angular_sum"],rtol=1e-7)
        result[str(seed)]={}
        for label in ("unchanged","mixed","interior"):
            def total(r): return sum(v["count"] for v in r["regions"].values())
            change=mean((r["regions"][label]["coupled"]-r["regions"][label]["fixed"])/total(r) for r in group)
            fraction=mean(r["regions"][label]["count"]/total(r) for r in group)
            result[str(seed)][label]=dict(fraction=fraction,contribution_deg=change)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (repeat.SEAL,paired.RESULT,paired.paired.RESULT,INDEX)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if deps!=data["inputs"] or data["results"]!=summarize(data["rows"]): raise ValueError("Region contributions differ")
    print(data["results"])
