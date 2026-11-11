"""Measure paired depth and camera-frame normal error after joint pose/depth adaptation."""
import argparse
import hashlib
from statistics import mean

import numpy as np

import anchor_prior_pose_errors as poses
import camera_tangent_depth as tangent
import cross_view_axial as axial
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

RESULT = PUBLIC/"tangent_paired_geometry.json"
DESIGN = ROOT/"docs/TANGENT_PAIRED_GEOMETRY.md"
SEEDS = (101,211,307,401)


def normals(points):
    normal = np.cross(points[1:-1,2:]-points[1:-1,:-2],points[2:,1:-1]-points[:-2,1:-1])
    length = np.linalg.norm(normal,axis=2)
    return normal/np.maximum(length,1e-15)[...,None],length>1e-15


def frame_score(z,k,valid,maps,scale,parent="equal",candidate="coupled"):
    valid = valid & np.isfinite(z) & (z>0)
    z = np.where(valid,z,0.)
    y,x = np.indices(z.shape,dtype=float)
    xyz = (np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(k).T)*z[...,None]
    ref,mask = normals(xyz)
    central = z[1:-1,1:-1]
    mask &= valid[1:-1,1:-1]
    for neighbor in (z[2:,1:-1],z[:-2,1:-1],z[1:-1,2:],z[1:-1,:-2]):
        mask &= (neighbor>0)&(np.abs(neighbor-central)<.05*central)
    metrics = {}
    for arm,points in maps.items():
        n,usable = normals(points)
        dot = np.abs(np.einsum("ijk,ijk->ij",ref,n))
        angle = np.where(usable,np.rad2deg(np.arccos(np.clip(dot,0,1))),90.)
        error = np.log(scale*points[...,2][valid]/z[valid])
        metrics[arm] = dict(depth_sse=float(error@error),angular_sum=float(angle[mask].sum()),
                            degenerate=int((mask & ~usable).sum()))
    initial = maps[parent][...,2][valid]
    u = np.log(maps[candidate][...,2][valid]/initial)
    error = np.log(scale*initial/z[valid])
    return dict(depth_count=int(valid.sum()),normal_count=int(mask.sum()),metrics=metrics,
                cross=float(u@error),energy=float(u@u))


def measure():
    fusion = tangent.fusion
    load = fusion.depth.cross.temporal.load_points
    records = []
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        joint=load(fusion.model.DATA,read(fusion.model.SEAL)["rows"][key]["hashes"]["joint"])
        equal=load(fusion.DATA,read(fusion.SEAL)["rows"][key]["hashes"]["equal"])
        coupled=load(tangent.DATA,read(tangent.SEAL)["rows"][key]["hashes"]["equal"])
        rig=read(poses.RESULT)["rows"][scene]
        scale=read(axial.SCALES)["batches"][key]["scale_to_metres"]
        for i,frame in enumerate(FRAMES):
            path=RAW/"reference"/slug/f"{frame:06d}.npz"
            if sha(path)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]:
                raise ValueError("Changed reference")
            maps={}
            for arm,cloud,cam in (("joint",joint,"joint"),("equal",equal,"equal"),("coupled",coupled,"equal")):
                e=np.array(rig[cam][i])
                maps[arm]=cloud[i]@e[:3,:3].T+e[:3,3]
            field=np.log(maps["coupled"][...,2]/maps["equal"][...,2])
            np.testing.assert_allclose(maps["equal"]*np.exp(field)[...,None],maps["coupled"],rtol=1e-5,atol=1e-6)
            active=field!=0
            for seed in SEEDS:
                digest=hashlib.sha256(f"{scene}/{frame}/{seed}".encode()).digest()
                permuted=np.zeros_like(field)
                permuted[active]=np.random.default_rng(int.from_bytes(digest[:8],"big")).permutation(field[active])
                np.testing.assert_array_equal(np.sort(permuted[active]),np.sort(field[active]))
                maps[str(seed)]=maps["equal"]*np.exp(permuted)[...,None]
            with np.load(path) as f:
                obs=frame_score(f["depth"].astype(float),f["k"],f["valid"],maps,scale)
            records.append(dict(scene=scene,frame=frame,changed=int(active.sum()),**obs))
    return records


def summarize(rows):
    if [(r["scene"],r["frame"]) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError("Unexpected targets")
    for r in rows:
        a,b=r["metrics"]["equal"],r["metrics"]["coupled"]
        np.testing.assert_allclose(b["depth_sse"],a["depth_sse"]+2*r["cross"]+r["energy"],rtol=1e-10)
    def group(items):
        return {arm:dict(depth_mse=mean(r["metrics"][arm]["depth_sse"]/r["depth_count"] for r in items),
                         angle_deg=mean(r["metrics"][arm]["angular_sum"]/r["normal_count"] for r in items),
                         degenerate=sum(r["metrics"][arm]["degenerate"] for r in items))
                for arm in ("joint","equal","coupled",*map(str,SEEDS))}
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r["scene"]==s]) for s in SCENES})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (tangent.SEAL,tangent.fusion.SEAL,tangent.fusion.model.SEAL,poses.RESULT,INDEX,axial.SCALES)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if deps!=data["inputs"] or summarize(data["rows"])!=data["results"]: raise ValueError("Geometry replay differs")
    print(data["results"]["overall"])
