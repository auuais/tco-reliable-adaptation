"""Transport depth across camera changes by nearest-hit rendering of the predicted mesh."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import anchor_prior_pose_errors as poses
import camera_tangent_depth as tangent
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

fusion=tangent.fusion
DATA=RAW/"camera_mesh_render"
SEAL=PUBLIC/"camera_mesh_predictions.json"
RESULT=PUBLIC/"camera_mesh_scores.json"
DESIGN=ROOT/"docs/CAMERA_MESH_RENDER.md"


def scene_for(vertices,triangles):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices,np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(triangles,np.uint32)))
    return scene


def mesh(points,depth):
    index=np.arange(depth.size,dtype=np.uint32).reshape(depth.shape)
    a,b=index[:-1,:-1],index[:-1,1:]
    c,d=index[1:,:-1],index[1:,1:]
    faces=np.concatenate((np.stack((a,b,d),axis=-1).reshape(-1,3),np.stack((a,d,c),axis=-1).reshape(-1,3)))
    zs=depth.ravel()[faces]
    valid=(zs.min(axis=1)>0) & (np.ptp(zs,axis=1)<=.05*zs.min(axis=1))
    return scene_for(points.reshape(-1,3),faces[valid])


def render(scene,parent,center):
    import open3d as o3d
    rays=np.empty((*parent.shape[:2],6),np.float32)
    rays[...,:3]=center
    rays[...,3:]=parent-center
    t=scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    hit=np.isfinite(t)&(t>0)
    out=parent.copy()
    out[hit]=rays[hit,:3]+t[hit,None]*rays[hit,3:]
    return out,dict(pixels=int(hit.size),hits=int(hit.sum()),mean_abs_relative_change=float(np.abs(t[hit]-1).mean()))


def check_renderer():
    y,x=np.indices((20,24),dtype=float)
    d=np.stack(((x-12)/80,(y-10)/80,np.ones_like(x)),axis=-1)
    # The plane equation is z - 0.2*x = 2.
    p=(d*(2/(1-.2*d[...,0]))[...,None]).astype(np.float32)
    tree=mesh(p,p[...,2])
    for center in (np.zeros(3),np.array([.009,0,.003])):
        parent=(center+d*2).astype(np.float32)
        out,_=render(tree,parent,center)
        t=(2-center[2]+.2*center[0])/(1-.2*d[...,0])
        np.testing.assert_allclose(out[2:-2,2:-2],(center+d*t[...,None])[2:-2,2:-2],atol=1e-5)
    identity,_=render(tree,p,np.zeros(3))
    np.testing.assert_allclose(identity,p,atol=1e-5)
    vertices=np.array([[-1,-1,3],[1,-1,3],[0,1,3],[-1,-1,1],[1,-1,1],[0,1,1]])
    overlap=scene_for(vertices,[[0,1,2],[3,4,5]])
    out,stats=render(overlap,np.array([[[0,0,2],[10,0,2]]],dtype=np.float32),np.zeros(3))
    np.testing.assert_allclose(out,[[[0,0,1],[10,0,2]]],atol=1e-6)
    assert stats["hits"]==1


def infer():
    check_renderer()
    records={}
    load=fusion.depth.cross.temporal.load_points
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        orig=read(fusion.model.SEAL)["rows"][key]
        pose=np.array(orig["fused"])
        base=load(fusion.model.DATA,orig["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",base,pose[:,2,:3])+pose[:,2,3,None,None]
        trees=[mesh(p,v) for p,v in zip(base,z)]
        hashes,stats={},{}
        for arm,h in read(fusion.SEAL)["rows"][key]["hashes"].items():
            fixed=load(fusion.DATA,h)
            cams=np.stack([poses.recover_pose(p,q,e) for p,q,e in zip(base,fixed,pose)])
            views,counts=zip(*(render(tree,q,c) for tree,q,c in zip(trees,fixed,fusion.centers(cams))))
            cloud=np.stack(views)
            digest=fusion.prior.tensor_digest(cloud)
            fusion.prior.write_arrays(DATA/"points"/(digest+".npz"),dict(points=cloud))
            hashes[arm],stats[arm]=digest,counts
        records[key]=dict(hashes=hashes,statistics=stats)
    return records


def score(item):
    key,row=item
    _,v=fusion.prior.evaluate_points(key,set(row["hashes"].values()),DATA)
    return key,{a:v[h] for a,h in row["hashes"].items()}


def summarize(obs):
    from score_depth_scale_study import HISTORICAL,aggregate
    old=read(tangent.RESULT)["results"]["references"]
    rows=[]
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        metrics={}
        for ref,g in old.items():
            arms={a:v["raw"] for a,v in g["scenes"][scene].items()}
            arms["surface_adaptive"]=arms.pop("guarded")
            for a,b in (("equal","guarded"),("adaptive","mesh_adaptive"),("full_prior","mesh_full")):
                arms[b]=obs[key][a][ref]
            metrics[ref]=arms
        rows.append(dict(scene=scene,input_key=key,case="base",metrics=metrics))
    comps=tuple(a for a in arms if a not in (*HISTORICAL,"guarded"))
    return aggregate(rows,comps+("guarded",),comps)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action=p.parse_args().action
    deps=[sha(x) for x in (fusion.SEAL,fusion.model.SEAL)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Changed prediction sources")
        deps=[sha(SEAL),sha(tangent.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report=read(RESULT)
        if deps!=report["inputs"] or summarize(report["observed"])!=report["results"]: raise ValueError("Changed surface comparison")
        if action=="verify":
            fusion.prior.old.verify_map_archive(PUBLIC/"camera_mesh_archive.json",{k:v["hashes"] for k,v in seal["rows"].items()})
        print(report["results"]["checks"])
