"""Limit render-induced range changes and score the resulting matched camera maps."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import camera_mesh_paired as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

mesh=paired.mesh
fusion=mesh.fusion
DATA=RAW/"mesh_range_limit"
SEAL=PUBLIC/"mesh_range_predictions.json"
RESULT=PUBLIC/"mesh_range_scores.json"
PIXELS=PUBLIC/"mesh_range_paired.json"
DESIGN=ROOT/"docs/CAMERA_MESH_RANGE_LIMIT.md"
ARMS=("reject","clip","taper")


def infer():
    records={}
    load=fusion.depth.cross.temporal.load_points
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        orig=read(fusion.model.SEAL)["rows"][key]
        joint=load(fusion.model.DATA,orig["hashes"]["joint"])
        fixed=load(fusion.DATA,read(fusion.SEAL)["rows"][key]["hashes"]["equal"])
        rendered=load(mesh.DATA,read(mesh.SEAL)["rows"][key]["hashes"]["equal"])
        pose=np.stack([paired.paired.poses.recover_pose(p,q,e) for p,q,e in zip(joint,fixed,np.array(orig["fused"]))])
        c=fusion.centers(pose)[:,None,None,:]
        ray=fixed-c
        fraction=np.linalg.norm(rendered-c,axis=-1)/np.linalg.norm(ray,axis=-1)-1
        np.testing.assert_allclose(c+(1+fraction[...,None])*ray,rendered,rtol=2e-5,atol=1e-6)
        keep=np.abs(fraction)<=.05
        updates={"clip":np.clip(fraction,-.05,.05),"taper":fraction*np.clip((.05-np.abs(fraction))/.04,0,1)}
        outputs={a:(fixed+ray*delta[...,None]).astype(np.float32) for a,delta in updates.items()}
        outputs["reject"]=np.where(keep[...,None],rendered,fixed)
        hashes={}
        for a,points in outputs.items():
            h=fusion.prior.tensor_digest(points)
            fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=points))
            hashes[a]=h
        records[key]=dict(hashes=hashes,rejected_fraction=float(1-keep.mean()))
    return records


def score(item):
    key,row=item
    _,v=fusion.prior.evaluate_points(key,set(row["hashes"].values()),DATA)
    return key,{a:v[h] for a,h in row["hashes"].items()}


def summarize(obs):
    from score_depth_scale_study import aggregate,HISTORICAL
    existing=read(mesh.RESULT)["results"]["references"]
    rows=[]
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        metrics={}
        for ref,g in existing.items():
            arms={a:v["raw"] for a,v in g["scenes"][scene].items()}
            arms["mesh_equal"]=arms.pop("guarded")
            arms.update({a:obs[key][a][ref] for a in ("clip","taper")})
            arms["guarded"]=obs[key]["reject"][ref]
            metrics[ref]=arms
        rows.append(dict(scene=scene,input_key=key,case="base",metrics=metrics))
    comps=tuple(a for a in arms if a not in (*HISTORICAL,"guarded"))
    return aggregate(rows,comps+("guarded",),comps)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","pixels","evaluate","verify"))
    action=p.parse_args().action
    deps=[sha(x) for x in (mesh.SEAL,fusion.SEAL,fusion.model.SEAL)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Changed input")
        pixel_deps=[sha(SEAL),sha(paired.RESULT)]
        if action=="pixels":
            if PIXELS.exists(): raise FileExistsError(PIXELS)
            rows=paired.measure(extra=lambda key:{a:(fusion.depth.cross.temporal.load_points(DATA,h),"equal")
                                              for a,h in seal["rows"][key]["hashes"].items()})
            dump(PIXELS,dict(inputs=pixel_deps,rows=rows,results=paired.summarize(rows)))
        else:
            deps=[sha(SEAL),sha(mesh.RESULT)]
            if action=="evaluate":
                if RESULT.exists(): raise FileExistsError(RESULT)
                with ProcessPoolExecutor(max_workers=3) as pool:
                    obs=dict(pool.map(score,seal["rows"].items()))
                dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
            data=read(RESULT)
            if deps!=data["inputs"] or summarize(data["observed"])!=data["results"]: raise ValueError("Surface replay differs")
            if action=="verify":
                fusion.prior.old.verify_map_archive(PUBLIC/"mesh_range_archive.json",{k:v["hashes"] for k,v in seal["rows"].items()})
                pixels=read(PIXELS)
                if pixels["inputs"]!=pixel_deps or paired.summarize(pixels["rows"])!=pixels["results"]: raise ValueError("Paired replay differs")
            print(data["results"]["checks"])
