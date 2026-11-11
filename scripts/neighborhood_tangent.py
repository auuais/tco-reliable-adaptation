"""Use depth-aware neighborhood normals for half-weight camera corrections."""
import argparse
import hashlib
from statistics import mean

import numpy as np

import prior_noise_paired as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

repeat=paired.repeat
fusion=repeat.fusion
SEEDS=(137,*repeat.SEEDS)
DATA=RAW/"neighborhood_tangent"
SEAL=PUBLIC/"neighborhood_tangent_predictions.json"
RESULT=PUBLIC/"neighborhood_tangent_paired.json"
DESIGN=ROOT/"docs/NEIGHBORHOOD_TANGENT.md"


def neighborhood(points,z):
    normals=np.zeros_like(points,dtype=float)
    normals[1:-1,1:-1]=paired.paired.normals(points.astype(float))[0]
    h,w=z.shape
    logz=np.log(z)
    padded=np.pad(normals,((3,3),(3,3),(0,0)))
    depths=np.pad(logz,3,mode="edge")
    field=np.zeros_like(normals)
    for y in range(7):
        for x in range(7):
            other=padded[y:y+h,x:x+w]
            dz=depths[y:y+h,x:x+w]-logz
            weight=np.exp(-((y-3)**2+(x-3)**2)/8-.5*(dz/.02)**2)
            alignment=np.sign(np.sum(other*normals,axis=-1))
            field+=other*(weight*alignment)[...,None]
    return field


def plane_check():
    y,x=np.indices((21,25),dtype=float)
    rays=np.stack(((x-12)/100,(y-10)/100,np.ones_like(x)),axis=-1)
    z=2/(1-.2*rays[...,0])
    original=rays*z[...,None]
    center=np.array([.01,0,.04])
    moved=original+center
    out,_=repeat.tangent.intersect(original,z,moved,center,neighborhood(original,z))
    actual=center+rays*((2-center[2]+.2*center[0])/(1-.2*rays[...,0]))[...,None]
    np.testing.assert_allclose(out[1:-1,1:-1],actual[1:-1,1:-1],atol=1e-9)


def fixed_dir(seed):
    return fusion.DATA if seed==137 else repeat.DATA


def infer():
    load=fusion.depth.cross.temporal.load_points
    rows={}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        source=read(fusion.model.SEAL)["rows"][key]
        pose=np.array(source["fused"])
        base=load(fusion.model.DATA,source["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",base,pose[:,2,:3])+pose[:,2,3,None,None]
        fields=[neighborhood(p,d) for p,d in zip(base,z)]
        center=fusion.centers(pose)
        rows[key]={}
        for seed in SEEDS:
            old=read(repeat.SEAL)["rows"][key][str(seed)]
            fixed=load(fixed_dir(seed),old["hashes"]["fixed"])
            _,sensor,_=fusion.align_sensor(pose,np.array(old["supplied"]))
            target=center+.5*(sensor-center)
            baseline=np.stack([repeat.tangent.intersect(p,d,q,c)[0] for p,d,q,c in zip(base,z,fixed,target)])
            if fusion.prior.tensor_digest(baseline)!=old["hashes"]["coupled"]: raise ValueError("Original update differs")
            views,stats=zip(*(repeat.tangent.intersect(p,d,q,c,n) for p,d,q,c,n in zip(base,z,fixed,target,fields)))
            cloud=np.stack(views)
            h=fusion.prior.tensor_digest(cloud)
            fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=cloud))
            rows[key][str(seed)]=dict(hashes=dict(fixed=old["hashes"]["fixed"],coupled=h),statistics=stats)
    return rows


def summarize(rows):
    measured=paired.summarize(rows,SEEDS)
    old=read(paired.RESULT)["results"]
    orig=read(paired.paired.RESULT)["results"]["overall"]["coupled"]
    outcomes={}
    for seed in SEEDS:
        val=measured[str(seed)]["overall"]
        prev=orig if seed==137 else old[str(seed)]["overall"]["coupled"]
        outcomes[str(seed)]=dict(fixed=val["equal"],updated=val["coupled"],original_depth=prev["depth_mse"],
                                 original_angle=prev["angle_deg"] if seed==137 else prev["normal_deg"])
    new_excess=mean(v["updated"]["normal_deg"]-v["fixed"]["normal_deg"] for v in outcomes.values())
    old_excess=mean(v["original_angle"]-v["fixed"]["normal_deg"] for v in outcomes.values())
    checks=dict(half_excess=new_excess<=.5*old_excess,
                depth=mean(v["updated"]["depth_mse"]-v["original_depth"] for v in outcomes.values())<=0,
                all_draws=all(v["updated"]["normal_deg"]<v["original_angle"] for v in outcomes.values()))
    return dict(per_seed=outcomes,measurements=measured,excess_reduction=1-new_excess/old_excess,
                checks=checks,advance_to_surface=all(checks.values()))


if __name__=="__main__":
    import sys
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","measure","verify"))
    action=p.parse_args().action
    plane_check()
    deps=[sha(repeat.SEAL),sha(fusion.model.SEAL),hashlib.sha256(DESIGN.read_text().encode()).hexdigest()]
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Sealed source changed")
        deps=[sha(x) for x in (SEAL,paired.RESULT,paired.paired.RESULT)]
        if action=="measure":
            if RESULT.exists(): raise FileExistsError(RESULT)
            rows=paired.measure(sys.modules[__name__],SEEDS,fixed_folder=fixed_dir)
            dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
        report=read(RESULT)
        if report["inputs"]!=deps or summarize(report["rows"])!=report["results"]: raise ValueError("Paired replay differs")
        if action=="verify":
            hashes={k:{s:v["hashes"]["coupled"] for s,v in r.items()} for k,r in seal["rows"].items()}
            fusion.prior.old.verify_map_archive(PUBLIC/"neighborhood_tangent_archive.json",hashes)
        print(report["results"]["checks"],report["results"]["excess_reduction"])
