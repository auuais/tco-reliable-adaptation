"""Optimize coupled camera and smooth depth changes against disjoint RGB feature tracks."""
import argparse
import hashlib
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import camera_mesh_range_limit as previous
import rgb_rigid_view as rgb
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

fusion=previous.fusion
DATA=RAW/"joint_rgb_reprojection"
SEAL=PUBLIC/"joint_reprojection_predictions.json"
RESULT=PUBLIC/"joint_reprojection_scores.json"
DESIGN=ROOT/"docs/JOINT_RGB_REPROJECTION.md"
ARMS=("pose","depth","joint","joint_prior")
UNIT=np.array([np.deg2rad(1.)]*3+[.01]*3+[.02]*2)


def sample_fold(pairs,z,k,scale):
    fields=[]
    for i,j,x,y in pairs:
        for a,b,xy,match in ((i,j,x,y),(j,i,y,x)):
            ray,usable=rgb.sample(xy,z[a],k[a],np.eye(3))
            if usable.any():
                fields.append((np.full(usable.sum(),a),np.full(usable.sum(),b),ray[usable]/scale,match[usable]))
    if not fields: return None
    return tuple(np.concatenate(col) for col in zip(*fields))


def robust(r):
    # Signed square root of twice the soft-L1 cost, with scale measured in pixels.
    return r*np.sqrt(2/(np.sqrt(1+(r/3)**2)+1))


def project(params,fold,rot,centers,k):
    a,b,rays,observed=fold
    basis=np.clip(1/rays[:,2]-1,-1,1)
    corrected=rays*np.exp(params[a,6]+params[a,7]*basis)[:,None]
    r=rot@Rotation.from_rotvec(params[:,:3]).as_matrix().transpose(0,2,1)
    c=centers+params[:,3:6]
    world=np.einsum("ni,nij->nj",corrected,r[a])+c[a]
    cam=np.einsum("nij,nj->ni",r[b],world-c[b])
    xy=cam[:,:2]/np.maximum(cam[:,2,None],1e-4)
    pixels=xy*np.stack((k[b,0,0],k[b,1,1]),axis=1)+k[b,:2,2]
    return pixels-observed


def fit(arm,folds,rot,c,k,prior_params):
    mask=np.ones((len(c),8),bool)
    mask[0,:7]=False
    if arm=="depth": mask[:,:6]=False
    if arm=="pose": mask[:,6:]=False
    zero=np.zeros_like(mask,dtype=float)
    train,test=folds
    if train is None or test is None:
        return zero,dict(accepted=False,reason="missing_fold")
    a,b,*_=train
    visited={0}
    for _ in range(len(c)-1):
        visited.update(b[np.isin(a,list(visited))].tolist())
    if len(visited)!=len(c): return zero,dict(accepted=False,reason="disconnected")
    def unpack(v):
        out=zero.copy()
        out[mask]=v
        return out*UNIT
    def cost(p,fold):
        return float(np.mean(robust(project(p,fold,rot,c,k))**2))
    def residual(v):
        p=unpack(v)
        image=robust(project(p,train,rot,c,k)).ravel()
        parts=[image/np.sqrt(image.size),v/np.sqrt(v.size)]
        if arm=="joint_prior":
            err=((p[1:,:6]-prior_params[1:,:6])/UNIT[:6]).ravel()
            parts.append(err/np.sqrt(err.size))
        return np.concatenate(parts)
    solution=least_squares(residual,np.zeros(mask.sum()),bounds=(-2.,2.),max_nfev=150)
    if not solution.success: raise RuntimeError(solution.message)
    proposed=unpack(solution.x)
    before,after=cost(zero,test),cost(proposed,test)
    accepted=after<=.99*before and before>0
    return proposed if accepted else zero,dict(accepted=accepted,train_before=cost(zero,train),train_after=cost(proposed,train),
            validation_before=before,validation_after=after,proposed=proposed.tolist(),evals=solution.nfev)


def infer():
    rows={}
    plan=read(rgb.tracks.PLAN)
    observations={(r["camera"]["scene"],r["camera"]["batch"]):r["tracks"] for r in read(rgb.tracks.TRACKS/"observations.json")}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(rgb.tracks.TRACKS/"redetected_tracks.zip") as archive:
        for scene in SCENES:
            slug=scene.split("/")[0]
            key=slug+"_base"
            source=read(fusion.model.SEAL)["rows"][key]
            e=np.array(source["fused"])
            rot,c=e[:,:3,:3],fusion.centers(e)
            base=fusion.depth.cross.temporal.load_points(fusion.model.DATA,source["hashes"]["joint"])
            z=np.einsum("nhwj,nj->nhw",base,rot[:,2,:])+e[:,2,3,None,None]
            scale=np.median(z)
            _,_,k,_,_=fusion.prior.inputs(key,read(fusion.prior.PARENT/"predictions.json")["rows"],plan)
            spec=plan["priors"][slug+"_wide"]
            path=fusion.depth.capture.original.INPUT/spec["file"]
            if sha(path)!=spec["sha256"]: raise ValueError("Supplied pose changed")
            with np.load(path) as f:
                rs,cs,_=fusion.align_sensor(e,f["extrinsics"].astype(float))
            q=rot[0].T@rs[0]
            rs=rs@q.T
            cs=(cs-cs[0])@q.T+c[0]
            target=np.zeros((len(c),8))
            target[:,:3]=Rotation.from_matrix(rs.transpose(0,2,1)@rot).as_rotvec()
            target[:,3:6]=(cs-c)/scale
            info=plan["inputs"][key]
            meta=observations[scene,info["batch"]]
            if meta["rgb_sha256"]!=info["rgb_sha256"]: raise ValueError("Wrong matched views")
            folds=[sample_fold(p,z,k,scale) for p in rgb.tracks.load_folds(archive,meta)]
            arms={}
            for arm in ARMS:
                p,stats=fit(arm,folds,rot,c/scale,k,target)
                basis=np.clip(scale/z-1,-1,1)
                factor=np.exp(p[:,6,None,None]+p[:,7,None,None]*basis)
                relative=Rotation.from_rotvec(p[:,:3]).as_matrix()
                points=np.stack([((b-center)@r.T*f[...,None]+center+scale*t).astype(np.float32)
                                 for b,center,r,f,t in zip(base,c,relative,factor,p[:,3:6])])
                h=fusion.prior.tensor_digest(points)
                if not stats["accepted"] and h!=source["hashes"]["joint"]: raise ValueError("Rejected update changed map")
                fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=points))
                arms[arm]=dict(hash=h,**stats)
            rows[key]=dict(matches=meta["matches_sha256"],fold_counts=[len(f[0]) if f else 0 for f in folds],arms=arms)
            print(key,{a:v["accepted"] for a,v in arms.items()},flush=True)
    return rows


def score(item):
    key,row=item
    arms=row["arms"]
    _,v=fusion.prior.evaluate_points(key,{x["hash"] for x in arms.values()},DATA)
    return key,{a:v[x["hash"]] for a,x in arms.items()}


def summarize(obs):
    from score_depth_scale_study import HISTORICAL,aggregate
    panel=read(previous.RESULT)["results"]["references"]
    rows=[]
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        metrics={}
        for ref,g in panel.items():
            arms={a:v["raw"] for a,v in g["scenes"][scene].items()}
            arms["mesh_reject"]=arms.pop("guarded")
            for arm in ARMS:
                arms["guarded" if arm=="joint_prior" else "reproject_"+arm]=obs[key][arm][ref]
            metrics[ref]=arms
        rows.append(dict(scene=scene,input_key=key,case="base",metrics=metrics))
    comps=tuple(a for a in arms if a not in (*HISTORICAL,"guarded"))
    return aggregate(rows,comps+("guarded",),comps)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action=p.parse_args().action
    deps=[sha(x) for x in (fusion.model.SEAL,rgb.tracks.PLAN,rgb.tracks.TRACKS/"observations.json")]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Input identity changed")
        deps=[sha(SEAL),sha(previous.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report=read(RESULT)
        if report["inputs"]!=deps or summarize(report["observed"])!=report["results"]: raise ValueError("Reprojection comparison differs")
        if action=="verify":
            hashes={k:{a:v["hash"] for a,v in r["arms"].items()} for k,r in seal["rows"].items()}
            fusion.prior.old.verify_map_archive(PUBLIC/"joint_reprojection_archive.json",hashes)
        print(report["results"]["checks"])
