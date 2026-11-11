"""Compare the primary focal rule with a published pose-independent calibrator."""
import argparse
import hashlib
import importlib.metadata
import io
from pathlib import Path
import sys
import zipfile

import numpy as np

from evaluation_stability import dump, sha
from focal_metadata_control import focal_proposal
from focal_zoom_gradient import LEVELS, noisy_poses
from focal_zoom_observations import PLAN as PARENT_PLAN, PUBLIC as PARENT, ROOT, SCALE, UPSTREAM
from score_saved_geometry import read
from statistics import mean


NAME="visual_focal_baseline"
PLAN=ROOT/"manifests"/(NAME+".json")
PUBLIC=ROOT/"results"/NAME
WHEEL="5035f1053cb4ae70aec4b57b56552e17eef9e96dc9ac3940ad83f0875e12f33c"
METHODS=("iterative_continuous","iterative_nearest")
OPTIONS=dict(max_epipolar_error=1.5,max_iterations=5000,min_iterations=1000,success_prob=.999,seed=941,real_focal_check=False)


def protocol():
    return dict(study=NAME,parent_sha256=sha(PARENT_PLAN),code_sha256=sha(Path(__file__)),
        paper="https://github.com/kocurvik/robust_self_calibration",package="poselib==2.0.5",
        linux_cp311_wheel_sha256=WHEEL,options=OPTIONS,max_solver_iterations=50,weights=[.0005,1.,.0005,1.],
        design="Use all retained redetected correspondences. Recenter around each supplied principal point and rescale the two axes to square pixels while preserving area. For each pair estimate F with PoseLib then use its published iterative focal solver, initialized at the supplied focal priors. Convert the positive focal back to the supplied aspect ratio. Use the median across at least two successful neighbors for each camera. A missing estimate leaves that camera unchanged. Preserve supplied principal points and poses. Compare unrestricted continuous focal changes for all five cameras and a separate two-candidate decision on the unique median outlier. Only images, tracks and supplied intrinsics enter either estimate.",
        evaluation="Use the same primary mild-noise and severe-noise groups, hidden changes and uniform controls as the frozen parent. Report field-of-view and full gradients at the same image-conditioned outputs. Report log-focal error across every camera to expose collateral edits. These are additional established-method comparators. The prespecified parent pass criterion is unchanged. A claimed improvement over the published solver needs a direct paired comparison.",
        cloud_usd=0)


def centered_pixels(xy,k):
    fxy=k[[0,1],[0,1]]
    f=np.sqrt(np.prod(fxy))
    return (np.asarray(xy,dtype=float)-k[:2,2])*(f/fxy)


def fundamental_pairs(pairs,k):
    import poselib
    fitted=[]
    for i,j,a,b in pairs:
        p,q=centered_pixels(a,k[i]),centered_pixels(b,k[j])
        f,info=poselib.estimate_fundamental(p,q,OPTIONS,{})
        n=int(info["num_inliers"])
        valid=(n>=8 and f.shape==(3,3) and np.isfinite(f).all() and np.linalg.norm(f)>0)
        fitted.append(dict(i=i,j=j,matches=len(p),inliers=n,f=f.tolist() if valid else None))
    return fitted


def estimate(fitted,k):
    import poselib
    focal=np.sqrt(k[:,0,0]*k[:,1,1])
    prior=[dict(model="SIMPLE_PINHOLE",width=-1,height=-1,params=[f,0.,0.]) for f in focal]
    neighbor=[[] for _ in k]
    rows=[]
    for pair in fitted:
        i,j=pair["i"],pair["j"]
        status="missing_fundamental"
        values=None
        iterations=None
        if pair["f"] is not None:
            first,second,iterations=poselib.focals_from_fundamental_iterative(
                np.asarray(pair["f"]),prior[i],prior[j],50,np.array([.0005,1.,.0005,1.]))
            x,y=first.focal(),second.focal()
            if np.isfinite([x,y]).all() and min(x,y)>0:
                values=[x,y]
                neighbor[i].append(x)
                neighbor[j].append(y)
                status="finite_positive"
            else:
                status="invalid_focal"
        rows.append(dict(i=i,j=j,status=status,focals=values,iterations=iterations))
    continuous=k.copy()
    for view,values in enumerate(neighbor):
        if len(values)>=2:
            continuous[view,[0,1],[0,1]]*=np.median(values)/focal[view]
    snapped=k.copy()
    view,proposal=focal_proposal(k)
    edited=False
    if view is not None and len(neighbor[view])>=2:
        x=continuous[view,[0,1],[0,1]]
        dist=lambda a:np.abs(np.log(x/a)).mean()
        edited=bool(dist(proposal[[0,1],[0,1]])<dist(k[view,[0,1],[0,1]]))
        if edited:
            snapped[view]=proposal
    return continuous,snapped,dict(pairs=rows,valid_neighbors=[len(v) for v in neighbor],snapped_edit=edited)


def keys(scene,tag):
    return scene.replace("/","_")+"/"+tag+".npz"


def pack_tracks():
    from focal_zoom_observations import RAW
    path=PARENT/"redetected_tracks.zip"
    if path.exists():
        raise FileExistsError(path)
    partial=path.with_suffix(".partial")
    with zipfile.ZipFile(partial,"w") as z:
        for o in read(PARENT/"observations.json"):
            r=o["tracks"]
            f=RAW/r["scene"].replace("/","_")/r["batch"]/"tracks.npz"
            if sha(f)!=r["matches_sha256"]:
                raise ValueError("Unverified redetected matches")
            z.write(f,keys(r["scene"],r["batch"]))
    partial.replace(path)


def pairs(archive,obs):
    t=obs["tracks"]
    b=archive.read(keys(t["scene"],t["batch"]))
    if hashlib.sha256(b).hexdigest()!=t["matches_sha256"]:
        raise ValueError("Redetected match file changed")
    with np.load(io.BytesIO(b),allow_pickle=False) as src:
        return [(i,j,src[f"{n}_xy_i"],src[f"{n}_xy_j"]) for n,(i,j) in enumerate(t["pair_order"])]


def summaries(rows):
    if len(rows)!=465:
        raise ValueError("Incomplete calibration comparison")
    def reduce(group):
        return {m:{k:mean(r["errors"][m][k] for r in group) for k in group[0]["errors"][m]} for m in group[0]["errors"]}
    visible=[r for r in rows if r["case"] in ("metadata_fault","valid_zoom")]
    focus=[r for r in visible if r["noise"]=="mild"]
    return dict(primary=reduce(focus),by_scene={s:reduce([r for r in focus if r["scene"]==s]) for s in sorted({r["scene"] for r in rows})},
                by_noise={level:reduce([r for r in visible if r["noise"]==level]) for level in ("clean","mild","strong")},
                controls={kind:reduce([r for r in rows if r["case"]==kind]) for kind in ("uniform_clean","hidden_zoom")})


def run(archives):
    import torch
    from audit_masked_camera_gradients import gradient, inputs
    if importlib.metadata.version("poselib")!="2.0.5":
        raise ValueError("Unexpected PoseLib version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    record=read(PARENT/"summary.json")
    obs={(o["camera"]["scene"],o["camera"]["batch"]):o for o in read(PARENT/"observations.json")}
    clean={s:(e,k) for s,_,e,k,_ in inputs(archives)}
    levels={(n,seed):(a,t) for n,seed,a,t in LEVELS}
    fits,estimates,rows,traces={},{},[],[]
    with np.load(PARENT/"gradient_arrays.zip",allow_pickle=False) as data:
        refs=data["variants"]
    with zipfile.ZipFile(PARENT/"redetected_tracks.zip") as zipped:
        for r,g in zip(record["rows"],refs):
            s,tag=r["scene"],r["batch"]
            view=r["view"]
            e,registered=clean[s]
            k=registered.copy()
            k[:,[0,1],[0,1]]*=SCALE
            truth=k.copy()
            if r["case"] in ("valid_zoom","hidden_zoom"):
                truth[view,[0,1],[0,1]]*=SCALE**r["direction"]
            if r["case"] in ("valid_zoom","metadata_fault"):
                k[view,[0,1],[0,1]]*=SCALE**r["direction"]
            key=keys(s,tag)
            if key not in fits:
                fits[key]=fundamental_pairs(pairs(zipped,obs[s,tag]),k)
            config=(s,r["case"],view,r["direction"])
            if config not in estimates:
                a,b,detail=estimate(fits[key],k)
                estimates[config]=(a,b,detail)
            a,b,detail=estimates[config]
            pose=noisy_poses(e,s,r["seed"],*levels[r["noise"],r["seed"]])
            enc=np.asarray(obs[s,tag]["camera"]["encoding"],np.float32)
            all_errors={}
            measure=lambda target:float(np.abs(np.log(target[:,[0,1],[0,1]]/truth[:,[0,1],[0,1]])).mean())
            for m in ("retain","consensus","model_nearest","first_fold","both_folds"):
                metric=r["errors"][m]
                if r["edits"][m]:
                    adjusted=k.copy()
                    idx,candidate=focal_proposal(k)
                    adjusted[idx]=candidate
                else:
                    adjusted=k
                all_errors[m]=dict(focal_gradient_relative=metric["focal_gradient_relative"],
                        whole_gradient_relative=metric["whole_gradient_relative"],all_camera_log_error=measure(adjusted))
            vs=[]
            for method,estimate_k in zip(METHODS,(a,b)):
                result=gradient(enc,pose,estimate_k,None)
                f64=result.astype(np.float64)
                conditional,ideal=g[2:].astype(np.float64)
                all_errors[method]=dict(focal_gradient_relative=float(np.linalg.norm(f64[:,7:]-conditional[:,7:]))/max(1e-6,float(np.linalg.norm(conditional[:,7:]))),
                    whole_gradient_relative=float(np.linalg.norm(f64-ideal))/max(1e-6,float(np.linalg.norm(ideal))),
                    all_camera_log_error=measure(estimate_k))
                vs.append(result)
            rows.append(dict(**{k:r[k] for k in ("scene","case","view","direction","noise","seed","batch")},errors=all_errors,
                             solver_snapped_edit=detail["snapped_edit"]))
            traces.append(vs)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    fitted=[dict(scene=s,case=c,view=v,direction=d,
                 continuous_xy=a[:,[0,1],[0,1]].tolist(),snapped_xy=b[:,[0,1],[0,1]].tolist(),**meta)
             for (s,c,v,d),(a,b,meta) in estimates.items()]
    dump(PUBLIC/"solver_outputs.json",dict(fundamentals=fits,calibrations=fitted))
    with (PUBLIC/"gradient_arrays.zip").open("xb") as f:
        np.savez_compressed(f,variants=np.stack(traces))
    dump(PUBLIC/"summary.json",dict(manifest_sha256=sha(PLAN),parent_summary_sha256=sha(PARENT/"summary.json"),
         solver_sha256=sha(PUBLIC/"solver_outputs.json"),track_sha256=sha(PARENT/"redetected_tracks.zip"),
         gradients_sha256=sha(PUBLIC/"gradient_arrays.zip"),rows=rows,results=summaries(rows),additional_cloud_usd=0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","pack","run"))
    p.add_argument("--archives",type=Path)
    args=p.parse_args()
    if args.action=="freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN,protocol())
        return
    if read(PLAN)!=protocol():
        raise ValueError("Solver comparison protocol changed")
    if args.action=="pack":
        pack_tracks()
    else:
        sys.path.insert(0,str(UPSTREAM))
        if any((PUBLIC/name).exists() for name in ("solver_outputs.json","gradient_arrays.zip","summary.json")):
            raise FileExistsError("Existing solver result")
        run(args.archives)


if __name__=="__main__":
    main()
