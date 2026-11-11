"""Evaluate focal decisions on redetected features and original TCO gradients."""
import argparse
import hashlib
from pathlib import Path
from statistics import mean
import sys

import numpy as np

from evaluation_stability import dump, sha
from focal_metadata_control import decisions, focal_proposal, fold_gain
from focal_zoom_observations import PLAN, PUBLIC, RAW, ROOT, SCALE, UPSTREAM, check_plan
from repair_one_camera import diameter, perturb_camera
from score_saved_geometry import read


METHODS=("retain","consensus","first_fold","both_folds","model_nearest")
CASES=("metadata_fault","valid_zoom","hidden_zoom","uniform_clean")
LEVELS=(("clean",0,0.,0.),("mild",137,1.,.02),("mild",941,1.,.02),
        ("strong",137,3.,.06),("strong",941,3.,.06))


def noisy_poses(e,scene,seed,angle,shift):
    out=e.copy()
    if angle or shift:
        for i in range(1,len(e)):
            token=hashlib.sha256(f"focal-zoom:{scene}:{seed}:{i}".encode()).digest()
            rng=np.random.default_rng(int.from_bytes(token[:8],"little"))
            dirs=rng.normal(size=(2,3))
            dirs/=np.linalg.norm(dirs,axis=1)[:,None]
            out[i],_=perturb_camera(e[i],np.eye(3),np.deg2rad(angle)*dirs[0],diameter(e)*shift*dirs[1],0.)
    return out


def get_batch(scene,tag):
    folder=RAW/scene.replace("/","_")/tag
    cam,track=(read(folder/(kind+".json")) for kind in ("camera","tracks"))
    for key in ("scene","batch","rgb_sha256","manifest_sha256"):
        if cam[key]!=track[key]:
            raise ValueError("Image or protocol differs across the two branches")
    if (cam["scene"]!=scene or cam["batch"]!=tag or cam["manifest_sha256"]!=sha(PLAN)
            or sha(folder/"tracks.npz")!=track["matches_sha256"]):
        raise ValueError("Invalid observation provenance")
    folds=[[],[]]
    with np.load(folder/"tracks.npz",allow_pickle=False) as data:
        for n,(i,j) in enumerate(track["pair_order"]):
            a,b,idx=(data[f"{n}_{k}"] for k in ("xy_i","xy_j","fold"))
            if len(a)!=len(b) or len(a)!=len(idx) or not np.isin(idx,[0,1]).all():
                raise ValueError("Invalid track partition")
            for side in (0,1):
                mask=idx==side
                if mask.sum()>=6:
                    folds[side].append((i,j,a[mask],b[mask]))
    return np.asarray(cam["encoding"],np.float32),folds,cam,track


def predict_edits(e,k,focal,folds):
    view,candidate=focal_proposal(k)
    edits=dict.fromkeys(METHODS,False)
    if view is None:
        return edits,[None,None],k.copy()
    gains=[fold_gain(e,k,candidate,view,rows) for rows in folds]
    edits.update(decisions(gains))
    observed=k[view,[0,1],[0,1]]
    proposed=candidate[[0,1],[0,1]]
    err=lambda a:np.mean(np.abs(np.log(focal[view]/a)))
    edits["model_nearest"]=bool(err(proposed)<err(observed))
    out=k.copy()
    out[view]=candidate
    return edits,gains,out


def errors(gradients,k_error,edits):
    # Original, consensus, correct K under supplied poses, and fully correct K/E.
    original,consensus,conditional,ideal=gradients.astype(np.float64)
    result={}
    for m,edit in edits.items():
        chosen=consensus if edit else original
        absolute=float(np.linalg.norm(chosen[:,7:]-conditional[:,7:]))
        result[m]=dict(focal_gradient_abs=absolute,
                       focal_gradient_relative=absolute/max(1e-6,float(np.linalg.norm(conditional[:,7:]))),
                       whole_gradient_relative=float(np.linalg.norm(chosen-ideal))/max(1e-6,float(np.linalg.norm(ideal))),
                       log_focal_error=k_error[int(edit)])
    return result


def observe(scene,native_e,base_k,batch,case,view,direction,level):
    import torch
    from audit_masked_camera_gradients import gradient
    from utils.geometry import pose_encoding_to_extri_intri
    code,folds,cam,_=batch
    noise,seed,angle,shift=level
    e=noisy_poses(native_e,scene,seed,angle,shift)
    observed,truth=base_k.copy(),base_k.copy()
    if case in ("metadata_fault","valid_zoom"):
        observed[view,[0,1],[0,1]] *= SCALE**direction
    if case in ("valid_zoom","hidden_zoom"):
        truth[view,[0,1],[0,1]] *= SCALE**direction
    with torch.no_grad():
        _,pred=pose_encoding_to_extri_intri(torch.from_numpy(code),(392,518),build_intrinsics=True)
    focal=pred.numpy()[0][:,[0,1],[0,1]]
    edits,gains,candidate=predict_edits(e,observed,focal,folds)
    # Only evaluate distinct gradient inputs. The four saved entries keep a fixed order.
    cached={}
    variants=[]
    for poses,intr in ((e,observed),(e,candidate),(e,truth),(native_e,truth)):
        key=(poses.tobytes(),intr.tobytes())
        if key not in cached:
            cached[key]=gradient(code,poses,intr,None)
        variants.append(cached[key])
    gradients=np.stack(variants)
    target=0 if view is None else view
    def log_error(k):
        return float(np.mean(np.abs(np.log(k[target,[0,1],[0,1]]/truth[target,[0,1],[0,1]]))))
    discrepancy=[log_error(k) for k in (observed,candidate)]
    row=dict(scene=scene,case=case,view=view,direction=direction,noise=noise,seed=seed,rgb_sha256=cam["rgb_sha256"],
             batch=cam["batch"],fold_gains=gains,edits=edits,log_focal_discrepancy=discrepancy,
             target_focal=truth[target,[0,1],[0,1]].tolist(),supplied_focal=observed[target,[0,1],[0,1]].tolist(),
             consensus_focal=candidate[target,[0,1],[0,1]].tolist(),model_focal=focal[target].tolist())
    row["errors"]=errors(gradients,discrepancy,edits)
    return row,gradients


def group(rows):
    means={m:{k:mean(r["errors"][m][k] for r in rows) for k in rows[0]["errors"][m]} for m in METHODS}
    valid=[r for r in rows if r["case"]=="valid_zoom"]
    bad=[r for r in rows if r["case"]=="metadata_fault"]
    return dict(n=len(rows),means=means,
                wrong_valid_edits={m:sum(r["edits"][m] for r in valid) for m in METHODS},
                correct_fault_edits={m:sum(r["edits"][m] for r in bad) for m in METHODS},
                unchanged_zero_focal_gradient_error=sum(r["errors"]["retain"]["focal_gradient_abs"]==0 for r in rows),
                two_fold_supported=sum(all(g is not None for g in r["fold_gains"]) for r in rows))


def summarize(rows):
    scenes=sorted(r["scene"] for r in read(ROOT/"results/masked_camera_gradients/inputs.json")["scenes"])
    configs=[(kind,v,d) for kind in CASES[:3] for v in range(5) for d in (-1,1)]+[("uniform_clean",None,0)]
    expected={(s,case,v,d,level,seed) for s in scenes for case,v,d in configs for level,seed,_,_ in LEVELS}
    key=lambda r:tuple(r[k] for k in ("scene","case","view","direction","noise","seed"))
    if len(rows)!=len(expected) or {key(r) for r in rows}!=expected:
        raise ValueError("Incomplete fresh-image gradient grid")
    visible=[r for r in rows if r["case"] in CASES[:2]]
    focus=[r for r in visible if r["noise"]=="mild"]
    primary=group(focus)
    scene_means={s:group([r for r in focus if r["scene"]==s]) for s in scenes}
    metric="focal_gradient_relative"
    def gain(g):
        values=g["means"]
        base=min(values[m][metric] for m in ("retain","consensus"))
        if base==0:
            raise ValueError("The best unconditional comparator has zero mean distortion")
        return 1-values["both_folds"][metric]/base
    valid=[r for r in focus if r["case"]=="valid_zoom"]
    checks=dict(gain=gain(primary)>=.2,scenes=all(gain(v)>=0 for v in scene_means.values()),
                pretrained=primary["means"]["both_folds"][metric]<=primary["means"]["model_nearest"][metric],
                valid_zooms=primary["wrong_valid_edits"]["both_folds"]/len(valid)<=.05)
    return dict(primary=primary,primary_gain=gain(primary),by_scene=scene_means,
                by_pose_noise={level:group([r for r in visible if r["noise"]==level]) for level in ("clean","mild","strong")},
                controls={kind:group([r for r in rows if r["case"]==kind]) for kind in CASES[2:]},
                checks=checks,motivates_dense_development_pilot=all(checks.values()))


def execute(archives):
    import torch
    from audit_masked_camera_gradients import inputs
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows,variants,observations=[],[],[]
    for scene,_,e,k,_ in inputs(archives):
        k[:,[0,1],[0,1]] *= SCALE
        base=get_batch(scene,"base")
        batches={"base":base}
        for v in range(5):
            for d,zoom in ((-1,"wide"),(1,"tight")):
                tag=f"v{v}_{zoom}"
                batches[tag]=get_batch(scene,tag)
                for case in CASES[:3]:
                    target=base if case=="metadata_fault" else batches[tag]
                    for level in LEVELS:
                        row,g=observe(scene,e,k,target,case,v,d,level)
                        rows.append(row)
                        variants.append(g)
        for level in LEVELS:
            row,g=observe(scene,e,k,base,"uniform_clean",None,0,level)
            rows.append(row)
            variants.append(g)
        for _,_,cam,track in batches.values():
            observations.append(dict(camera=cam,tracks=track))
    result=summarize(rows)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    trace=PUBLIC/"gradient_arrays.zip"
    with trace.open("xb") as stream:
        np.savez_compressed(stream,variants=np.stack(variants))
    dump(PUBLIC/"observations.json",observations)
    dump(PUBLIC/"summary.json",dict(manifest_sha256=sha(PLAN),trace_sha256=sha(trace),
         observations_sha256=sha(PUBLIC/"observations.json"),rows=rows,results=result))


def verify():
    from verify_masked_camera_gradients import compare
    record=read(PUBLIC/"summary.json")
    trace=PUBLIC/"gradient_arrays.zip"
    observations=read(PUBLIC/"observations.json")
    if (record["manifest_sha256"]!=sha(PLAN) or record["trace_sha256"]!=sha(trace)
            or record["observations_sha256"]!=sha(PUBLIC/"observations.json") or len(observations)!=33):
        raise ValueError("Frozen evidence differs")
    obs={(o["camera"]["scene"],o["camera"]["batch"]):o["camera"]["rgb_sha256"] for o in observations}
    with np.load(trace,allow_pickle=False) as f:
        gradients=f["variants"]
    if len(obs)!=33 or gradients.shape!=(465,4,5,9) or not np.isfinite(gradients).all():
        raise ValueError("Invalid gradient traces or camera batches")
    for r,g in zip(record["rows"],gradients):
        if obs[r["scene"],r["batch"]]!=r["rgb_sha256"]:
            raise ValueError("Camera output did not use this image batch")
        a,b,c,p=(np.array(r[n]) for n in ("supplied_focal","consensus_focal","target_focal","model_focal"))
        distance=lambda x,y:float(np.mean(np.abs(np.log(x/y))))
        k_error=[distance(a,c),distance(b,c)]
        edits=decisions(r["fold_gains"]) if r["case"] in CASES[:2] else dict.fromkeys(METHODS,False)
        edits["model_nearest"]=distance(p,b)<distance(p,a)
        if edits!=r["edits"] or not np.allclose(k_error,r["log_focal_discrepancy"],atol=1e-14,rtol=1e-14):
            raise ValueError("Focal metadata decision or label changed")
        compare(r["errors"],errors(g,k_error,edits))
    compare(record["results"],summarize(record["rows"]))
    print("Verified all 33 camera batches, 465 conditions and 1,860 original loss gradient records.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("run","verify"))
    p.add_argument("--archives",type=Path)
    args=p.parse_args()
    check_plan()
    sys.path.insert(0,str(UPSTREAM))
    if args.action=="run":
        if any((PUBLIC/n).exists() for n in ("summary.json","gradient_arrays.zip","observations.json")):
            raise FileExistsError("Existing result")
        execute(args.archives)
    else:
        verify()


if __name__=="__main__":
    main()
