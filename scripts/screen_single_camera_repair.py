"""Test actual camera restoration after selective RGB-only repair on development windows."""
import argparse
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np

from evaluation_stability import dump, sha
from localize_camera_prior_faults import CALIBRATION, ROOT, TRACKS, WINDOWS, load_inputs, score
from calibration_audit import resized_k
from repair_one_camera import diameter, fit_camera, perturb_camera, prepare_edges, validation_score, virtual_error
from score_saved_geometry import read


STUDY = "single_camera_repair"
PLAN = ROOT / "manifests" / (STUDY + ".json")
PUBLIC = ROOT / "results" / STUDY
FAULTS = ("rotation", "translation", "focal", "mixed")
METHODS = ("unchanged", "always", "alert_only", "validate_only", "alert_and_validate", "oracle_index")


def protocol():
    sources = [Path(__file__).resolve(), ROOT/"scripts/repair_one_camera.py", ROOT/"scripts/localize_camera_prior_faults.py",
               ROOT/"manifests/single_camera_fault_localization.json", TRACKS, WINDOWS, CALIBRATION]
    return dict(study=STUDY, sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                faults=list(FAULTS), methods=list(METHODS), seed=977, rotation_deg=3, translation_diameter_fraction=.06,
                focal_factor=1.12, windows=9, folds=2, cloud_usd=0,
                design="Use calibrated RGB camera convention. Inject one fresh signed synthetic perturbation into each of five view locations; mixed means all three components at that location. Each of nine known development windows has 20 corrupted and one clean case, evaluated in both directions of the existing connected-track split. Fit and select using only the training fold. Hold every other camera fixed. Use up to 64 matches from each incident edge and require at least two neighbors with six matches per edge.",
                decision="Select a unique highest median epipolar-error camera. Fit its rotation, center and shared focal multiplier by robust epipolar optimization with an identity-centered penalty. Alert if the training-fold score exceeds the maximum clean score in the other two scenes on the same fold. Validate if median incident edge error decreases strictly on the other track fold, using at least two neighbors. Primary method requires both. Oracle-index uses the known injected location solely as an offline comparator and never edits clean cases.",
                outcome="Primary endpoint is mean virtual reprojection error over all five cameras at a fixed 7x5 pixel grid at 1, 2 and 4 metres in each uncorrupted camera. It captures cross-component damage and incorrect edits. Report pixels for all fault types, clean and all scenes. No actual test points, clean pose labels or virtual projections enter fitting or acceptance. Fold reversals and cameras in a scene are repeated units.",
                gate="To justify a later dense pilot require >=20% lower mean error than unchanged over the four equally weighted corrupted profiles, a decrease within each profile, no increase within any scene over corruptions, and no clean window/fold above 1 pixel mean error. Compare simpler always, alert-only and validation-only baselines explicitly. Meeting this mechanism gate cannot release transfer funds.")


def dataset(folder):
    from replay_sensor_windows_from_zips import sequence_archive
    calibration = np.array(read(CALIBRATION)["calibration"])
    windows = read(WINDOWS)["rows"]
    result = []
    with zipfile.ZipFile(TRACKS) as matches:
        for label in sorted({r["scene"] for r in windows}):
            scene, seq = label.split("/")
            with sequence_archive(folder,scene,seq) as sequence:
                for rec in [r for r in windows if r["scene"]==label]:
                    tag, cams, pairs = load_inputs(sequence,matches,rec)
                    split = [[],[]]
                    with np.load(io.BytesIO(matches.read(tag+".npz")),allow_pickle=False) as source:
                        for n,(i,j,a,b) in enumerate(pairs):
                            fold = source[f"{n}_fold"]
                            if len(fold)!=len(a) or not np.isin(fold,[0,1]).all():
                                raise ValueError("Invalid match partition")
                            for side in (0,1):
                                keep=fold==side
                                if keep.sum()>=6:
                                    split[side].append((i,j,a[keep],b[keep]))
                    k = np.repeat(resized_k(525.,518,392,uniform=False)[None],5,axis=0).astype(float)
                    result.append((label,tag,calibration@cams,k,split))
    return result


def inject(e,k,view,fault,tag):
    obs, intr = e.copy(), k.copy()
    if fault=="clean":
        return obs,intr
    def stream(kind):
        key=hashlib.sha256(f"977:{tag}:{view}:{kind}".encode()).digest()
        return np.random.default_rng(int.from_bytes(key[:8],"little"))
    def direction(kind):
        v=stream(kind).normal(size=3)
        return v/np.linalg.norm(v)
    rot=np.deg2rad(3)*direction("rotation") if fault in ("rotation","mixed") else np.zeros(3)
    shift=.06*diameter(e)*direction("translation") if fault in ("translation","mixed") else np.zeros(3)
    sign=1 if stream("focal").integers(2) else -1
    focal=sign*np.log(1.12) if fault in ("focal","mixed") else 0.
    obs[view],intr[view]=perturb_camera(e[view],k[view],rot,shift,focal)
    return obs,intr


def screen_case(label,tag,clean_e,clean_k,folds,side,view,fault,threshold):
    obs,k=inject(clean_e,clean_k,view,fault,tag)
    ranked=score(obs,k,folds[side])
    suspect=ranked["winners"][0] if len(ranked["winners"])==1 else None
    alert=threshold is not None and ranked["best"] is not None and ranked["best"]>threshold
    fitted={}
    for index in {suspect,view}-{None}:
        fitted[index]=fit_camera(obs,k,folds[side],index)
    proposal=fitted.get(suspect)
    before=after=None
    if proposal is not None:
        edges=prepare_edges(obs,k,folds[1-side],suspect)
        before=validation_score((obs[suspect],k[suspect]),edges)
        after=validation_score(proposal,edges)
    valid=before is not None and after<before
    base=[virtual_error(clean_e[i],clean_k[i],obs[i],k[i]) for i in range(5)]
    base_error=float(np.mean(base))
    edit={m:None for m in METHODS}
    if proposal is not None:
        edit["always"]=suspect
        if alert: edit["alert_only"]=suspect
        if valid: edit["validate_only"]=suspect
        if alert and valid: edit["alert_and_validate"]=suspect
    if fitted.get(view) is not None:
        edit["oracle_index"]=view
    errors={m:base_error for m in METHODS}
    for method,i in edit.items():
        if i is not None:
            new=virtual_error(clean_e[i],clean_k[i],*fitted[i])
            errors[method]=(sum(base)-base[i]+new)/5
    return dict(scene=label,window=tag,train_fold=side,true_view=view,fault=fault,suspect=suspect,
                train_score=ranked["best"],threshold=threshold,alert=alert,validation_before=before,
                validation_after=after,validation_pass=valid,edited=edit,error_px=errors)


def summarize(rows):
    cases=[r for r in rows if r["fault"]!="clean"]
    clean=[r for r in rows if r["fault"]=="clean"]
    tags={r["scene"].replace("/","_")+"_"+str(r["frames"][0]) for r in read(WINDOWS)["rows"]}
    wanted={(w,s,f,v) for w in tags for s in (0,1) for f in FAULTS for v in range(5)}
    wanted.update((w,s,"clean",None) for w in tags for s in (0,1))
    if len(rows)!=len(wanted) or {(r["window"],r["train_fold"],r["fault"],r["true_view"]) for r in rows}!=wanted:
        raise ValueError("Incomplete single-camera repair grid")
    def means(group):
        return {m:float(np.mean([r["error_px"][m] for r in group])) for m in METHODS}
    def group_report(group):
        return dict(mean_error_px=means(group),edits={m:sum(r["edited"][m] is not None for r in group) for m in METHODS},
                    count=len(group))
    primary="alert_and_validate"
    by_fault={f:group_report([r for r in cases if r["fault"]==f]) for f in FAULTS}
    by_scene={s:group_report([r for r in cases if r["scene"]==s]) for s in sorted({r["scene"] for r in cases})}
    overall=means(cases)
    gain=1-overall[primary]/overall["unchanged"]
    worst_clean=max(r["error_px"][primary] for r in clean)
    flags=dict(overall=gain>=.20,profiles=all(d["mean_error_px"][primary]<d["mean_error_px"]["unchanged"] for d in by_fault.values()),
               scenes=all(d["mean_error_px"][primary]<=d["mean_error_px"]["unchanged"] for d in by_scene.values()),
               clean=worst_clean<=1.)
    return dict(all_corrupted=group_report(cases),clean=group_report(clean),by_fault=by_fault,by_scene=by_scene,
                overall_gain=gain,maximum_clean_mean_error_px=worst_clean,checks=flags,motivates_dense_pilot=all(flags.values()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","run","verify"))
    p.add_argument("--archives",type=Path)
    args=p.parse_args()
    if args.action=="freeze":
        if PLAN.exists(): raise FileExistsError(PLAN)
        dump(PLAN,protocol())
        return
    if read(PLAN)!=protocol(): raise ValueError("Repair protocol changed")
    path=PUBLIC/"summary.json"
    if args.action=="verify":
        data=read(path)
        if data["manifest_sha256"]!=sha(PLAN) or data["results"]!=summarize(data["rows"]):
            raise ValueError("Published repair summary changed")
        print("Verified 360 corrupted and 18 clean fold evaluations.")
        return
    if path.exists(): raise FileExistsError(path)
    windows=dataset(args.archives)
    clean={(tag,side):score(e,k,folds[side])["best"] for _,tag,e,k,folds in windows for side in (0,1)}
    rows=[]
    for label,tag,e,k,folds in windows:
        for side in (0,1):
            calibration=[clean[w,side] for s,w,*_ in windows if s!=label and clean[w,side] is not None]
            cutoff=max(calibration,default=None)
            profiles=[("clean",None)]+[(f,i) for f in FAULTS for i in range(5)]
            for fault,view in profiles:
                rows.append(screen_case(label,tag,e,k,folds,side,view,fault,cutoff))
        print("Completed",tag,flush=True)
    dump(path,dict(manifest_sha256=sha(PLAN),rows=rows,results=summarize(rows),additional_cloud_usd=0))


if __name__=="__main__":
    main()
