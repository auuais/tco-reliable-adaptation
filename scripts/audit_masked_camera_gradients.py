"""Compare external camera-head gradients under clean and locally corrupted priors."""
import argparse
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

from calibration_audit import resized_k
from evaluation_stability import dump, sha
from fit_sensor_extrinsic import homogeneous
from localize_camera_prior_faults import score
from masked_camera_supervision import without_camera
from prior_corruptions import tensor_hash
from score_saved_geometry import read
from screen_single_camera_repair import FAULTS, inject


ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/"third_party/TCO"
PUBLIC=ROOT/"results/masked_camera_gradients"
PLAN=ROOT/"manifests/masked_camera_gradients.json"
SEEDS=(211,367,859)
METHODS=("unchanged","omit_on_alert","global_dropout_on_alert","random_omission_on_alert","always_omit_highest","oracle_omission")
CHANNELS={"whole":slice(0,9),"translation":slice(0,3),"quaternion":slice(3,7),"field_of_view":slice(7,9)}


def protocol():
    names=["scripts/audit_masked_camera_gradients.py","scripts/pack_camera_gradient_inputs.py","scripts/masked_camera_supervision.py",
           "scripts/localize_camera_prior_faults.py","scripts/screen_single_camera_repair.py","scripts/repair_one_camera.py",
           "results/masked_camera_gradients/inputs.json","results/masked_camera_gradients/rgb_matches.zip",
           "results/single_camera_fault_localization/summary.json","manifests/sensor_frame_v14.json","configs/7scenes_baseline.yaml",
           "third_party/TCO/constraints.py","third_party/TCO/tco_loss.py","third_party/TCO/utils/geometry.py"]
    return dict(study="masked_camera_gradients",sources={n:sha(ROOT/n) for n in names},seeds=list(SEEDS),frames=[0,200,400,600,800],
                methods=list(METHODS),channels=list(CHANNELS),faults=list(FAULTS),cloud_usd=0,
                design="Three captured VGGT initial camera encodings. Use the upstream decoder and TCO camera supervision on CPU in FP32, pose weight 1, translation weight 2, intrinsics weight .01, angle rotation, normed L1 translations. Measure derivatives with respect to all 45 encoding entries. Use clean calibrated supplied cameras as the offline gradient reference. Exclude visual/depth losses, backbone Jacobians and optimizer state. No inference-time access to clean gradients.",
                faults_design="Each of five cameras separately receives 3 degrees of rotation, 6% of window-diameter center displacement, a 1.12 focal factor or its inverse, or their joint perturbation. Three fresh deterministic seeds per location/profile. Include three clean inputs. Use all matched observations for ranking. For each target scene use the largest clean calibrated detector score on six offset-50/100/150 windows from the other two scenes. Unchanged input is used on a tie or when no alert occurs.",
                analysis="Primary error is the Euclidean norm of a method gradient minus the clean unmasked gradient, divided by the clean whole-encoding gradient norm. Report contributions of translations, quaternions and FoV using that same denominator. Random omission is the exact average of the five possible exclusion errors on the same alert, not the error of their averaged gradient. Known-index omission is privileged. Treat repeated faults as synthetic interventions within three development scenes.",
                gate="Require at least 20% lower mean corrupted error for alert-based omission vs unchanged, lower than same-alert global dropout and random omission, no scene worse than unchanged, no fault profile more than 5% worse, and maximum clean distortion <=0.05. This would justify a local dense pilot only. A failed primary comparison cannot release later allocations.")


def gradient(code,e,k,omission):
    import tco_loss
    from utils.geometry import extrinsics_to_se3,invert_se3,pose_encoding_to_extri_intri
    z=torch.from_numpy(np.asarray(code,np.float32)).clone().requires_grad_()
    extr,intr=pose_encoding_to_extri_intri(z,(392,518),build_intrinsics=True)
    predictions=dict(camera_poses=invert_se3(extrinsics_to_se3(extr)),intrinsics=intr,pose_type="abs")
    loss,_=without_camera(tco_loss.compute_tco_losses,omission)(
        predictions,torch.empty(1,5,3,1,1),gt_extrinsics=torch.from_numpy(e)[None].float(),
        gt_intrinsics=torch.from_numpy(k)[None].float(),lambda_pose=1.,pose_translation_weight=2.,
        lambda_intrinsics=.01,pose_rot_loss_type="angle",pose_trans_loss_type="normed_l1",canonicalize_to_first_view=True)
    out=torch.autograd.grad(loss,z)[0].detach().numpy()[0]
    if not np.isfinite(out).all():raise ValueError("Nonfinite camera derivative")
    return out


def inputs(archives):
    from replay_sensor_windows_from_zips import sequence_archive
    from utils.geometry import closed_form_inverse_se3
    source=read(PUBLIC/"inputs.json")
    h=np.asarray(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    if sha(PUBLIC/"rgb_matches.zip")!=source["matches_sha256"]:raise ValueError("Unexpected feature tracks")
    with zipfile.ZipFile(PUBLIC/"rgb_matches.zip") as tracks:
        for rec in source["scenes"]:
            label=rec["scene"]
            scene,seq=label.split("/")
            cameras=[]
            with sequence_archive(archives,scene,seq) as archive:
                for frame in rec["frames"]:
                    name=f"frame-{frame:06d}.pose.txt"
                    buf=archive.read(seq+"/"+name)
                    if hashlib.sha256(buf).hexdigest()!=rec["frame_pose_hashes"][name]:
                        raise ValueError("Unexpected pose file")
                    cameras.append(closed_form_inverse_se3(np.loadtxt(io.BytesIO(buf))[None])[0,:3])
            raw=np.stack(cameras)
            calibrated=h@homogeneous(raw)
            if (tensor_hash(raw.astype(np.float32))!=rec["supplied_extrinsics_fp32_sha256"]
                    or tensor_hash(calibrated.astype(np.float32))!=rec["calibrated_extrinsics_fp32_sha256"]):
                raise ValueError("FP32 pose-loss inputs changed")
            pairs=[]
            for p in rec["pairs"]:
                blob=tracks.read(p["file"])
                if hashlib.sha256(blob).hexdigest()!=p["sha256"]:raise ValueError("Unexpected matches")
                with np.load(io.BytesIO(blob),allow_pickle=False) as f:
                    pairs.append((p["i"],p["j"],f["xy_i"],f["xy_j"]))
            yield label,np.asarray(rec["initial_encoding"],np.float32),calibrated,np.repeat(resized_k(525.,518,392,uniform=False)[None],5,axis=0).astype(float),pairs


def errors(row,variants,reference):
    suspect=row["suspect"]
    chosen=0 if suspect is None else 1+suspect
    alert=row["alert"]
    indices={"unchanged":[0],"omit_on_alert":[chosen if alert else 0],
             "global_dropout_on_alert":[6 if alert else 0],"random_omission_on_alert":list(range(1,6)) if alert else [0],
             "always_omit_highest":[chosen],"oracle_omission":[0 if row["view"] is None else 1+row["view"]]}
    # Index 6 is the gradient from zero external supervision; visual gradients are outside this audit.
    observed=np.concatenate([variants,np.zeros_like(variants[:1])]).astype(np.float64)
    ref=reference.astype(np.float64)
    scale=np.linalg.norm(ref)
    if scale<1e-8:raise ValueError("Clean whole-code gradient is negligible")
    distances=observed-ref
    return {m:{channel:float(np.mean(np.linalg.norm(distances[ii,:,sl],axis=(1,2)))/scale)
               for channel,sl in CHANNELS.items()} for m,ii in indices.items()}


def aggregate(rows):
    faulty=[r for r in rows if r["fault"]!="clean"]
    clean=[r for r in rows if r["fault"]=="clean"]
    scenes={r["scene"] for r in read(PUBLIC/"inputs.json")["scenes"]}
    expected={(s,f,v,t) for s in scenes for f in FAULTS for v in range(5) for t in SEEDS}|{(s,"clean",None,None) for s in scenes}
    if len(rows)!=len(expected) or {(r["scene"],r["fault"],r["view"],r["seed"]) for r in rows}!=expected:
        raise ValueError("Unexpected scene/fault grid")
    def means(cases):
        return {m:{c:float(np.mean([r["errors"][m][c] for r in cases])) for c in CHANNELS} for m in METHODS}
    overall=means(faulty)
    profiles={f:means([r for r in faulty if r["fault"]==f]) for f in FAULTS}
    scenes_table={s:means([r for r in faulty if r["scene"]==s]) for s in sorted(scenes)}
    target="omit_on_alert"
    value=overall[target]["whole"]
    gain=1-value/overall["unchanged"]["whole"]
    worst_clean=max(r["errors"][target]["whole"] for r in clean)
    checks=dict(gain=gain>=.20,baselines=all(value<overall[m]["whole"] for m in ("global_dropout_on_alert","random_omission_on_alert")),
                scenes=all(t[target]["whole"]<=t["unchanged"]["whole"] for t in scenes_table.values()),
                profiles=all(t[target]["whole"]<=1.05*t["unchanged"]["whole"] for t in profiles.values()),
                clean=worst_clean<=.05)
    return dict(overall=overall,by_fault=profiles,by_scene=scenes_table,clean=means(clean),
                clean_alerts=sum(r["alert"] for r in clean),corrupted_alerts=sum(r["alert"] for r in faulty),
                worst_clean=worst_clean,gain=gain,checks=checks,motivates_dense_pilot=all(checks.values()))


def run(archives):
    calibrators=read(ROOT/"results/single_camera_fault_localization/summary.json")["rows"]
    rows,variants,references=[],[],{}
    for label,code,e,k,pairs in inputs(archives):
        cutoff=max(r["best"] for r in calibrators if r["fault"]=="clean" and r["scene"]!=label and r["convention"]=="fixed_rgb")
        references[label]=gradient(code,e,k,None)
        cases=[("clean",None,None)]+[(f,v,t) for f in FAULTS for v in range(5) for t in SEEDS]
        for fault,view,seed in cases:
            obs,intr=inject(e,k,view,fault,label+"_offset0_"+str(seed))
            ranked=score(obs,intr,pairs)
            suspect=ranked["winners"][0] if len(ranked["winners"])==1 else None
            alert=suspect is not None and ranked["best"]>cutoff
            row=dict(scene=label,fault=fault,view=view,seed=seed,suspect=suspect,alert=alert,cutoff=cutoff,
                     top_score=ranked["best"],eligible_degree=ranked["degree"])
            g=np.stack([gradient(code,obs,intr,i) for i in (None,0,1,2,3,4)])
            row["errors"]=errors(row,g,references[label])
            rows.append(row)
            variants.append(g)
    trace=PUBLIC/"gradient_traces.zip"
    labels=sorted(references)
    with trace.open("xb") as stream:
        np.savez_compressed(stream,variants=np.stack(variants),clean=np.stack([references[s] for s in labels]),scenes=labels)
    dump(PUBLIC/"summary.json",dict(manifest_sha256=sha(PLAN),trace_sha256=sha(trace),rows=rows,results=aggregate(rows),cloud_usd=0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","run","verify"))
    p.add_argument("--archives",type=Path)
    args=p.parse_args()
    if args.action=="freeze":
        if PLAN.exists():raise FileExistsError(PLAN)
        dump(PLAN,protocol())
        return
    if read(PLAN)!=protocol():raise ValueError("Gradient protocol changed")
    trace=PUBLIC/"gradient_traces.zip"
    if args.action=="run":
        if trace.exists() or (PUBLIC/"summary.json").exists():raise FileExistsError("Existing output")
        torch.set_num_threads(1)
        sys.path.insert(0,str(UPSTREAM))
        run(args.archives)
    else:
        s=read(PUBLIC/"summary.json")
        if s["manifest_sha256"]!=sha(PLAN) or s["trace_sha256"]!=sha(trace):raise ValueError("Trace identity changed")
        with np.load(trace,allow_pickle=False) as data:
            if data["variants"].shape!=(183,6,5,9):raise ValueError("Incomplete gradient array")
            ref=dict(zip(data["scenes"].tolist(),data["clean"]))
            for row,g in zip(s["rows"],data["variants"]):
                if row["errors"]!=errors(row,g,ref[row["scene"]]):raise ValueError("Gradient errors changed")
        if s["results"]!=aggregate(s["rows"]):raise ValueError("Gradient decision changed")
        print("Verified 183 camera-prior cases from all 1,098 recorded gradient arrays.")


if __name__=="__main__":main()
