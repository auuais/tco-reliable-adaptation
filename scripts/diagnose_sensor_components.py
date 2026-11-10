"""Separate dense axial-depth quality from camera agreement with published RGB SfM."""
import argparse
import hashlib
from pathlib import Path
from statistics import mean,median
from urllib.request import urlopen

import numpy as np

from audit_sensor_frame import effective_camera, SCENES
from calibration_audit import resized_k
from camera_depth_swap import relative_rig
from evaluation_stability import dump,sha
from pointmap_camera_diagnostic import recover
from score_saved_geometry import read

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"manifests/sensor_component_metrics.json"
STUDIES={"sensor_extrinsic_v15":0,"sensor_tco_v17":50}
KEYS=("depth_abs_rel","depth_delta_1_25","pair_rotation_median_deg","pair_translation_direction_median_deg","mean_abs_log_focal")


def plan():
    return dict(studies=STUDIES,metrics=list(KEYS),code_sha256=sha(Path(__file__)),
                v15_seal_sha256=sha(ROOT/"results/sensor_extrinsic_v15/seal.json"),
                v17_job_plan_sha256=sha(ROOT/"manifests/sensor_tco_v17.json"),
                public_pose_release=read(ROOT/"results/sensor_frame_v14/alternative_rgb_sfm.json")["revision"],
                method="Offline diagnostic, outside all advancement gates. Recover each prediction's pinhole cameras and axial depths from the sealed map with the v10 camera fit. Recover reference axial depth with the exact inverse of the loader's camera-to-world transform. A single median ratio of GT to estimated depth over all valid pixels fixes scale for the complete five-frame window; report mean absolute relative depth error and delta<1.25. Compare predicted cameras to pinned public RGB SfM camera estimates for the same frames. For all ten pairs use relative rotation angle and the angle between translation directions expressed in the target camera. Focal metric is the mean absolute log ratio over two axes and all cameras. Per-scene means average seeds, global means average scenes. These alternative labels never enter the method. No claim of metric-scale recovery or independently surveyed camera poses.")


def public_rows(study,subset):
    baseline = read(ROOT/"results/sensor_frame_v14/alternative_rgb_sfm.json")
    chosen = {}
    for scene in SCENES:
        s,seq=scene.split("/")
        digest=next(r["source_file_sha256"] for r in baseline["rows"] if r["scene"]==scene)
        url=f"https://raw.githubusercontent.com/tsattler/visloc_pseudo_gt_limitations/{baseline['revision']}/pgt/sfm/7scenes/{s}_test.txt"
        with urlopen(url,timeout=30) as f:blob=f.read()
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError("Published RGB pose table changed")
        keys=[f"{seq}/frame-{i+STUDIES[study]:06d}.color.png" for i in (0,200,400,600,800)]
        values={}
        for line in blob.decode().splitlines():
            cols=line.split()
            if cols and cols[0] in keys:
                if len(cols)!=9 or cols[0] in values:raise ValueError("Invalid published row")
                values[cols[0]]=[float(x) for x in cols[1:]]
        if set(values)!=set(keys):raise ValueError("One of the five published pose estimates is missing")
        chosen[scene]=dict(url=url,file_sha256=digest,keys=keys,poses=[values[k] for k in keys])
    if subset.exists():raise FileExistsError(subset)
    dump(subset,chosen)


def camera_metrics(model,ref):
    from scipy.spatial.transform import Rotation
    angles,directions=[],[]
    for i,j in zip(*np.triu_indices(len(model.center),1)):
        model_r=model.rotation[j]@model.rotation[i].T
        ref_r=ref.rotation[j]@ref.rotation[i].T
        angles.append(float(np.rad2deg(Rotation.from_matrix(model_r@ref_r.T).magnitude())))
        t=model.rotation[j]@(model.center[i]-model.center[j])
        u=ref.rotation[j]@(ref.center[i]-ref.center[j])
        if min(np.linalg.norm(t),np.linalg.norm(u))<1e-7:raise ValueError("Translation direction undefined")
        directions.append(float(np.rad2deg(np.arccos(np.clip(t@u/(np.linalg.norm(t)*np.linalg.norm(u)),-1,1)))))
    focal=np.abs(np.log(model.intrinsic[:,[0,1],[0,1]]/ref.intrinsic[:,[0,1],[0,1]]))
    return dict(pair_rotation_median_deg=median(angles),pair_translation_direction_median_deg=median(directions),
                mean_abs_log_focal=float(np.mean(focal)),rotation_deg_by_pair=angles,translation_deg_by_pair=directions)


def depth_metrics(est,gt,mask):
    z,g=est[mask],gt[mask]
    if np.any(z<=0) or np.any(g<=0) or not (np.isfinite(z).all() and np.isfinite(g).all()):
        raise ValueError("Invalid axial depth")
    scale=float(np.median(g/z))
    ratio=scale*z/g
    return dict(depth_abs_rel=float(np.mean(np.abs(ratio-1))),depth_delta_1_25=float(np.mean(np.maximum(ratio,1/ratio)<1.25)),
                valid_depth_pixels=len(z),depth_alignment_scale=scale)


def score(study,subset,evidence):
    from scipy.spatial.transform import Rotation
    from repeatability_capture import array_digest
    pub=ROOT/"results"/study
    seal=read(pub/"seal.json")
    scene_rows=read(subset)
    rows=[]
    for scene in SCENES:
        if study=="sensor_extrinsic_v15":
            first=next(r for r in read(ROOT/"results/pointmap_precision_v9/summary.json")["rows"] if r["scene"]==scene)
            src=evidence/"runs"/("repeat6_"+scene.replace("/","_")+"_both_released_seed42_r0")
            source=evidence/read(src/"run.json")["evaluation_inputs_relative"]
            expected=first["reference_tensor_sha256"]
            maps=[(r["method"],r["seed"],ROOT/r["path"],r["sha256"]) for r in seal["rows"] if r["scene"]==scene]
        else:
            group=seal["common"][scene]
            source,expected=ROOT/group["inputs"],group["reference"]
            maps=[(m,42,ROOT/"reports"/study/"sealed"/key/"points.npy",seal["entries"][key]["sha256"])
                  for m,key in seal["maps"][scene].items()]
        arrays=dict(np.load(source,allow_pickle=False))
        if array_digest(arrays)!=expected:raise ValueError("Reference source changed")
        old=np.stack([effective_camera(e) for e in arrays["extrinsics"]])
        gt=np.einsum("nhwj,nj->nhw",arrays["reference_points"],old[:,2,:3])+old[:,2,3,None,None]
        poses=np.array(scene_rows[scene]["poses"])
        ref=np.repeat(np.eye(4)[None],5,axis=0)
        if not np.allclose(np.linalg.norm(poses[:,:4],axis=1),1,rtol=0,atol=1e-5):
            raise ValueError("Invalid published quaternion")
        ref[:,:3,:3]=Rotation.from_quat(poses[:,[1,2,3,0]]).as_matrix()
        ref[:,:3,3]=poses[:,4:7]
        k=np.stack([resized_k(f,gt.shape[2],gt.shape[1]) for f in poses[:,7]])
        rgb=relative_rig(ref,k)
        for method,seed,path,signature in maps:
            if sha(path)!=signature:raise ValueError("Sealed prediction changed")
            fitted=[recover(v) for v in np.load(path,allow_pickle=False)]
            rig=relative_rig(np.stack([c["extrinsics"] for c,_ in fitted]),np.stack([c["intrinsics"] for c,_ in fitted]))
            z=np.stack([d for _,d in fitted])
            rows.append(dict(scene=scene,method=method,seed=seed,prediction_sha256=signature,
                             **depth_metrics(z,gt,arrays["valid_mask"]),**camera_metrics(rig,rgb)))
    tables={}
    for method in sorted({r["method"] for r in rows}):
        scenes={s:{k:mean(r[k] for r in rows if r["method"]==method and r["scene"]==s) for k in KEYS} for s in SCENES}
        tables[method]=dict(scenes=scenes,overall={k:mean(v[k] for v in scenes.values()) for k in KEYS})
    if len(rows)!=(36 if study=="sensor_extrinsic_v15" else 33):raise ValueError("Incomplete metric grid")
    direct_methods=("direct","fixed_rgb","fitted_rgb","intrinsics_only","frame_only") if study=="sensor_extrinsic_v15" else ("direct_released","direct_fixed_rgb")
    control=max(abs(tables[m]["scenes"][s]["depth_abs_rel"]-tables["initial"]["scenes"][s]["depth_abs_rel"])
                for m in direct_methods for s in SCENES)
    dest=pub/"component_metrics.json"
    if dest.exists():raise FileExistsError(dest)
    dump(dest,dict(protocol_sha256=sha(PLAN),prediction_seal_sha256=sha(pub/"seal.json"),subset_sha256=sha(subset),
                   source_pose_sha256={s:v["file_sha256"] for s,v in scene_rows.items()},rows=rows,tables=tables,
                   direct_depth_abs_rel_max_difference=control,control_pass=control<=1e-5))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","fetch","score"))
    parser.add_argument("--study",choices=STUDIES)
    parser.add_argument("--evidence",type=Path)
    args=parser.parse_args()
    if args.action=="freeze":
        if PLAN.exists():raise FileExistsError(PLAN)
        dump(PLAN,plan())
    elif read(PLAN)!=plan():raise ValueError("Diagnostic changed")
    else:
        subset=ROOT/"reports"/args.study/"component_rgb_poses.json"
        if args.action=="fetch":public_rows(args.study,subset)
        else:score(args.study,subset,args.evidence)


if __name__=="__main__":main()
