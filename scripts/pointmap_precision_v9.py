"""CPU control: identical saved model heads, different coordinate arithmetic only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]
STUDY="pointmap_precision_v9"
SCENES=["heads/seq-01","chess/seq-03","fire/seq-03"]
MODES=["saved_zero","upstream_cpu_float32","upstream_cpu_bfloat16","numpy_float64"]


def worker(index,evidence,output):
    import random
    import torch
    import open3d as o3d
    from run_repeat_calibration import evaluate,tco_module
    from pointmap_camera_diagnostic import recover
    from repeatability_capture import array_digest
    manifest=json.loads((ROOT/"manifests"/(STUDY+".json")).read_text());job=manifest["jobs"][index]
    scene,mode=job["scene"],job["mode"];folder=output/job["job_id"];folder.mkdir(parents=True,exist_ok=False)
    source=evidence/"runs"/f"repeat6_{scene.replace('/','_')}_zero_released_seed42_r0"
    record=json.loads((source/"run.json").read_text());capture=json.loads((source/"capture.json").read_text())["records"]["initial_predictions"]
    assert sha(source/capture["file"])==capture["file_sha256"]
    initial=dict(np.load(source/capture["file"],allow_pickle=False));assert array_digest(initial)==capture["tensor_sha256"]
    inputs=dict(np.load(evidence/record["evaluation_inputs_relative"],allow_pickle=False));assert array_digest(inputs)==record["evaluation_tensor_sha256"]
    original=np.load(source/"predictions.npy",allow_pickle=False)
    depth=initial["depth"][0,...,0];extrinsic=initial["extrinsics"][0];intrinsic=initial["intrinsics"][0]
    y,x=np.meshgrid(np.arange(depth.shape[1]),np.arange(depth.shape[2]),indexing="ij");double=[]
    for d,e,k in zip(depth.astype(np.float64),extrinsic.astype(np.float64),intrinsic.astype(np.float64)):
        camera=np.stack([(x-k[0,2])*d/k[0,0],(y-k[1,2])*d/k[1,1],d],axis=-1)
        double.append(camera@e[:3,:3]-e[:3,:3].T@e[:3,3])
    double=np.stack(double)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);o3d.utility.random.seed(42)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if mode=="saved_zero":predicted=original
    elif mode=="numpy_float64":predicted=double
    else:
        with torch.no_grad(),torch.autocast("cpu",dtype=torch.bfloat16,enabled=mode.endswith("bfloat16")):
            value=tco_module.unproject_depth_to_world_points_torch(
                torch.from_numpy(initial["depth"]),torch.from_numpy(initial["extrinsics"]),torch.from_numpy(initial["intrinsics"]))
        predicted=value[0].float().numpy()
    assert predicted.shape==original.shape and np.isfinite(predicted).all()
    np.save(folder/"predictions.npy",predicted)
    cameras=[]
    for i,points in enumerate(predicted):
        try:
            camera,z=recover(points)
            cameras.append(dict(view=i,status="recovered",dense_reprojection_max_pixels=camera["dense_reprojection_max_pixels"],
                intrinsic_max_absolute_error=float(np.max(np.abs(camera["intrinsics"]-intrinsic[i]))),
                extrinsic_max_absolute_error=float(np.max(np.abs(camera["extrinsics"]-extrinsic[i]))),
                depth_max_absolute_error=float(np.max(np.abs(z-depth[i])))))
        except ValueError as exc:cameras.append(dict(view=i,status="rejected",reason=str(exc)))
    metrics=evaluate(predicted,dict(images=torch.from_numpy(inputs["images"]),pointclouds=inputs["reference_points"],valid_mask=inputs["valid_mask"]),folder)
    result=dict(**job,metrics=metrics,initial_heads_tensor_sha256=capture["tensor_sha256"],reference_tensor_sha256=record["evaluation_tensor_sha256"],
        saved_prediction_sha256=sha(source/"predictions.npy"),prediction_sha256=sha(folder/"predictions.npy"),output_dtype=str(predicted.dtype),
        bitwise_equal_to_saved=bool(np.array_equal(predicted,original)),
        fraction_float32_values_exactly_bfloat16=float(np.mean((predicted.astype(np.float32).view(np.uint32)&65535)==0)),
        max_raw_coordinate_error_vs_float64=float(np.max(np.abs(predicted-double))),raw_coordinate_rmse_vs_float64=float(np.sqrt(np.mean((predicted-double)**2))),
        camera_recovery=cameras,status="complete",model_inference_performed=False,adaptation_performed=False)
    dump(folder/"result.json",result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--evidence-root",type=Path);parser.add_argument("--worker",type=int)
    parser.add_argument("--upstream-geometry",type=Path,default=ROOT/"third_party/TCO/utils/geometry.py")
    args=parser.parse_args();manifest=ROOT/"manifests"/(STUDY+".json");output=ROOT/"reports"/STUDY
    if args.freeze:
        assert not manifest.exists()
        plan=dict(study_id=STUDY,jobs=[dict(job_id=f"{s.replace('/','_')}_{mode}",scene=s,mode=mode) for s in SCENES for mode in MODES],
            sources={name:sha(ROOT/name) for name in ["scripts/pointmap_precision_v9.py","scripts/pointmap_camera_diagnostic.py","scripts/run_repeat_calibration.py"]},
            upstream_geometry_lf_sha256=__import__('hashlib').sha256(args.upstream_geometry.read_bytes().replace(b"\r\n",b"\n")).hexdigest(),
            hypothesis="Our TCO zero-step path computes points outside the head's autocast-disabled scope; adapted final points are computed inside it. Isolate coordinate arithmetic using identical saved depth/camera tensors.",
            modes=MODES,planned_evaluations=12,threads=4,seed=42,timeout_seconds_per_job=600,
            scope="Post-hoc CPU diagnostic of this reproduction's zero-step comparator, not proof about the authors' published baseline. No model rerun or cloud compute.",
            metric_reference="Unchanged released v6 reference for every mode; saved zero rescored in the same CPU runtime",
            recovery_guard="DLT null-space gap >=1000, all-pixel reprojection <=0.02px, positive depth; reject rather than relax after observing results",
            later_allocations_released=False)
        dump(manifest,plan)
    plan=json.loads(manifest.read_text())
    for name,digest in plan["sources"].items():assert sha(ROOT/name)==digest,name
    if args.freeze or args.dry_run:print(json.dumps(dict(study=STUDY,manifest_sha256=sha(manifest),evaluations=12)));return
    assert args.evidence_root is not None
    assert __import__('hashlib').sha256(args.upstream_geometry.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==plan["upstream_geometry_lf_sha256"]
    if args.worker is not None:worker(args.worker,args.evidence_root,output);return
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42")
    for index,job in enumerate(plan["jobs"]):
        print("Pointmap precision",index+1,"/12",job["job_id"],flush=True)
        command=[sys.executable,"-u",str(Path(__file__)),"--worker",str(index),"--evidence-root",str(args.evidence_root)]
        with (output/(job["job_id"]+".log")).open("x") as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=plan["timeout_seconds_per_job"])
    rows=[json.loads((output/job["job_id"]/"result.json").read_text()) for job in plan["jobs"]]
    contrasts=[]
    for scene in SCENES:
        selected={r["mode"]:r for r in rows if r["scene"]==scene};base=selected["saved_zero"]
        assert len({r["initial_heads_tensor_sha256"] for r in selected.values()})==len({r["reference_tensor_sha256"] for r in selected.values()})==1
        contrasts.append(dict(scene=scene,modes={mode:dict(metrics=r["metrics"],relative_distance_change_vs_saved={k:r["metrics"][k]/base["metrics"][k]-1 for k in ["accuracy_mean_m","completeness_mean_m"]},
            max_raw_coordinate_error_vs_float64=r["max_raw_coordinate_error_vs_float64"],camera_recovery_passes=sum(c["status"]=="recovered" for c in r["camera_recovery"])) for mode,r in selected.items()}))
    public=ROOT/"results"/STUDY;public.mkdir(parents=True,exist_ok=False)
    dump(public/"summary.json",dict(study_id=STUDY,manifest_sha256=sha(manifest),completed_evaluations=len(rows),rows=rows,contrasts=contrasts,
        interpretation=plan["scope"],historical_primary_results_unchanged=True,later_allocations_released=False))
    print(json.dumps(contrasts,indent=2))


if __name__=="__main__":main()
