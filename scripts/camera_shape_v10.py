"""Offline only: separate camera-prior fit from depth drift on immutable v6 outputs."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from evaluation_stability import sha, dump
from pointmap_camera_diagnostic import recover
from repeatability_capture import array_digest
from evaluate_fixed_matches import rank_correlation

ROOT=Path(__file__).resolve().parents[1]
STUDY="camera_shape_v10"
read=lambda p:json.loads(p.read_text())

def proper(M):
    u,_,v=np.linalg.svd(M)
    u[:,:,-1]*=np.linalg.det(u@v)[:,None]
    return u@v

def relative_rays(E):
    # Preserve the supplied camera center even for slightly nonorthogonal input R.
    E=E.astype(np.float64)
    centers=-np.linalg.solve(E[:,:3,:3],E[:,:3,3,None])[...,0]
    rotations=proper(E[:,:3,:3]);i,j=np.triu_indices(len(E),1)
    return rotations[j]@rotations[i].transpose(0,2,1), (centers[j]-centers[i])@rotations[0].T

def compare(E,K,targetE,targetK):
    r,d=relative_rays(E);r0,d0=relative_rays(targetE)
    angles=np.degrees(np.arccos(np.clip((np.trace(r@r0.transpose(0,2,1),axis1=1,axis2=2)-1)/2,-1,1)))
    denominator=np.sum(d*d);energy=np.sum(d0*d0);assert min(denominator,energy)>1e-12
    scale=max(0,float(np.sum(d*d0)/denominator))
    ids=[0,1]; focals=K[:,ids,ids];target=targetK[:,ids,ids];assert np.all(focals>0) and np.all(target>0)
    return dict(rotation_pair_mean_deg=float(np.mean(angles)),center_displacement_relative_rmse=float(np.sqrt(np.sum((scale*d-d0)**2)/energy)),focal_mean_abs_log=float(np.abs(np.log(focals/target)).mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");p.add_argument("--evidence",type=Path);args=p.parse_args()
    mf=ROOT/"manifests"/(STUDY+".json");v6=ROOT/"results/repeat_calibration_v6/per_run.csv"
    rows=list(csv.DictReader(v6.open()));assert len(rows)==40
    plan=dict(study_id=STUDY,cells=[r["run_id"] for r in rows],rows_sha256=sha(v6),
        sources={f:sha(ROOT/f) for f in ["scripts/camera_shape_v10.py","scripts/pointmap_camera_diagnostic.py","scripts/evaluate_fixed_matches.py"]},
        protocol="Use initial captured cameras/depth, recover adapted cameras with frozen v9 DLT guard. Reject entire adapted run if any view fails. Report all rows and reasons. Ten pair relative rotation angles in degrees; center displacement vectors in first camera axes after fitting one positive least-squares scale; focal mean absolute log ratio. Compare to that run's supplied priors (including calibration variants), not independent truth. Dense depth drift: absolute log(final/initial) after subtracting one global all-pixel median log-ratio. No GT valid masks, alignment or reference depth in drift. Depth drift has no correctness sign.",
        analysis="Post-hoc mechanism diagnosis after v6 labels were known. For all 37 adapted rows and descriptively by scene, report median after-minus-before residuals, counts improving, harm and unadjusted rank association with original v6 reconstruction score. These repeated scenes/starts are not independent. No causal or generalization test. No gate or new spending.")
    if args.freeze:
        assert not mf.exists();dump(mf,plan);return
    assert read(mf)==plan and args.evidence is not None
    out=ROOT/"results"/STUDY;out.mkdir(exist_ok=False);result=[]
    for row in rows:
        folder=args.evidence/"runs"/row["run_id"];cap=read(folder/"capture.json")["records"]["initial_predictions"]
        assert sha(folder/cap["file"])==cap["file_sha256"] and sha(folder/"predictions.npy")==row["prediction_file_sha256"]
        initial=dict(np.load(folder/cap["file"],allow_pickle=False));assert array_digest(initial)==cap["tensor_sha256"]
        run=read(folder/"run.json");K,E,Z=initial["intrinsics"][0],initial["extrinsics"][0],initial["depth"][0,...,0]
        priors=dict(np.load(folder/"supplied_priors.npz",allow_pickle=False));before=compare(E,K,priors["extrinsics"],priors["intrinsics"])
        b=dict(run_id=row["run_id"],scene=row["scene"],condition=row["condition"],calibration=row["calibration"],seed=int(row["seed"]),repeat=int(row["execution_repeat"]),before=before,initial_heads_sha256=cap["tensor_sha256"],final_points_sha256=row["prediction_file_sha256"],v6_primary_score=float(row["score"]),v6_harmful=row["harmful"]=="True")
        if not int(row["steps"]):
            result.append(dict(**b,status="zero_exact_initial_components"));continue
        v=[];errors=[]
        for i,points in enumerate(np.load(folder/"predictions.npy",allow_pickle=False)):
            try:v.append(recover(points))
            except ValueError as exc:errors.append(dict(view=i,reason=str(exc)))
        if errors:
            result.append(dict(**b,status="recovery_rejected",errors=errors));continue
        finalE=np.stack([c["extrinsics"] for c,z in v]);finalK=np.stack([c["intrinsics"] for c,z in v]);z=np.stack([z for c,z in v])
        assert np.isfinite(Z).all() and np.all(Z>0)
        log=np.log(z/Z);med=float(np.median(log));drift=np.abs(log-med)
        after=compare(finalE,finalK,priors["extrinsics"],priors["intrinsics"])
        result.append(dict(**b,status="recovered",after=after,delta={k:after[k]-before[k] for k in before},depth_median_log_scale=med,
          depth_drift_median_abs_log=float(np.median(drift)),depth_drift_p90_abs_log=float(np.percentile(drift,90)),
          max_reprojection_pixels=max(c["dense_reprojection_max_pixels"] for c,z in v)))
    adapted=[x for x in result if x["status"]=="recovered"]
    def summarize(group):
        if not group:return dict(n=0)
        output=dict(n=len(group),harmful=sum(x["v6_harmful"] for x in group))
        for k in list(before)+["depth_drift_median_abs_log"]:
            values=[x[k] if k.startswith("depth") else x["delta"][k] for x in group]
            output[k]=dict(median=float(np.median(values)),rank_with_v6_primary_score=rank_correlation(values,[x["v6_primary_score"] for x in group]))
            if not k.startswith("depth"):output[k]["closer_to_supplied_prior"]=sum(v<0 for v in values)
        return output
    dump(out/"summary.json",dict(manifest_sha256=sha(mf),rows=result,scenes={s:summarize([x for x in adapted if x["scene"]==s]) for s in sorted(set(r["scene"] for r in rows))},
      all_adapted=summarize(adapted),rejected=sum(x["status"]=="recovery_rejected" for x in result),scope=plan["analysis"]))
if __name__=="__main__":main()
