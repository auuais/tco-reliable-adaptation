"""Fail-closed complete-batch validation and descriptive calibration/repeat results."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump, sha
from prior_ablation_config import effective_config, validate_loss_components
from repeatability_capture import array_digest

ROOT=Path(__file__).resolve().parents[1]
STUDY="repeat_calibration_v6"
METRICS=["accuracy_mean_m","completeness_mean_m","normal_consistency_mean"]


def read(path):return json.loads(path.read_text())


def validate_job(job,record,manifest_digest,runner_digest):
    assert record["status"]=="complete" and not record["research_dirty"] and not record["base_parameters_modified"]
    assert record["manifest_sha256"]==manifest_digest and record["runner_sha256"]==runner_digest
    keys=dict(scene="scene",calibration="calibration",seed="seed",execution_repeat="execution_repeat",
              steps="steps",size="size",frames="frames",learning_rate="learning_rate",
              pose_scale="pose_weight_scale",intrinsics_scale="intrinsics_weight_scale")
    for planned,actual in keys.items():assert record["arguments"][actual]==job[planned],(planned,job["run_id"])
    c=record["configuration"]
    assert c["lambda_pose"]==job["pose_scale"] and c["lambda_intrinsics"]==.01*job["intrinsics_scale"]
    assert c["tco_lr"]==.0003 and c["lambda_mv_consistency"]==.2
    assert record["image_shape"]==[5,3,392,518]
    assert all(np.isfinite(record["metrics"][k]) for k in METRICS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=ROOT/"results"/STUDY)
    args=parser.parse_args()
    manifest_path=ROOT/"manifests"/(STUDY+".json")
    manifest=read(manifest_path);digest=sha(manifest_path)
    assert read(args.evidence_root/"manifests"/(STUDY+".json"))==manifest
    status=read(args.evidence_root/"reports"/STUDY/"batch_status.json")
    assert [s["run_id"] for s in status]==[j["run_id"] for j in manifest["jobs"]]
    assert all(s["status"]=="complete" for s in status)
    rows=[];records={};captures={};identities=set();source_signatures=set();scene_signatures={}
    for job in manifest["jobs"]:
        folder=args.evidence_root/"runs"/job["run_id"]
        record=read(folder/"run.json")
        validate_job(job,record,digest,manifest["sources"]["scripts/run_repeat_calibration.py"])
        identities.add(record["gpu_identity"])
        source_signatures.add(json.dumps({k:record[k] for k in ["upstream_commit","upstream_method_sha256","research_commit",
                               "versions","numerical_settings","gsplat_backend","checkpoint_download"]},sort_keys=True))
        sig={k:record[k] for k in ["input_sha256","reference_sha256","evaluation_tensor_sha256"]}
        assert scene_signatures.setdefault(job["scene"],sig)==sig,"Unmatched scene inputs/reference"
        prediction=np.load(folder/"predictions.npy",allow_pickle=False)
        assert prediction.shape==(5,392,518,3) and np.isfinite(prediction).all()
        inputs_file=args.evidence_root/record["evaluation_inputs_relative"]
        arrays=dict(np.load(inputs_file,allow_pickle=False))
        assert array_digest(arrays)==record["evaluation_tensor_sha256"]
        supplied=dict(np.load(folder/"supplied_priors.npz",allow_pickle=False))
        from calibration_variants import calibration_prior
        np.testing.assert_array_equal(supplied["extrinsics"],arrays["extrinsics"])
        np.testing.assert_array_equal(supplied["intrinsics"],calibration_prior(arrays["intrinsics"],518,392,job["calibration"]))
        from prior_corruptions import tensor_hash
        assert record["prior_metadata"]["intrinsics_sha256"]==tensor_hash(supplied["intrinsics"])
        capture=read(folder/"capture.json")
        for item in capture["records"].values():
            assert sha(folder/item["file"])==item.get("file_sha256",item.get("sha256"))
            if "tensor_sha256" in item:
                assert array_digest(dict(np.load(folder/item["file"],allow_pickle=False)))==item["tensor_sha256"]
        required=["seeded_rng","before_inference_rng","initial_predictions","before_adapter_initialization_rng","after_inference_rng"]
        if job["steps"]:required += ["initial_adapters","final_adapters","after_adapter_initialization_rng"]
        assert all(k in capture["records"] for k in required)
        if job["steps"]:
            losses=read(folder/"losses.json");gradients=read(folder/"gradients.json")
            residuals=read(folder/"calibration_residuals.json")
            assert len(losses)==len(gradients)==len(residuals)==job["steps"]
            for index,residual in enumerate(residuals):
                assert residual["step"]==index and residual["active"]==bool(job["intrinsics_scale"])
                target=np.asarray(residual["target_focal_xy_pixels"])
                np.testing.assert_array_equal(target,supplied["intrinsics"][None,...,[0,1],[0,1]])
                actual=np.asarray(residual["predicted_focal_xy_pixels"],dtype=np.float32)-target.astype(np.float32)
                np.testing.assert_array_equal(actual,np.asarray(residual["residual_xy_pixels"],dtype=np.float32))
                np.testing.assert_array_equal(np.clip(actual,-1,1),residual["smooth_l1_derivative_before_reduction"])
            for loss in losses:validate_loss_components(record["configuration"],loss)
            assert all(g["finite"] and g["gradient_norm_after_upstream_clipping"]>0 for g in gradients)
        assert read(folder/"metrics.json")==record["metrics"]
        row={**job,**{k:record["metrics"][k] for k in METRICS},
             "inference_and_adaptation_seconds":record["inference_and_adaptation_seconds"],
             "capture_observation_seconds":record["capture_observation_seconds"],
             "peak_allocated_GiB":record["peak_allocated_bytes"]/2**30,
             "prediction_file_sha256":sha(folder/"predictions.npy")}
        rows.append(row);records[job["run_id"]]=record;captures[job["run_id"]]=capture
    assert len(identities)==len(source_signatures)==1
    zero={r["scene"]:r for r in rows if r["condition"]=="zero"}
    for row in rows:
        base=zero[row["scene"]]
        row["score"]=(row[METRICS[0]]/base[METRICS[0]]+row[METRICS[1]]/base[METRICS[1]])/2
        row["harmful"]=any(row[k]>1.01*base[k] for k in METRICS[:2])
    calibration=[r for r in rows if r["condition"]=="both" and r["execution_repeat"]==0]
    repeats=[]
    for scene in ["heads/seq-01","fire/seq-03"]:
        for condition in ["both","visual"]:
            group=sorted([r for r in rows if r["scene"]==scene and r["condition"]==condition and r["calibration"]=="released" and r["seed"]==42],key=lambda r:r["execution_repeat"])
            assert [r["execution_repeat"] for r in group]==[0,1,2]
            distances=[]
            for a,b in itertools.combinations(group,2):
                pa=np.load(args.evidence_root/"runs"/a["run_id"]/"predictions.npy")
                pb=np.load(args.evidence_root/"runs"/b["run_id"]/"predictions.npy")
                delta=pa.astype(np.float64)-pb
                distances.append(dict(a=a["run_id"],b=b["run_id"],bitwise_equal=bool(np.array_equal(pa,pb)),
                                      raw_coordinate_rmse=float(np.sqrt(np.mean(delta**2))),max_absolute=float(np.max(np.abs(delta)))))
            hashes={name:[captures[r["run_id"]]["records"][name].get("tensor_sha256",captures[r["run_id"]]["records"][name].get("sha256")) for r in group]
                    for name in ["seeded_rng","before_inference_rng","initial_predictions","before_adapter_initialization_rng",
                                 "initial_adapters","after_adapter_initialization_rng","after_inference_rng","final_adapters"]}
            repeats.append(dict(scene=scene,condition=condition,run_ids=[r["run_id"] for r in group],scores=[r["score"] for r in group],
                                score_range=max(r["score"] for r in group)-min(r["score"] for r in group),
                                state_hashes=hashes,all_equal_by_state={k:len(set(v))==1 for k,v in hashes.items()},prediction_differences=distances))
    maximum_noise=max(r["score_range"] for r in repeats if r["condition"]=="both")
    conditions={}
    for variant in ["released","axis_nominal","axis_registration"]:
        group=[r for r in calibration if r["calibration"]==variant]
        assert len(group)==9
        conditions[variant]=dict(score=float(np.mean([r["score"] for r in group])),harmful_cells=sum(r["harmful"] for r in group),
            scenes={scene:dict(score=float(np.mean([r["score"] for r in group if r["scene"]==scene])),
                scores=[r["score"] for r in group if r["scene"]==scene]) for scene in zero},
            means={k:float(np.mean([r[k] for r in group])) for k in METRICS})
    released=conditions["released"]
    for name,value in conditions.items():
        if name=="released":continue
        absolute_gain=released["score"]-value["score"]
        checks=dict(relative_gain_at_least_5pct=absolute_gain/released["score"]>=.05,
            harm_not_increased=value["harmful_cells"]<=released["harmful_cells"],
            no_scene_regression_over_1pct=all(value["scenes"][s]["score"]<=1.01*released["scenes"][s]["score"] for s in zero),
            gain_exceeds_observed_repeat_range=absolute_gain>maximum_noise)
        value.update(relative_gain=absolute_gain/released["score"],gate_checks=checks,passes_development_diagnostic_gate=all(checks.values()))
    summary=dict(study_id=STUDY,manifest_sha256=digest,completed_jobs=len(rows),updates=sum(r["steps"] for r in rows),
        all_evidence_validated=True,single_physical_gpu=True,common_reference_verified=True,conditions=conditions,repeats=repeats,
        maximum_both_repeat_score_range=maximum_noise,rows=rows,execution_commit=next(iter(records.values()))["research_commit"],
        gpu=next(iter(records.values()))["gpu"],batch_seconds=sum(s["seconds"] for s in status),
        limitation="Development diagnostics only. Calibration interventions are engineering controls, not a demonstrated novel transferable method.")
    dump(args.output/"summary.json",summary)
    loss_rows=[]
    for job in manifest["jobs"]:
        if not job["steps"]:continue
        folder=args.evidence_root/"runs"/job["run_id"]
        for loss,residual in zip(read(folder/"losses.json"),read(folder/"calibration_residuals.json")):
            loss_rows.append(dict(run_id=job["run_id"],**loss,
                focal_fraction_in_quadratic_region=residual["fraction_in_quadratic_region"]))
    dump(args.output/"per_step_calibration.json",{j["run_id"]:read(args.evidence_root/"runs"/j["run_id"]/"calibration_residuals.json") for j in manifest["jobs"] if j["steps"]})
    with (args.output/"per_step_losses.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(loss_rows[0]));writer.writeheader();writer.writerows(loss_rows)
    with (args.output/"per_run.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(completed_jobs=len(rows),conditions=conditions,maximum_both_repeat_score_range=maximum_noise),indent=2))


if __name__=="__main__":main()
