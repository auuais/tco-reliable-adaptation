"""Validate the complete frozen local study and report all outcomes, including nulls."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump, sha
from prior_ablation_config import validate_loss_components
from repeatability_capture import array_digest

ROOT=Path(__file__).resolve().parents[1]
STUDY="pose_precision_v7_local"
METRICS=["accuracy_mean_m","completeness_mean_m","normal_consistency_mean"]


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--evidence-root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=ROOT/"results"/STUDY);args=parser.parse_args()
    manifest=ROOT/"manifests"/(STUDY+".json");plan=read(manifest);digest=sha(manifest)
    for name,expected in plan["sources"].items():assert sha(ROOT/name)==expected,name
    status=read(args.evidence_root/"reports"/STUDY/"batch_status.json")
    assert [r["run_id"] for r in status]==[j["run_id"] for j in plan["jobs"]]
    assert all(r["status"]=="complete" for r in status)
    records={};rows=[];inputs={};captures={};signature=None
    for job in plan["jobs"]:
        folder=args.evidence_root/"runs"/job["run_id"];r=read(folder/"run.json")
        assert r["status"]=="complete" and not r["research_dirty"] and not r["base_parameters_modified"]
        assert r["study_id"]==STUDY and r["manifest_sha256"]==digest and r["fresh_model_per_run"]
        assert r["gpu"]==plan["local_gpu"] and r["image_shape"]==[5,3,392,518]
        for key in ["scene","seed","steps","execution_repeat"]:assert r["arguments"][key]==job[key]
        for key in ["frames","size","learning_rate","calibration"]:assert r["arguments"][key]==plan[key]
        assert r["arguments"]["pose_weight_scale"]==r["arguments"]["intrinsics_weight_scale"]==1.
        c=r["configuration"]
        assert c["lambda_pose"]==1 and c["lambda_intrinsics"]==.01 and c["lambda_mv_consistency"]==.2 and c["tco_lr"]==.0003
        assert r["runner_sha256"]==plan["sources"]["scripts/run_repeat_calibration.py"]
        assert r["precision_study_wrapper_sha256"]==plan["sources"]["scripts/run_pose_precision_study.py"]
        assert r["pose_precision_intervention"]["wrapper_sha256"]==plan["sources"]["scripts/run_pose_precision.py"]
        assert r["pose_precision_intervention"]["mode"]==job["mode"]
        assert r["memory_optimization"]["wrapper_sha256"]==plan["sources"]["scripts/run_memory_probe.py"]
        assert r["dataset_cache_sha256"]==plan["dataset_cache_sha256"]
        sig={k:r[k] for k in ["research_commit","gpu_identity","upstream_commit","upstream_method_sha256","versions","checkpoint_download","gsplat_backend"]}
        sig["numerical_settings"]={k:v for k,v in r["numerical_settings"].items() if k!="pythonhashseed"}
        assert r["numerical_settings"]["pythonhashseed"]==str(job["seed"])
        if signature is None:signature=sig
        assert sig==signature,"Execution provenance mismatch"
        scene_sig={k:r[k] for k in ["input_sha256","reference_sha256","evaluation_tensor_sha256"]}
        assert inputs.setdefault(job["scene"],scene_sig)==scene_sig
        arrays=dict(np.load(args.evidence_root/r["evaluation_inputs_relative"],allow_pickle=False))
        assert array_digest(arrays)==r["evaluation_tensor_sha256"]
        priors=dict(np.load(folder/"supplied_priors.npz",allow_pickle=False))
        for key in ["extrinsics","intrinsics"]:np.testing.assert_array_equal(priors[key],arrays[key])
        pred=np.load(folder/"predictions.npy",allow_pickle=False)
        assert pred.shape==(5,392,518,3) and np.isfinite(pred).all()
        cap=read(folder/"capture.json")["records"]
        required=["seeded_rng","before_inference_rng","initial_predictions","before_adapter_initialization_rng","after_inference_rng"]
        if job["steps"]:required += ["initial_adapters","final_adapters","after_adapter_initialization_rng"]
        assert all(k in cap for k in required)
        for item in cap.values():
            assert sha(folder/item["file"])==item.get("file_sha256",item.get("sha256"))
            if "tensor_sha256" in item:assert array_digest(dict(np.load(folder/item["file"],allow_pickle=False)))==item["tensor_sha256"]
        if job["steps"]:
            losses=read(folder/"losses.json");grads=read(folder/"gradients.json");calls=read(folder/"pose_precision.json")["calls"]
            assert len(losses)==len(grads)==len(calls)==40
            for loss in losses:validate_loss_components(c,loss)
            assert all(g["finite"] and g["gradient_norm_after_upstream_clipping"]>0 for g in grads)
            assert all(call["outer_autocast_enabled"] and call["input_predicted_dtype"]=="torch.float32" for call in calls)
            first=read(folder/"first_clipped_gradients.json")
            assert sha(folder/"first_clipped_gradients.npz")==first["file_sha256"]
            assert array_digest(dict(np.load(folder/"first_clipped_gradients.npz",allow_pickle=False)))==first["tensor_sha256"]
        assert all(np.isfinite(r["metrics"][k]) for k in METRICS)
        np.testing.assert_equal(read(folder/"metrics.json"),r["metrics"])
        records[job["run_id"]]=r;captures[job["run_id"]]=cap
        rows.append(dict(**job,metrics={k:r["metrics"][k] for k in METRICS},prediction_sha256=sha(folder/"predictions.npy"),
                         inference_and_adaptation_seconds=r["inference_and_adaptation_seconds"],peak_allocated_GiB=r["peak_allocated_bytes"]/2**30,
                         peak_reserved_GiB=r["peak_reserved_bytes"]/2**30))
    zero={r["scene"]:r for r in rows if r["steps"]==0}
    for scene in zero:
        for seed in [42,123,2026]:
            pair=[r for r in rows if r["scene"]==scene and r["seed"]==seed and r["steps"]==40 and r["execution_repeat"]==0]
            assert len(pair)==2
            for key in ["initial_predictions","initial_adapters"]:
                assert captures[pair[0]["run_id"]][key]["tensor_sha256"]==captures[pair[1]["run_id"]][key]["tensor_sha256"],"Unmatched initialization"
            a,b=[dict(np.load(args.evidence_root/"runs"/r["run_id"]/"pose_inputs_step0.npz",allow_pickle=False)) for r in pair]
            for key in a:np.testing.assert_array_equal(a[key],b[key])
    for row in rows:
        base=zero[row["scene"]]["metrics"]
        row["score"]=float(np.mean([row["metrics"][k]/base[k] for k in METRICS[:2]]))
        row["harmful"]=any(row["metrics"][k]>1.01*base[k] for k in METRICS[:2])
    repeats=[]
    for scene in ["heads/seq-01","fire/seq-03"]:
        for mode in ["released","float32"]:
            group=[r for r in rows if r["scene"]==scene and r["mode"]==mode and r["seed"]==42 and r["steps"]==40]
            assert len(group)==3
            keys=["initial_predictions","initial_adapters","after_adapter_initialization_rng","after_inference_rng","final_adapters"]
            equality={k:len({captures[r["run_id"]][k].get("tensor_sha256",captures[r["run_id"]][k].get("sha256")) for r in group})==1 for k in keys}
            repeats.append(dict(scene=scene,mode=mode,scores=[r["score"] for r in group],score_range=float(np.ptp([r["score"] for r in group])),state_equal=equality,
                                first_clipped_gradients_equal=len({read(args.evidence_root/"runs"/r["run_id"]/"first_clipped_gradients.json")["tensor_sha256"] for r in group})==1))
    conditions={}
    for mode in ["released","float32"]:
        group=[r for r in rows if r["mode"]==mode and r["steps"]==40 and r["execution_repeat"]==0];assert len(group)==9
        conditions[mode]=dict(score=float(np.mean([r["score"] for r in group])),harmful_cells=sum(r["harmful"] for r in group),
            scenes={s:float(np.mean([r["score"] for r in group if r["scene"]==s])) for s in zero},
            means={k:float(np.mean([r["metrics"][k] for r in group])) for k in METRICS})
    old,new=conditions["released"],conditions["float32"];gain=old["score"]-new["score"];noise=max(r["score_range"] for r in repeats)
    gate=plan["gate"]
    checks=dict(relative_gain_at_least_threshold=gain/old["score"]>=gate["min_relative_gain"],
        harm_not_increased=new["harmful_cells"]<=old["harmful_cells"],
        no_scene_regression_over_threshold=all(new["scenes"][s]<=old["scenes"][s]*(1+gate["max_scene_relative_regression"]) for s in zero),
        absolute_gain_exceeds_repeat_range=gain>noise)
    result=dict(study_id=STUDY,all_evidence_validated=True,completed_jobs=len(rows),updates=sum(r["steps"] for r in rows),
        execution_commit=signature["research_commit"],manifest_sha256=digest,conditions=conditions,repeats=repeats,
        relative_gain=gain/old["score"],maximum_repeat_range=noise,gate_checks=checks,passes_development_gate=all(checks.values()),
        later_allocations_automatically_released=False,rows=rows,
        limitations=["Three development sequences; not held-out or cross-dataset evidence", "Precision repair is an engineering baseline, not established novelty",
                     "Both modes use local inactive-module offloading and capture overhead; no cross-device performance claim",
                     "GPU reservation can exceed physical VRAM under Windows/WSL", "Observed ranges are not confidence intervals"])
    args.output.mkdir(parents=True,exist_ok=True);dump(args.output/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ["rows"]},indent=2))


if __name__=="__main__":main()
