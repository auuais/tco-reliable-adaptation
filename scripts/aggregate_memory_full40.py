"""Validate full-size local engineering feasibility, without an accuracy claim."""
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump,sha
from prior_ablation_config import validate_loss_components

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=ROOT/"manifests/local_memory_v3_full40.json";plan=json.loads(manifest.read_text())
    for name,digest in plan["sources"].items():assert sha(ROOT/name)==digest
    records=[];rows=[]
    for job in plan["jobs"]:
        folder=ROOT/"runs"/job["run_id"];record=json.loads((folder/"run.json").read_text())
        assert record["status"]=="complete" and not record["research_dirty"] and not record["base_parameters_modified"]
        assert record["image_shape"]==[5,3,392,518] and record["arguments"]["steps"]==job["steps"]
        assert record["manifest_sha256"]==sha(manifest) and record["configuration"]["tco_lr"]==.0003
        assert record["dataset_cache_sha256"]==plan["dataset_cache_sha256"]
        assert record["runner_sha256"]==plan["sources"]["scripts/run_experiment.py"]
        assert record["memory_optimization"]["wrapper_sha256"]==plan["sources"]["scripts/run_memory_probe.py"]
        predictions=np.load(folder/"predictions.npy",allow_pickle=False)
        assert predictions.shape==(5,392,518,3) and np.isfinite(predictions).all()
        if job["steps"]:
            losses=json.loads((folder/"losses.json").read_text());gradients=json.loads((folder/"gradients.json").read_text())
            assert len(losses)==len(gradients)==40
            for loss in losses:validate_loss_components(record["configuration"],loss)
            assert all(g["finite"] and g["gradient_norm_after_upstream_clipping"]>0 for g in gradients)
        if records:
            for key in ["input_sha256","reference_sha256","evaluation_tensor_sha256","upstream_method_sha256","versions","research_commit"]:
                assert record[key]==records[0][key]
        records.append(record)
        rows.append(dict(run_id=job["run_id"],steps=job["steps"],peak_allocated_GiB=record["peak_allocated_bytes"]/2**30,
            peak_reserved_GiB=record["peak_reserved_bytes"]/2**30,inference_and_adaptation_seconds=record["inference_and_adaptation_seconds"],
            model_load_seconds=record["model_load_seconds"],prediction_sha256=sha(folder/"predictions.npy")))
    original=np.load(ROOT/"runs/memory2_control_s0/predictions.npy",allow_pickle=False)
    current=np.load(ROOT/"runs/memory3_offload_s0/predictions.npy",allow_pickle=False)
    assert np.array_equal(original,current),"Fresh offload zero differs from historical canonical control"
    result=dict(study_id=plan["study_id"],status="complete",gpu=records[0]["gpu"],execution_commit=records[0]["research_commit"],
        manifest_sha256=sha(manifest),jobs=2,updates=40,zero_bitwise_equal_to_historical_canonical_control=True,
        all_gradients_and_predictions_finite=True,tracked_base_parameters_unchanged=True,runs=rows,
        limitations=["One scene and one adapted execution; feasibility only", "No adapted-output equivalence or accuracy comparison",
            "Shared Windows desktop GPU; observed times are not a controlled speed comparison", "Allocated memory is not physical VRAM residency",
            "Offloading inactive modules is an engineering change, not a novel reconstruction method"])
    dump(ROOT/"results/local_memory_v3_full40/summary.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
