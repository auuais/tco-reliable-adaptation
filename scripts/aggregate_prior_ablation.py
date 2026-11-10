"""Fail-closed validation and summaries for the frozen v5 component ablation."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics

import numpy as np
from prior_corruptions import tensor_hash
from prior_ablation_config import CONDITIONS, effective_config, validate_loss_components

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ["accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean"]


def validate_record(record, job, plan, manifest_hash, control):
    assert record["status"] == "complete" and not record["research_dirty"]
    assert record["study_id"] == plan["study_id"]
    assert record["manifest_sha256"] == manifest_hash
    assert not record["base_parameters_modified"]
    expected = dict(scene=job["scene"], steps=job["steps"], frames=job["frames"],
        size=job["size"], seed=job["seed"], prior_profile=job["profile"],
        prior_weight=job["weight"], prior_seed=job["prior_seed"], learning_rate=job["learning_rate"],
        pose_weight_scale=job["pose_scale"], intrinsics_weight_scale=job["intrinsics_scale"])
    assert all(record["arguments"][k] == v for k, v in expected.items())
    assert record["runner_sha256"] == plan["sources"]["scripts/run_prior_ablation.py"]
    assert record["prior_generator_sha256"] == plan["sources"]["scripts/prior_corruptions.py"]
    for key in ["research_commit", "input_sha256", "reference_sha256", "upstream_commit",
                "upstream_method_sha256", "versions", "gpu", "gpu_identity", "gsplat_backend",
                "checkpoint_download", "image_shape", "evaluation_tensor_sha256", "prior_metadata"]:
        assert record[key] == control[key], key
    assert record["ablation_config_sha256"] == plan["sources"]["scripts/prior_ablation_config.py"]
    cfg = effective_config(control["configuration"], job["pose_scale"], job["intrinsics_scale"],
                           job["steps"], job["seed"], job["learning_rate"])
    assert record["configuration"] == cfg
    if job["steps"]:
        assert record["optimizer_learning_rates"] and all(v == job["learning_rate"] for v in record["optimizer_learning_rates"])
    assert all(math.isfinite(record["metrics"][k]) for k in METRICS)
    assert record["metrics"]["protocol"] == control["metrics"]["protocol"]
    assert record["inference_and_adaptation_seconds"] > 0


def summarize(values):
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=pathlib.Path, default=ROOT/"runs")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT/"reports/prior_ablation_v5")
    args = parser.parse_args()
    manifest = ROOT/"manifests/prior_ablation_v5.json"
    plan = json.loads(manifest.read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    records = {j["run_id"]: json.loads((args.runs/j["run_id"]/"run.json").read_text()) for j in plan["jobs"]}
    controls = {j["scene"]: records[j["run_id"]] for j in plan["jobs"] if j["condition"] == "zero"}
    rows = []
    for j in plan["jobs"]:
        record, folder = records[j["run_id"]], args.runs/j["run_id"]
        control = controls[j["scene"]]
        validate_record(record, j, plan, digest, control)
        prior = np.load(folder/"supplied_priors.npz")
        for name, field in [("extrinsics", "extrinsics_sha256"), ("intrinsics", "intrinsics_sha256")]:
            assert tensor_hash(prior[name]) == record["prior_metadata"][field]
        gradients = json.loads((folder/"gradients.json").read_text()) if j["steps"] else []
        losses = json.loads((folder/"losses.json").read_text()) if j["steps"] else []
        assert len(gradients) == len(losses) == j["steps"]
        assert all(g["finite"] and g["gradient_norm_after_upstream_clipping"] > 0 for g in gradients)
        assert all(math.isfinite(v) for loss in losses for v in loss.values())
        for loss in losses: validate_loss_components(record["configuration"], loss)
        prediction = np.load(folder/"predictions.npy", mmap_mode="r")
        assert prediction.shape == (5, 392, 518, 3) and np.isfinite(prediction).all()
        row = dict(run_id=j["run_id"], scene=j["scene"], condition=j["condition"], seed=j["seed"],
            steps=j["steps"], learning_rate=j["learning_rate"], prior_weight=j["weight"], pose_scale=j["pose_scale"], intrinsics_scale=j["intrinsics_scale"],
            seconds=record["inference_and_adaptation_seconds"], peak_allocated_gib=record["peak_allocated_bytes"]/2**30,
            **{k: record["metrics"][k] for k in METRICS})
        row["relative_error_score"] = statistics.mean(row[k]/control["metrics"][k] for k in METRICS[:2])
        row["harmful_vs_zero"] = any(row[k] > 1.01*control["metrics"][k] for k in METRICS[:2])
        rows.append(row)
    assert len(rows) == 39
    groups = []
    for scene in controls:
        for condition in CONDITIONS:
            group = [r for r in rows if r["scene"] == scene and r["condition"] == condition]
            assert len(group) == 3 and {r["seed"] for r in group} == {42, 123, 2026}
            groups.append(dict(scene=scene, condition=condition, n=3,
                metrics={k: summarize([r[k] for r in group]) for k in METRICS + ["relative_error_score", "seconds"]},
                harmful_executions=sum(r["harmful_vs_zero"] for r in group)))
    contrasts = []
    for scene in controls:
        paired=[]
        for seed in [42,123,2026]:
            cell={r["condition"]:r["relative_error_score"] for r in rows if r["scene"]==scene and r["seed"]==seed and r["steps"]}
            assert set(cell)==set(CONDITIONS)
            paired.append(dict(seed=seed, pose_minus_visual=cell["pose_only"]-cell["visual"],
                intrinsics_minus_visual=cell["intrinsics_only"]-cell["visual"],
                both_minus_pose=cell["both"]-cell["pose_only"],both_minus_intrinsics=cell["both"]-cell["intrinsics_only"],
                interaction=cell["both"]-cell["pose_only"]-cell["intrinsics_only"]+cell["visual"]))
        contrasts.append(dict(scene=scene, paired=paired, summary={k:summarize([r[k] for r in paired]) for k in paired[0] if k!="seed"}))
    result = dict(contrasts=contrasts, study_id=plan["study_id"], status="complete", jobs=39, adaptation_updates=1440,
        code_commit=next(iter(records.values()))["research_commit"], manifest_sha256=digest,
        baselines={s: {k: r["metrics"][k] for k in METRICS} for s, r in controls.items()},
        groups=groups, limitations=["three inspected sequences", "three optimizer seeds with unresolved execution variability",
        "reference poses and unchanged nominal intrinsics; calibration caveats separately audited", "aligned shape, not absolute metric scale", "development screen, not generalization proof"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (args.output/"per_run.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(status="complete", jobs=len(rows), groups=len(groups), code_commit=result["code_commit"])))


if __name__ == "__main__":
    main()
