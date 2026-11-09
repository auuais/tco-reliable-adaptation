"""Validate local offload feasibility without claiming adapted-output equivalence."""
import hashlib
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
manifest = ROOT/"manifests/local_memory_v2.json"
plan = json.loads(manifest.read_text())
records = {j["run_id"]: json.loads((ROOT/"runs"/j["run_id"]/"run.json").read_text()) for j in plan["jobs"]}
control = records["memory2_control_s0"]
rows = []
for job in plan["jobs"]:
    r = records[job["run_id"]]
    assert r["status"] == "complete" and not r["research_dirty"] and not r["base_parameters_modified"]
    assert r["arguments"]["steps"] == job["steps"]
    assert r["dataset_cache_sha256"] == plan["dataset_cache_sha256"]
    for key in ["research_commit", "input_sha256", "reference_sha256", "upstream_method_sha256",
                "runner_sha256", "versions", "checkpoint_download", "image_shape", "gpu", "evaluation_tensor_sha256"]:
        assert r[key] == control[key], key
    cfg = dict(control["configuration"]); cfg["tco_steps"] = job["steps"]
    assert cfg == r["configuration"]
    pred = np.load(ROOT/"runs"/job["run_id"]/"predictions.npy", mmap_mode="r")
    assert pred.shape == (5, 392, 518, 3) and np.isfinite(pred).all()
    if job["steps"]:
        gradients = json.loads((ROOT/"runs"/job["run_id"]/"gradients.json").read_text())
        assert len(gradients) == 1 and gradients[0]["finite"] and gradients[0]["gradient_norm_after_upstream_clipping"] > 0
    if "offload" in job["run_id"]:
        assert r["memory_optimization"]["wrapper_sha256"] == hashlib.sha256((ROOT/"scripts/run_memory_probe.py").read_bytes()).hexdigest()
    rows.append(dict(run_id=job["run_id"], steps=job["steps"], status=r["status"],
        peak_allocated_gib=r["peak_allocated_bytes"]/2**30,
        peak_reserved_gib=r["peak_reserved_bytes"]/2**30,
        inference_and_adaptation_seconds=r["inference_and_adaptation_seconds"],
        model_load_seconds=r["model_load_seconds"], base_parameter_count=r["base_parameter_count"]))
a = np.load(ROOT/"runs/memory2_control_s0/predictions.npy", mmap_mode="r")
b = np.load(ROOT/"runs/memory2_offload_s0/predictions.npy", mmap_mode="r")
assert np.array_equal(a, b)
result = dict(study_id=plan["study_id"], status="complete", gpu=control["gpu"],
    code_commit=control["research_commit"], manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    zero_step_pointmaps_bitwise_equal=True, zero_step_max_absolute_difference=0.0,
    runs=rows, limitations=["one scene and one execution per condition", "one adaptation step only; 40 steps not tested",
        "no adapted accuracy or adapted-output equivalence claim", "shared Windows desktop GPU; timings are not a controlled speed benchmark",
        "memory allocation is not a direct measure of physical VRAM residency"],
    earlier_attempt=dict(study_id="local_memory_v1", run_id="memory1_control_s0", status="failed_before_model_load",
        reason="Concurrent Office preprocessing exposed incomplete registered depth; preserved and replaced by a frozen metadata-cache protocol"))
dest = ROOT/"results/local_memory_v2"
dest.mkdir(parents=True, exist_ok=True)
(dest/"summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result, indent=2))
