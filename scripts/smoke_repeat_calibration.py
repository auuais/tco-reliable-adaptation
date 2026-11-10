"""Exercise real adaptation, calibration isolation, and state capture on RTX 3080."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cases = [("zero_"+k,0,k,1) for k in ["released","axis_nominal","axis_registration"]]
cases += [("both_released",40,"released",1), ("both_axis_nominal",1,"axis_nominal",1),
          ("both_axis_registration",1,"axis_registration",1), ("visual_released",1,"released",0)]
records = []
for condition,steps,calibration,scale in cases:
    run_id = "repeat6_local_smoke_"+condition
    subprocess.run([sys.executable,"-u","scripts/run_repeat_calibration.py","--run-id",run_id,
        "--study-id","repeat_calibration_v6_local_smoke","--scene","heads/seq-01","--size","224",
        "--frames","0","200","--steps",str(steps),"--learning-rate",".0003",
        "--calibration",calibration,"--pose-weight-scale",str(scale),"--intrinsics-weight-scale",str(scale),
        "--dataset-cache",str(ROOT/"environment/validation_dataset_cache.npy")],cwd=ROOT,check=True)
    directory=ROOT/"runs"/run_id
    record=json.loads((directory/"run.json").read_text())
    capture=json.loads((directory/"capture.json").read_text())
    assert record["status"]=="complete" and not record["base_parameters_modified"]
    assert "initial_predictions" in capture["records"]
    if steps:
        assert capture["records"]["initial_adapters"]["tensor_sha256"] != capture["records"]["final_adapters"]["tensor_sha256"]
        assert len(json.loads((directory/"losses.json").read_text()))==steps
    if records:
        assert record["evaluation_tensor_sha256"]==records[0]["evaluation_tensor_sha256"]
    records.append(record)
zero=[np.load(ROOT/"runs"/r["run_id"]/"predictions.npy") for r in records[:3]]
for prediction in zero[1:]: np.testing.assert_array_equal(prediction,zero[0])
previous=ROOT/"runs/prior5_local_smoke_zero_both/predictions.npy"
comparison=None
if previous.exists():
    comparison=bool(np.array_equal(zero[0],np.load(previous)))
    assert comparison, "Instrumentation changed the zero-step smoke prediction"
out=ROOT/"results/repeat_calibration_v6_preflight/smoke.json"
assert not out.exists()
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(status="passed",scope="Two 168x224 views; integration only",runs=len(records),
    updates=sum(r["arguments"]["steps"] for r in records),zero_step_calibration_invariant=True,
    original_v5_zero_step_bitwise_equal=comparison,reference_unchanged=True,
    forty_step_adaptation_completed=True,run_ids=[r["run_id"] for r in records],gpu=records[0]["gpu"],
    max_allocated_GiB=max(r["peak_allocated_bytes"] for r in records)/2**30),indent=2)+"\n")
print(out.read_text())
