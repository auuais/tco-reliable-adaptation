"""Four local integration cases; not a development accuracy experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/pose_precision_v7_preflight"


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    cases=[(mode,steps) for steps in [0,2] for mode in ["released","float32"]]
    commands=[]
    for mode,steps in cases:
        commands.append([sys.executable,"-u","scripts/run_pose_precision.py","--pose-precision",mode,"--study-id","pose_precision_v7_preflight",
            "--run-id",f"precision7_local_{mode}_s{steps}","--scene","heads/seq-01","--frames","0","200","--size","224",
            "--steps",str(steps),"--seed","42","--learning-rate",".0003","--dataset-cache",str(ROOT/"environment/validation_dataset_cache.npy"),"--skip-evaluation"])
    dump(OUT/"plan.json",dict(commands=commands,sources={name:sha(ROOT/"scripts"/name) for name in
        ["run_pose_precision.py","preflight_pose_precision.py","run_memory_probe.py","run_experiment.py"]},
        purpose="Check zero-step invariance, matched first pose inputs, intervention dtypes, finite updates and base-weight preservation; no accuracy claim"))
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42")
    for index,command in enumerate(commands):
        print("Pose precision preflight",index+1,"/4",flush=True)
        with (OUT/f"job{index}.log").open("x") as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
    records=[]
    for mode,steps in cases:
        path=ROOT/"runs"/f"precision7_local_{mode}_s{steps}"
        record=json.loads((path/"run.json").read_text())
        assert record["status"]=="complete" and not record["base_parameters_modified"] and not record["research_dirty"]
        if steps:
            history=json.loads((path/"pose_precision.json").read_text())
            assert len(history["calls"])==2 and history["mode"]==mode
            assert all(c["outer_autocast_enabled"] and c["input_predicted_dtype"]=="torch.float32" for c in history["calls"])
        records.append(record)
    zeros=[np.load(ROOT/"runs"/f"precision7_local_{mode}_s0/predictions.npy",allow_pickle=False) for mode in ["released","float32"]]
    np.testing.assert_array_equal(*zeros)
    arrays=[dict(np.load(ROOT/"runs"/f"precision7_local_{mode}_s2/pose_inputs_step0.npz",allow_pickle=False)) for mode in ["released","float32"]]
    for key in arrays[0]:np.testing.assert_array_equal(arrays[0][key],arrays[1][key])
    first=[json.loads((ROOT/"runs"/f"precision7_local_{mode}_s2/pose_precision.json").read_text())["calls"][0] for mode in ["released","float32"]]
    dump(ROOT/"results/pose_precision_v7_preflight/summary.json",dict(status="passed",jobs=4,updates=4,
        zero_step_bitwise_equal=True,first_pose_inputs_bitwise_equal=True,first_pose_loss_calls=first,
        plan_sha256=sha(OUT/"plan.json"),execution_commit=records[0]["research_commit"],
        scope="Two small views; local plumbing and precision audit only; no accuracy or repeatability benefit demonstrated"))
    print(json.dumps(dict(status="passed",first_pose_loss_calls=first),indent=2))


if __name__=="__main__":main()
