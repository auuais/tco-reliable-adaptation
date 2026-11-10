"""Five fixed local follow-ups after the math/no-rendering repeat result."""
import json
import os
from pathlib import Path
import subprocess
import sys
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/backend_isolation_v6_local"


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    cases=[("default","off",0,True),("flash","off",0,False),("flash","off",1,False),
           ("math","on",0,False),("math","on",1,False)]
    commands=[]
    for backend,rendering,repeat,profile in cases:
        command=[sys.executable,"-u","scripts/run_backend_isolation.py","--backend",backend,"--rendering",rendering,
            "--run-id",f"backend6_local_{backend}_render{rendering}_r{repeat}","--scene","heads/seq-01","--frames","0","200",
            "--size","224","--steps","2","--seed","42","--dataset-cache",str(ROOT/"environment/validation_dataset_cache.npy"),"--skip-evaluation"]
        if profile:command.append("--profile-operators")
        commands.append(command)
    dump(OUT/"plan.json",dict(commands=commands,sources={name:sha(ROOT/"scripts"/name) for name in
        ["run_backend_isolation.py","run_backend_isolation_batch.py","run_numerical_probe.py","run_baseline.py"]},
        decision="Observe default attention operator names; compare flash without rendering and math with rendering within their two fresh-process repeats; no accuracy claim"))
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42")
    for index,command in enumerate(commands):
        print("Backend diagnostic",index+1,"/5",flush=True)
        with (OUT/f"job{index}.log").open("x") as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
    print("All five backend diagnostics completed",flush=True)


if __name__=="__main__":main()
