"""Run only independent math/rendering cases after unsupported flash-only case.

Preserves the failed case and original five-case plan; does not retry or replace
the flash-only experiment. This continuation has no paid-compute component.
"""
import json
import os
from pathlib import Path
import subprocess
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/backend_isolation_v6_local"


def main():
    plan=json.loads((OUT/"plan.json").read_text())
    for name,digest in plan["sources"].items():assert sha(ROOT/"scripts"/name)==digest
    failed=json.loads((ROOT/"runs/backend6_local_flash_renderoff_r0/run.json").read_text())
    assert failed["status"]=="failed" and "No available kernel" in failed["error"]
    assert not (OUT/"continuation.json").exists()
    dump(OUT/"continuation.json",dict(original_plan_sha256=sha(OUT/"plan.json"),
        failed_case="flash_renderoff_r0",unattempted_case="flash_renderoff_r1",
        reason="Flash-only policy cannot execute all model attention calls; preserve failure, skip its redundant repeat",
        independent_cases=[3,4],continuation_source_sha256=sha(Path(__file__))))
    env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42")
    for index in [3,4]:
        print("Independent original-plan math/rendering case",index,flush=True)
        with (OUT/f"job{index}.log").open("x") as stream:
            subprocess.run(plan["commands"][index],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)


if __name__=="__main__":main()
