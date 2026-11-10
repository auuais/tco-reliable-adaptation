"""Four fixed local process repeats, separate from the frozen cloud v6 study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/attention_probe_v6_local"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    commands=[]
    for policy in ["default", "math"]:
        for repeat in range(2):
            commands.append([sys.executable, "-u", "scripts/run_attention_probe.py", "--attention-policy", policy,
                "--run-id", f"attention6_local_{policy}_r{repeat}", "--scene", "heads/seq-01",
                "--frames", "0", "200", "--size", "224", "--steps", "2", "--seed", "42",
                "--dataset-cache", str(ROOT/"environment/validation_dataset_cache.npy"), "--skip-evaluation"])
    sources={name:hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() for name in
             ["run_attention_probe.py", "run_attention_probe_batch.py", "run_numerical_probe.py", "run_baseline.py"]}
    (OUT/"plan.json").write_text(json.dumps(dict(commands=commands, sources=sources,
        scope="Exploratory local operator diagnosis, four fresh processes, no rendering, no reference scoring",
        decision="Compare within-policy first clipped gradients, RNG and two-step predictions; do not compare accuracy or claim operator causality from backend association"), indent=2)+"\n")
    env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONHASHSEED="42")
    for index, command in enumerate(commands):
        print("Attention diagnostic", index+1, "/4", command[5:8], flush=True)
        with (OUT/f"job{index}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
    print("All four local attention diagnostics completed", flush=True)


if __name__ == "__main__":
    main()
