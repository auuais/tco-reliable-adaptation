"""Compare all four local diagnostic executions without selecting a winner."""
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    comparisons=[]
    for policy in ["default","math"]:
        ids=[f"attention6_local_{policy}_r{r}" for r in [0,1]]
        dirs=[ROOT/"runs"/key for key in ids]
        records=[json.loads((path/"run.json").read_text()) for path in dirs]
        assert all(r["status"]=="complete" and r["arguments"]["steps"]==2 for r in records)
        probes=[json.loads((path/"numerical_probe.json").read_text()) for path in dirs]
        losses=[json.loads((path/"losses.json").read_text()) for path in dirs]
        gradients=[dict(np.load(path/"first_gradients.npz",allow_pickle=False)) for path in dirs]
        assert gradients[0].keys()==gradients[1].keys()
        x,y=[np.concatenate([arrays[k].ravel().astype(np.float64) for k in sorted(arrays)]) for arrays in gradients]
        predictions=[np.load(path/"predictions.npy",allow_pickle=False) for path in dirs]
        a,b=probes
        comparisons.append(dict(policy=policy,run_ids=ids,same_initial_adapters=a["parameters"]==b["parameters"],
            same_rng_before_setup=a["rng_before_setup"]==b["rng_before_setup"],same_rng_after_setup=a["rng_after_setup"]==b["rng_after_setup"],
            same_first_scalar_losses=losses[0][0]==losses[1][0],same_step_rng=[p["rng"]==q["rng"] for p,q in zip(a["steps"],b["steps"])],
            identical_first_clipped_gradients=bool(np.array_equal(x,y)),first_gradient_relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
            first_gradient_cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
            identical_final_pointmaps=bool(np.array_equal(*predictions)),
            maximum_pointmap_difference=float(np.max(np.abs(predictions[0]-predictions[1]))),
            peak_allocated_GiB=[r["peak_allocated_bytes"]/2**30 for r in records],
            operator_settings=[json.loads((p/"attention_probe.json").read_text()) for p in dirs]))
    dump(ROOT/"results/attention_probe_v6_local/summary.json",dict(comparisons=comparisons,
        plan_sha256=sha(ROOT/"reports/attention_probe_v6_local/plan.json"),gpu=records[0]["gpu"],
        scope="Two 168x224 views, Heads, seed 42, two updates, rendering disabled, local RTX 3080 only",
        limitation="Exploratory numerical diagnosis, not full-resolution reproduction or accuracy/performance evidence. Backend association does not prove the responsible kernel."))
    print(json.dumps(comparisons,indent=2))


if __name__=="__main__":main()
