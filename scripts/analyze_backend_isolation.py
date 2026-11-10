"""Account for completed, failed and unattempted local backend cases."""
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan_path=ROOT/"reports/backend_isolation_v6_local/plan.json"
    plan=json.loads(plan_path.read_text());cases=[]
    for command in plan["commands"]:
        key=command[command.index("--run-id")+1];path=ROOT/"runs"/key
        if not (path/"run.json").exists():cases.append(dict(run_id=key,status="not_attempted"));continue
        record=json.loads((path/"run.json").read_text())
        cases.append(dict(run_id=key,status=record["status"],error=record.get("error"),
            settings=json.loads((path/"backend_isolation.json").read_text())))
    ids=[f"backend6_local_math_renderon_r{r}" for r in [0,1]]
    paths=[ROOT/"runs"/key for key in ids]
    records=[json.loads((p/"run.json").read_text()) for p in paths]
    assert all(r["status"]=="complete" for r in records)
    probes=[json.loads((p/"numerical_probe.json").read_text()) for p in paths]
    arrays=[dict(np.load(p/"first_gradients.npz",allow_pickle=False)) for p in paths]
    x,y=[np.concatenate([a[k].astype(np.float64).ravel() for k in sorted(a)]) for a in arrays]
    predictions=[np.load(p/"predictions.npy",allow_pickle=False) for p in paths]
    losses=[json.loads((p/"losses.json").read_text()) for p in paths]
    a,b=probes
    comparison=dict(run_ids=ids,same_initial_adapters=a["parameters"]==b["parameters"],
        same_setup_rng=a["rng_before_setup"]==b["rng_before_setup"] and a["rng_after_setup"]==b["rng_after_setup"],
        same_step_rng=[p["rng"]==q["rng"] for p,q in zip(a["steps"],b["steps"])],same_first_scalar_losses=losses[0][0]==losses[1][0],
        identical_first_clipped_gradients=bool(np.array_equal(x,y)),first_gradient_relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
        first_gradient_cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
        identical_final_pointmaps=bool(np.array_equal(*predictions)),maximum_pointmap_difference=float(np.max(np.abs(predictions[0]-predictions[1]))))
    out=dict(plan_sha256=sha(plan_path),planned_cases=5,completed=sum(c["status"]=="complete" for c in cases),
        failed=sum(c["status"]=="failed" for c in cases),unattempted=sum(c["status"]=="not_attempted" for c in cases),
        cases=cases,math_with_rendering=comparison,
        limitation="Only two updates/two small views on RTX 3080. Operator trace records default-path names, not causality. Flash-only fails on an unsupported call; no equivalent-method or accuracy/speed claim.")
    dump(ROOT/"results/attention_probe_v6_local/backend_followup.json",out)
    print(json.dumps(dict(completed=out["completed"],failed=out["failed"],unattempted=out["unattempted"],math_with_rendering=comparison),indent=2))


if __name__=="__main__":main()
