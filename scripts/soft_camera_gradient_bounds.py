"""Evaluate partial camera omission from the published TCO output-gradient arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation_stability import dump, sha


ROOT = Path(__file__).resolve().parents[1]
STUDY = "soft_camera_gradient_bounds"
PLAN = ROOT / "manifests" / (STUDY + ".json")
OUT = ROOT / "results" / STUDY / "summary.json"
PARENT = ROOT / "results/masked_camera_gradients"
FAMILIES = ("detected", "random", "global")
ORACLES = ("oracle_detected", "oracle_actual", "oracle_best_view")
POLICIES = ("unchanged", *FAMILIES, *ORACLES)
CLEAN_CAP = 0.05


def protocol():
    sources = [Path(__file__), PARENT / "summary.json", PARENT / "gradient_traces.zip",
               ROOT / "manifests/masked_camera_gradients.json"]
    return dict(study=STUDY, sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                data="Secondary development analysis of the already published 183 cases and frozen alert decisions; 180 injected faults, three clean windows, three scenes. No new independent observations.",
                loss="A constant convex weight on the original and one view-excluded camera loss gives the same convex mixture of their derivatives. Error is the Euclidean distance to the clean unmasked derivative, normalized by its norm. Visual loss and target parameters are outside this audit.",
                fitted="Hold out one development scene. Fit one scalar per policy on the two remaining scenes: sum of normalized squared errors over corrupted cases, and a maximum normalized distortion of 0.05 on every training clean case and possible omission. Apply weights using only the signed-off alert/suspect at evaluation. Do not refit on the target scene. Random policy averages the errors of the five equiprobable view choices; global policy downweights all camera terms.",
                bounds="Compute the exact clamped line projection per case using its privileged clean gradient. Report detected view with alert; known faulty view with no alert restriction; best of all five views with no restriction. Clean ground truth never enters the deployed policies. The best-view envelope is a gain ceiling for selecting one whole-view omission and one mixing weight per case.",
                decision="Require detected policy at least 20 percent lower mean corrupted normalized error than unchanged, lower than both fitted controls, no scene worse, no fault profile more than 5 percent worse and maximum clean distortion at most 0.05. A pass motivates an independent development window only. If even the best-view envelope has gain below 20 percent, set aside every one-view scalar mixture on this grid.",
                methods=list(POLICIES),clean_cap=CLEAN_CAP,cloud_usd=0)


def load_cases():
    parent = json.loads((PARENT / "summary.json").read_text())
    with np.load(PARENT / "gradient_traces.zip",allow_pickle=False) as src:
        observed = src["variants"].astype(np.float64)
        references = dict(zip(src["scenes"].tolist(),src["clean"].astype(np.float64)))
    if observed.shape != (183,6,5,9) or len(parent["rows"]) != len(observed) or len(references)!=3:
        raise ValueError("Unexpected gradient grid")
    cases = []
    for pos,(row,g) in enumerate(zip(parent["rows"],observed)):
        clean = references[row["scene"]].ravel()
        scale = np.linalg.norm(clean)
        full = g[0].ravel()
        if scale<=1e-8 or not np.isfinite(g).all():
            raise ValueError("Invalid gradient")
        b = (full-clean) / scale
        directions = (g[1:].reshape(5,-1)-full) / scale
        if not np.isclose(np.linalg.norm(b),row["errors"]["unchanged"]["whole"],atol=1e-12,rtol=1e-12):
            raise ValueError("Case order or parent reference changed")
        cases.append(dict(index=pos,scene=row["scene"],fault=row["fault"],view=row["view"],seed=row["seed"],
                          alert=row["alert"],suspect=row["suspect"],b=b,d=directions,global_d=-full/scale))
    return cases


def policy_directions(case, method):
    if not case["alert"]:
        return np.zeros((1,case["b"].size))
    if method=="global":
        return case["global_d"][None,:]
    if method=="random":
        return case["d"]
    return case["d"][[case["suspect"]]]


def fit(cases,method):
    dot, curvature, cap = 0.,0.,1.
    for case in cases:
        directions = policy_directions(case,method)
        if case["fault"]=="clean":
            if np.any(case["b"]):
                raise ValueError("Clean reference should match the full gradient")
            largest = np.linalg.norm(directions,axis=1).max()
            if largest>0:
                cap = min(cap, CLEAN_CAP/largest)
        else:
            dot += float(np.mean(directions @ case["b"]))
            curvature += float(np.mean(np.sum(directions**2,axis=1)))
    return float(np.clip(-dot/curvature,0.,cap)) if curvature>0 else 0.


def best_mixture(b,d):
    length = np.dot(d,d)
    weight = float(np.clip(-np.dot(b,d)/length,0.,1.)) if length>0 else 0.
    end = b + weight*d
    # First-order optimality against both feasible endpoints.
    derivative = np.dot(d,end)
    if min(-weight*derivative, (1-weight)*derivative)<-1e-10*(1+length+np.dot(b,b)):
        raise ValueError("Projection failed its optimality check")
    return float(np.linalg.norm(end)),weight


def evaluate():
    cases = load_cases()
    scenes = sorted({c["scene"] for c in cases})
    fitted = {s:{m:fit([c for c in cases if c["scene"]!=s],m) for m in FAMILIES} for s in scenes}
    rows = []
    for case in cases:
        b = case["b"]
        values = dict(unchanged=float(np.linalg.norm(b)))
        for method,weight in fitted[case["scene"]].items():
            values[method] = float(np.mean(np.linalg.norm(b + weight*policy_directions(case,method),axis=1)))
        candidates = [best_mixture(b,d) for d in case["d"]]
        minimum = min(x[0] for x in candidates)
        best_view = next(i for i,x in enumerate(candidates) if x[0] <= minimum + 1e-12)
        default = (values["unchanged"],0.)
        selected = dict(oracle_detected=candidates[case["suspect"]] if case["alert"] else default,
                        oracle_actual=candidates[case["view"]] if case["view"] is not None else default,
                        oracle_best_view=(minimum,candidates[best_view][1]))
        values.update({m:v[0] for m,v in selected.items()})
        if min(values[m]-values["oracle_best_view"] for m in ("detected","oracle_actual","oracle_detected")) < -1e-10:
            raise ValueError("View-selection envelope violated")
        rows.append(dict(**{k:case[k] for k in ("index","scene","fault","view","seed","alert","suspect")},
                         errors=values,oracle_weights={m:v[1] for m,v in selected.items()},oracle_best_view=best_view))

    faulty = [r for r in rows if r["fault"]!="clean"]
    clean = [r for r in rows if r["fault"]=="clean"]
    def means(subset):
        return {m:float(np.mean([r["errors"][m] for r in subset])) for m in POLICIES}
    overall = means(faulty)
    groups = {k:{v:means([r for r in faulty if r[k]==v]) for v in sorted({r[k] for r in faulty})}
              for k in ("scene","fault")}
    gain = {k:1-v/overall["unchanged"] for k,v in overall.items()}
    worst = max(r["errors"]["detected"] for r in clean)
    checks = dict(gain=gain["detected"]>=.20,
                  controls=all(overall["detected"]<overall[k] for k in ("random","global")),
                  scenes=all(v["detected"]<=v["unchanged"] for v in groups["scene"].values()),
                  profiles=all(v["detected"]<=1.05*v["unchanged"] for v in groups["fault"].values()),
                  clean=worst<=CLEAN_CAP+1e-12)
    return dict(manifest_sha256=sha(PLAN),rows=rows,scene_excluded_weights=fitted,overall=overall,groups=groups,
                relative_gain=gain,worst_clean_detected=worst,checks=checks,
                motivates_fresh_development_window=all(checks.values()),
                best_view_gain_below_target=gain["oracle_best_view"]<.20,additional_cloud_usd=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","run","verify"))
    args = p.parse_args()
    if args.action=="freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN,protocol())
        return
    if json.loads(PLAN.read_text())!=protocol():
        raise ValueError("Frozen source or input changed")
    result=evaluate()
    if args.action=="run":
        if OUT.exists():
            raise FileExistsError(OUT)
        dump(OUT,result)
    else:
        from verify_masked_camera_gradients import compare
        stored=json.loads(OUT.read_text())
        for row,replay in zip(stored["rows"],result["rows"],strict=True):
            compare(row,replay)
        compare({k:v for k,v in stored.items() if k!="rows"},{k:v for k,v in result.items() if k!="rows"})
        print("Verified the 183 cases, cross-scene weights and analytic gradient-error bounds.")


if __name__=="__main__":
    main()
