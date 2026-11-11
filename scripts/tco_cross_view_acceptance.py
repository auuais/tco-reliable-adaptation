"""Replay an outcome-blind update acceptance rule on the 37 V6 TCO predictions."""
import argparse
import hashlib
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.spatial import cKDTree

from dense_rgb_pixel_check import similarity
from evaluate_fixed_matches import rank_correlation
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, ROOT, SCENES
from repeatability_capture import array_digest
from score_saved_geometry import read

HISTORY = ROOT/"results/camera_shape_v10/summary.json"
DESIGN = ROOT/"docs/TCO_CROSS_VIEW_ACCEPTANCE.md"
RESULT = PUBLIC/"tco_cross_view_acceptance.json"


def load_map(folder, row):
    path = folder/row["run_id"]/"predictions.npy"
    if sha(path)!=row["final_points_sha256"]: raise ValueError("Changed V6 output")
    data = np.load(path,allow_pickle=False)
    if data.ndim!=4 or data.shape[0]!=5 or data.shape[-1]!=3: raise ValueError("Unexpected V6 shape")
    return data[:,::4,::4].reshape(5,-1,3).astype(float)


def support(points, trees, radius):
    rows = []
    for i,view in enumerate(points):
        distance = np.full((len(view),len(trees)),np.inf)
        for j,tree in enumerate(trees):
            if i!=j:
                distance[:,j] = tree.query(view,distance_upper_bound=radius,workers=2)[0]
        ids = np.argsort(distance,axis=1)[:,:2]
        keep = np.isfinite(np.take_along_axis(distance,ids,axis=1)[:,1])
        rows.append((np.flatnonzero(keep),ids[keep]))
    return rows


def distance(points, trees, frozen):
    total = 0.
    count = 0
    for query,(ids,sources) in zip(points,frozen):
        for j,tree in enumerate(trees):
            selected = ids[np.any(sources==j,axis=1)]
            if len(selected):
                total += tree.query(query[selected],workers=2)[0].sum()
                count += len(selected)
    return float(total/count) if count else None


def measure(folder):
    rows = read(HISTORY)["rows"]
    observations = {}
    for scene in SCENES:
        records = [r for r in rows if r["scene"]==scene]
        zero, = [r for r in records if r["condition"]=="zero"]
        base = load_map(folder,zero)
        cap = read(folder/zero["run_id"]/"capture.json")["records"]["initial_predictions"]
        path = folder/zero["run_id"]/cap["file"]
        if sha(path)!=cap["file_sha256"]: raise ValueError("Initial archive differs")
        with np.load(path) as f:
            arrays = dict(f)
        if array_digest(arrays)!=zero["initial_heads_sha256"]: raise ValueError("Initial tensors differ")
        gauge = float(np.median(arrays["depth"]))
        trees = [cKDTree(p) for p in base]
        frozen = support(base,trees,.02*gauge)
        coverage = sum(len(ids) for ids,_ in frozen)/(base.shape[0]*base.shape[1])
        initial = distance(base,trees,frozen)
        for r in records:
            if r["condition"]=="zero": continue
            if r["initial_heads_sha256"]!=zero["initial_heads_sha256"]: raise ValueError("Incomparable initial heads")
            points = load_map(folder,r)
            scale,rot,trans = similarity(points.reshape(-1,3),base.reshape(-1,3))
            aligned = scale*points@rot+trans
            internal = distance(aligned,[cKDTree(p) for p in aligned],frozen)
            fixed = distance(aligned,trees,frozen)
            observations[r["run_id"]] = dict(support=coverage,initial=initial/gauge if initial is not None else None,
                    internal=internal/gauge if internal is not None else None,fixed=fixed/gauge if fixed is not None else None,
                    scene_similarity_scale=float(scale))
        print(scene,coverage,flush=True)
    return observations


def summarize(observations):
    records = [r for r in read(HISTORY)["rows"] if r["condition"]!="zero"]
    if set(observations)!={r["run_id"] for r in records} or len(records)!=37:
        raise ValueError("Missing adapted run")
    def group(rows):
        accepted, rejected, internal, fixed, scores, outcomes = [],[],[],[],[],[]
        for r in rows:
            obs = observations[r["run_id"]]
            eligible = obs["support"]>=.05 and obs["initial"] is not None and obs["initial"]>0
            use = eligible and obs["internal"]<=.99*obs["initial"]
            (accepted if use else rejected).append(r)
            outcomes.append(r["v6_primary_score"] if use else 1.)
            if obs["initial"]:
                internal.append(obs["internal"]/obs["initial"]-1)
                fixed.append(obs["fixed"]/obs["initial"]-1)
                scores.append(r["v6_primary_score"])
        def counts(group):
            return dict(total=len(group),harmful=sum(r["v6_harmful"] for r in group),
                        improved=sum(r["v6_primary_score"]<1 for r in group))
        return dict(accepted=counts(accepted),rejected=counts(rejected),selected_score=mean(outcomes),
                    all_adapted_score=mean(r["v6_primary_score"] for r in rows),
                    internal_rank=rank_correlation(internal,scores),fixed_rank=rank_correlation(fixed,scores))
    return {name:dict(overall=group(rows),scenes={s:group([r for r in rows if r["scene"]==s]) for s in SCENES})
            for name,rows in (("all",records),("repeat_zero",[r for r in records if r["repeat"]==0]))}


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    p.add_argument("--evidence",type=Path)
    args = p.parse_args()
    deps = dict(history=sha(HISTORY),protocol=hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if args.action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        obs = measure(args.evidence/"runs")
        dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
    data = read(RESULT)
    replay = summarize(data["observed"])
    if deps!=data["inputs"]: raise ValueError("Provenance differs")
    for name in replay:
        for unit in ["overall",*SCENES]:
            a = replay[name]["overall"] if unit=="overall" else replay[name]["scenes"][unit]
            stored = data["results"][name]
            b = stored["overall"] if unit=="overall" else stored["scenes"][unit]
            for k,v in a.items():
                if isinstance(v,float):
                    np.testing.assert_allclose(v,b[k],rtol=1e-12,atol=1e-12)
                elif v!=b[k]:
                    raise ValueError("Acceptance counts differ")
    print(replay["all"]["overall"])
