"""Test whether anchor-order dispersion adds information about correction benefit."""
import argparse
import hashlib
from statistics import mean

import numpy as np

import anchor_ensemble_surface as ensemble
import motion_reliability_weights as base
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

RESULT = PUBLIC/"anchor_disagreement_benefit.json"
DESIGN = ROOT/"docs/ANCHOR_DISAGREEMENT_BENEFIT.md"
RULES = ("original","size","disagreement","both")


def dispersion(key,depth):
    fields = []
    captures = read(ensemble.capture.RESULT)["rows"]
    offsets = read(ensemble.SEAL)["rows"][key]["log_offsets"]
    for i,offset in enumerate(offsets):
        name = f"{key}_{i}"
        path = ensemble.capture.DATA/(name+".npz")
        if sha(path)!=captures[name]["sha256"]: raise ValueError("Changed capture")
        with np.load(path) as f:
            fields.append(np.log(f["depth"][...,0].astype(float)/depth)-offset)
    return np.median(np.abs(fields-np.median(fields,axis=0)),axis=0)


def cells(update,masks,spread):
    for name,mask in zip(base.CELLS,masks):
        size_cut,spread_cut = (np.median(np.abs(update[mask])),np.median(spread[mask])) if mask.any() else (0.,0.)
        large = np.abs(update)>size_cut
        dispersed = spread>spread_cut
        for m in range(2):
            for d in range(2):
                yield f"{name}/{m}/{d}",mask & (large==m) & (dispersed==d)


def measure():
    targets = []
    archived = read(base.prior.ESTIMATES)["targets"]
    index = read(base.INDEX)["frames"]
    for scene in SCENES:
        slug = scene.split("/")[0]
        key = slug+"_base"
        _,depth,_,_,_ = base.inputs(key,read(base.PARENT/"predictions.json")["rows"],read(base.PLAN))
        spread = dispersion(key,depth)
        scale = read(base.prior.PRIOR/"source_transplant.json")["batches"][key]["scale_to_metres"]
        for i,frame in enumerate(FRAMES):
            path = base.prior.MAPS/(base.prior.tag(scene,"short",frame)+".npz")
            ref = RAW/"reference"/slug/f"{frame:06d}.npz"
            if sha(path)!=archived[path.stem]["archive_sha256"] or sha(ref)!=index[scene][str(frame)]["reference_sha256"]:
                raise ValueError("Input differs")
            with np.load(path) as f:
                update,masks = base.reliability_cells(depth[i],f["frozen_any_source"].astype(float),f["frozen_two_sided"])
            bins = list(cells(update,masks,spread[i]))
            with np.load(ref) as f:
                valid = f["valid"]
                error = np.zeros_like(depth[i])
                error[valid] = np.log(scale*depth[i][valid]/f["depth"][valid])
            stats = {}
            for label,mask in bins:
                selected = valid & mask
                u,e = update[selected],error[selected]
                stats[label] = dict(n=len(u),A=float(u@u),B=float(e@u),helped=int(np.sum((2*e+u)*u<0)))
            targets.append(dict(scene=scene,frame=frame,C=float(error[valid]@error[valid]),cells=stats))
    return targets


def group(label,rule):
    source,size,spread = label.split("/")
    return (source,size if rule in ("size","both") else "",spread if rule in ("disagreement","both") else "")


def fit(train,rule):
    labels = list(train[0]["cells"])
    weights = {}
    for label in labels:
        pool = [k for k in labels if group(k,rule)==group(label,rule)]
        parent = [k for k in labels if group(k,"original")==group(label,"original")]
        weights[label] = base.strength(train,pool,base.strength(train,parent,base.strength(train,labels)))
    return weights


def summarize(rows):
    old = read(base.RESULT)
    if [(r["scene"],r["frame"]) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError("Wrong target set")
    folds,tests = {},[]
    for scene in SCENES:
        train = [r for r in rows if r["scene"]!=scene]
        folds[scene] = {rule:fit(train,rule) for rule in RULES}
    for row,record,prev in zip(rows,old["targets"],old["results"]["targets"]):
        np.testing.assert_allclose(row["C"],record["C"])
        for parent in base.CELLS:
            children = [v for k,v in row["cells"].items() if k.split("/")[0]==parent]
            for field in ("A","B","n"):
                np.testing.assert_allclose(sum(v[field] for v in children),record["cells"][parent][field],rtol=1e-9,atol=1e-9)
        risk = {r:base.risk(row,w) for r,w in folds[row["scene"]].items()}
        np.testing.assert_allclose(risk["original"],prev["risk"]["weighted"],rtol=1e-9)
        risk["target_oracle"] = base.risk(row,fit([row],"both"))
        if risk["target_oracle"]>min(risk.values())+1e-10: raise ValueError("Invalid bound")
        tests.append(dict(scene=row["scene"],frame=row["frame"],risk=risk))
    def means(test):
        return {r:mean(v["risk"][r] for v in test) for r in (*RULES,"target_oracle")}
    overall = means(tests)
    scenes = {s:means([r for r in tests if r["scene"]==s]) for s in SCENES}
    comparator = min(("size","original"),key=overall.get)
    gain = 1-overall["both"]/overall[comparator]
    rates = {}
    for d in ("0","1"):
        subset = [v for r in rows for k,v in r["cells"].items() if k.endswith("/"+d)]
        rates[d] = sum(v["helped"] for v in subset)/sum(v["n"] for v in subset)
    return dict(folds=folds,tests=tests,scenes=scenes,overall=overall,comparator=comparator,gain=gain,
                full_step_benefit_rate=rates,consider_surface=gain>=.02 and all(v["both"]<v[comparator] for v in scenes.values()))


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps = [sha(p) for p in (base.RESULT,base.prior.ESTIMATES,base.INDEX,base.PLAN,ensemble.SEAL,ensemble.capture.RESULT,base.prior.PRIOR/"source_transplant.json")]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows = measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    report = read(RESULT)
    if deps!=report["inputs"] or report["results"]!=summarize(report["rows"]): raise ValueError("Replay failed")
    print(report["results"]["overall"],report["results"]["gain"],report["results"]["consider_surface"])
