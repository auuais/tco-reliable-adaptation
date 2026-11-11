"""Compare view-specific prior gains against equally small fixed and displaced gains."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import numpy as np

import anchor_prior_fusion as fusion
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

DATA = RAW/"anchor_prior_controls"
SEAL = PUBLIC/"anchor_prior_control_predictions.json"
RESULT = PUBLIC/"anchor_prior_control_scores.json"
DESIGN = ROOT/"docs/ANCHOR_PRIOR_GAIN_CONTROLS.md"
FIXED = dict(fixed02=.02,fixed05=.05,fixed10=.10,fixed20=.20)
ARMS = (*FIXED,"scene_mean","permuted")


def infer():
    records = {}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        slug = scene.split("/")[0]
        key = slug+"_base"
        source = read(fusion.model.SEAL)["rows"][key]
        original = read(fusion.SEAL)["rows"][key]
        fused = np.array(source["fused"])
        rot,c = fused[:,:3,:3],fusion.centers(fused)
        points = fusion.depth.cross.temporal.load_points(fusion.model.DATA,source["hashes"]["joint"])
        spec = read(fusion.prior.PLAN)["priors"][slug+"_wide"]
        path = fusion.depth.capture.original.INPUT/spec["file"]
        if sha(path)!=spec["sha256"]: raise ValueError("Prior changed")
        with np.load(path) as f:
            rs,cs,_ = fusion.align_sensor(fused,f["extrinsics"].astype(float))
        a,b = np.array(original["angular_gain"]),np.array(original["center_gain"])
        gains = {"adaptive":(a,b),"scene_mean":(np.full(len(c),a.mean()),np.full(len(c),b.mean())),
                 "permuted":(np.roll(a,1),np.roll(b,1))}
        gains.update({name:(np.full(len(c),w),)*2 for name,w in FIXED.items()})
        hashes = {}
        for name,(angular,center) in gains.items():
            output = fusion.blend(points,rot,c,rs,cs,angular,center)
            digest = fusion.prior.tensor_digest(output)
            if name=="adaptive":
                if digest!=original["hashes"]["adaptive"]: raise ValueError("Adaptive map differs")
            else:
                fusion.prior.write_arrays(DATA/"points"/(digest+".npz"),dict(points=output))
                hashes[name] = digest
        records[key] = hashes
    return records


def score(item):
    key,hashes = item
    _,values = fusion.prior.evaluate_points(key,set(hashes.values()),DATA)
    return key,{a:values[h] for a,h in hashes.items()}


def summarize(obs):
    parent = read(fusion.RESULT)["results"]["references"]
    by_ref = {}
    for ref,report in parent.items():
        scenes = {}
        for scene in SCENES:
            key = scene.split("/")[0]+"_base"
            base = report["scenes"][scene]["frozen"]["raw"]
            if set(obs[key])!=set(ARMS): raise ValueError("Incomplete set of controls")
            scores = {a:mean(obs[key][a][ref][m]/base[m] for m in fusion.depth.cross.temporal.METRICS[:2]) for a in ARMS}
            scores.update({a:report["scenes"][scene][b]["score"] for a,b in (("adaptive","guarded"),("no_prior","joint"))})
            scenes[scene] = scores
        overall = {a:mean(v[a] for v in scenes.values()) for a in (*ARMS,"adaptive","no_prior")}
        best = min(FIXED,key=overall.get)
        by_ref[ref] = dict(scenes=scenes,overall=overall,best_fixed=best,
                           gain_vs_best_fixed=1-overall["adaptive"]/overall[best],
                           gain_vs_scene_mean=1-overall["adaptive"]/overall["scene_mean"])
    return by_ref


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action=p.parse_args().action
    deps=[sha(x) for x in (fusion.SEAL,fusion.model.SEAL,fusion.prior.PLAN)]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,hashes=infer()))
    else:
        seal=read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Control input changed")
        deps=[sha(SEAL),sha(fusion.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["hashes"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report=read(RESULT)
        if deps!=report["inputs"] or report["results"]!=summarize(report["observed"]): raise ValueError("Control replay failed")
        if action=="verify":
            fusion.prior.old.verify_map_archive(PUBLIC/"anchor_prior_control_archive.json",seal["hashes"])
        print({k:v["overall"] for k,v in report["results"].items()})
