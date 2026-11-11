"""Reverse the direction of each estimated camera move while matching its magnitude."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import camera_prior_noise_repeats as repeat
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,RAW,ROOT,SCENES
from score_saved_geometry import read

fusion=repeat.fusion
SEEDS=(137,*repeat.SEEDS)
DATA=RAW/"prior_direction"
SEAL=PUBLIC/"prior_direction_predictions.json"
RESULT=PUBLIC/"prior_direction_scores.json"
DESIGN=ROOT/"docs/CAMERA_PRIOR_DIRECTION_CONTROL.md"


def infer():
    rows={}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key=scene.split("/")[0]+"_base"
        source=read(fusion.model.SEAL)["rows"][key]
        pose=np.array(source["fused"])
        c=fusion.centers(pose)
        rot=pose[:,:3,:3]
        points=fusion.depth.cross.temporal.load_points(fusion.model.DATA,source["hashes"]["joint"])
        z=np.einsum("nhwj,nj->nhw",points,pose[:,2,:3])+pose[:,2,3,None,None]
        rows[key]={}
        for seed in SEEDS:
            old=read(repeat.SEAL)["rows"][key][str(seed)]
            rs,cs,_=fusion.align_sensor(pose,np.array(old["supplied"]))
            for sign in (1,-1):
                w=np.full(len(c),sign*.5)
                moved=fusion.blend(points,rot,c,rs,cs,w,w)
                moved_c=c+sign*.5*(cs-c)
                coupled=np.stack([repeat.tangent.intersect(p,d,q,target)[0] for p,d,q,target in zip(points,z,moved,moved_c)])
                digest={}
                for arm,cloud in (("fixed",moved),("coupled",coupled)):
                    h=fusion.prior.tensor_digest(cloud)
                    if sign==1:
                        if h!=old["hashes"][arm]: raise ValueError("Positive control differs")
                    else:
                        fusion.prior.write_arrays(DATA/"points"/(h+".npz"),dict(points=cloud))
                        digest[arm]=h
                if sign==-1: rows[key][str(seed)]=dict(hashes=digest)
    return rows


def score(item):
    key,rows=item
    _,metrics=fusion.prior.evaluate_points(key,{r["hashes"]["coupled"] for r in rows.values()},DATA)
    return key,{s:metrics[v["hashes"]["coupled"]] for s,v in rows.items()}


def summarize(obs):
    base=read(repeat.tangent.RESULT)["results"]["references"]
    new=read(repeat.RESULT)["observed"]
    results={}
    for seed in SEEDS:
        comparisons={}
        for ref,table in base.items():
            scenes={}
            for scene in SCENES:
                key=scene.split("/")[0]+"_base"
                orig=table["scenes"][scene]
                correct=orig["surface_equal"]["raw"] if seed==137 else new[key][str(seed)]["coupled"][ref]
                opposite=obs[key][str(seed)][ref]
                def normalized(v):
                    return float(np.mean([v[m]/orig["frozen"]["raw"][m] for m in fusion.depth.cross.temporal.METRICS[:2]]))
                scenes[scene]=dict(correct=normalized(correct),opposite=normalized(opposite),correct_raw=correct,opposite_raw=opposite)
            comparisons[ref]=dict(scenes=scenes,correct=float(np.mean([v["correct"] for v in scenes.values()])),
                                  opposite=float(np.mean([v["opposite"] for v in scenes.values()])))
        results[str(seed)]=comparisons
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action=p.parse_args().action
    deps=[sha(repeat.SEAL),sha(fusion.model.SEAL),hashlib.sha256(DESIGN.read_text().encode()).hexdigest()]
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal=read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Sealed inputs differ")
        deps=[sha(SEAL),sha(repeat.RESULT),sha(repeat.tangent.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs=dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report=read(RESULT)
        if report["inputs"]!=deps or summarize(report["observed"])!=report["results"]: raise ValueError("Direction replay differs")
        if action=="verify":
            hashes={k:{f"{s}_{a}":h for s,r in rows.items() for a,h in r["hashes"].items()} for k,rows in seal["rows"].items()}
            fusion.prior.old.verify_map_archive(PUBLIC/"prior_direction_archive.json",hashes)
        print({s:{ref:v["correct"]-v["opposite"] for ref,v in z.items()} for s,z in report["results"].items()})
