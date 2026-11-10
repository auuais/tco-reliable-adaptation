"""Quantify reference-dependent TCO gains under recorded identical-seed reruns."""
import argparse
from itertools import product
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np

from evaluation_stability import dump,sha
from score_saved_geometry import read
from screen_sensor_extrinsic import SCENES, REFERENCES, METRICS, make_reference, source_run

ROOT=Path(__file__).resolve().parents[1]
NAME="reference_repeat_sensitivity"
PLAN=ROOT/"manifests"/(NAME+".json")
PUBLIC=ROOT/"results"/NAME
RAW=ROOT/"reports"/NAME
V15=ROOT/"results/sensor_extrinsic_v15/summary.json"
REPEATED=(SCENES[0],SCENES[2])
METHODS=("tco","half_shape")


def plan():
    runs=[r for r in read(ROOT/"results/depth_trust_v11/sealed_predictions.json")["records"]
          if r["mode"] in METHODS and r["repeat"] in (1,2)]
    if len(runs)!=8:raise ValueError("Unexpected repeat grid")
    return dict(runs=runs,code_sha256=sha(Path(__file__)),primary_sha256=sha(V15),
                evaluator_sha256=sha(ROOT/"scripts/run_repeat_calibration.py"),
                reference_builder_sha256=sha(ROOT/"scripts/screen_sensor_extrinsic.py"),
                references=list(REFERENCES),analysis="Retain the three-seed, three-scene primary analysis. Rescore four archived additional TCO executions and four matched half-depth maps against the same three references. Substitute one same-seed execution for seed 42 in each of Heads and Fire, keeping all other primary cells fixed. Report all nine combinations and the observed envelope, the unmodified primary score, scene-specific rerun ranges, paired reference contrasts, and equal-weight two-scene means leaving one development scene out. Pair the same executions across methods and references. Descriptive sensitivity after seeing v15. No independence or confidence-interval interpretation.",
                additional_cloud_usd=0)


def score_group(scene,ref,evidence,subset):
    from repeatability_capture import array_digest
    from run_repeat_calibration import evaluate
    import torch
    import open3d as o3d
    v15=read(V15)["rows"]
    src=read(evidence/"runs"/source_run(scene)/"run.json")
    arrays=dict(np.load(evidence/src["evaluation_inputs_relative"],allow_pickle=False))
    if array_digest(arrays)!=src["evaluation_tensor_sha256"]:raise ValueError("Changed reference input")
    points=make_reference(arrays,scene,ref,subset)
    mask=arrays["valid_mask"]
    fingerprint=array_digest(dict(points=points,mask=mask))
    if {r["reference_sha256"] for r in v15 if r["scene"]==scene and r["reference"]==ref}!={fingerprint}:
        raise ValueError("Reference differs from primary study")
    data=dict(images=torch.from_numpy(arrays["images"]),pointclouds=points,valid_mask=mask)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for r in plan()["runs"]:
        if r["scene"]!=scene:continue
        pred=ROOT/"reports/depth_trust_v11"/r["job_id"]/"predictions.npy"
        if sha(pred)!=r["output_sha256"]:raise ValueError("Changed repeat point map")
        out=RAW/r["job_id"]/ref
        out.mkdir(parents=True,exist_ok=True)
        result=out/"evaluation.json"
        meta=dict(scene=scene,reference=ref,method=r["mode"],seed=r["seed"],repeat=r["repeat"],
                  prediction_sha256=r["output_sha256"],reference_sha256=fingerprint)
        if result.exists():
            if any(read(result)[k]!=v for k,v in meta.items()):raise ValueError("Checkpoint mismatch")
            continue
        np.random.seed(42)
        torch.manual_seed(42)
        o3d.utility.random.seed(42)
        m=evaluate(np.load(pred,allow_pickle=False),data,out)
        dump(result,dict(**meta,metrics={k:m[k] for k in METRICS}))


def summarize(rows):
    primary=read(V15)
    extra={(r["scene"],r["seed"],r["mode"],r["repeat"]):r["output_sha256"] for r in plan()["runs"]}
    observed=set()
    for r in rows:
        key=r["scene"],r["seed"],r["method"],r["repeat"]
        if r["prediction_sha256"]!=extra[key] or (key,r["reference"]) in observed:raise ValueError("Repeat identity mismatch")
        observed.add((key,r["reference"]))
    if observed!=set(product(extra,REFERENCES)):raise ValueError("Incomplete repeat references")
    group={}
    by_ref={}
    for ref in REFERENCES:
        for scene in SCENES:
            base=next(r for r in primary["rows"] if r["reference"]==ref and r["scene"]==scene and r["method"]=="initial")
            for r in [dict(v,repeat=0) for v in primary["rows"]]+rows:
                if r["method"] not in METHODS or r["scene"]!=scene or r["reference"]!=ref:continue
                if r["reference_sha256"]!=base["reference_sha256"]:raise ValueError("Mismatched denominator")
                group[scene,r["seed"],r["repeat"],r["method"],ref]=mean(r["metrics"][k]/base["metrics"][k] for k in METRICS[:2])
        by_ref[ref]={}
        for method in METHODS:
            swaps=[]
            for heads,fire in product(range(3),repeat=2):
                repeat={SCENES[0]:heads,SCENES[1]:0,SCENES[2]:fire}
                scenes={s:mean(group[s,seed,repeat[s] if seed==42 else 0,method,ref] for seed in (42,123,2026)) for s in SCENES}
                swaps.append(dict(heads=heads,fire=fire,score=mean(scenes.values()),scenes=scenes))
            reference_score=primary["tables"][ref][method]["score"]
            if abs(swaps[0]["score"]-reference_score)>1e-12:raise ValueError("Original analysis not reproduced")
            by_ref[ref][method]=dict(swaps=swaps,primary=reference_score,
                       range=[min(v["score"] for v in swaps),max(v["score"] for v in swaps)],
                       seed42_scene_ranges={s:[min(group[s,42,i,method,ref] for i in range(3)),
                                              max(group[s,42,i,method,ref] for i in range(3))] for s in REPEATED},
                       leave_one_scene_out={drop:mean(v for s,v in swaps[0]["scenes"].items() if s!=drop) for drop in SCENES})
    contrasts={}
    for method in METHODS:
        contrasts[method]={}
        for ref in REFERENCES[1:]:
            diffs=[a["score"]-b["score"] for a,b in zip(by_ref[ref][method]["swaps"],by_ref["released"][method]["swaps"])]
            contrasts[method][ref+"_minus_released"]=dict(primary=diffs[0],range=[min(diffs),max(diffs)])
    return dict(manifest_sha256=sha(PLAN),rows=rows,by_reference=by_ref,paired_reference_contrasts=contrasts,
                additional_cloud_usd=0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","score","group","collect","verify"))
    p.add_argument("--scene",choices=REPEATED)
    p.add_argument("--reference",choices=REFERENCES)
    p.add_argument("--evidence",type=Path)
    p.add_argument("--subset",type=Path)
    args=p.parse_args()
    if args.action=="freeze":
        if PLAN.exists():raise FileExistsError(PLAN)
        dump(PLAN,plan())
    elif read(PLAN)!=plan():raise ValueError("Sensitivity differs from protocol")
    elif args.action=="group":score_group(args.scene,args.reference,args.evidence,args.subset)
    elif args.action=="score":
        from concurrent.futures import ThreadPoolExecutor
        def run(pair):
            scene,ref=pair
            RAW.mkdir(parents=True,exist_ok=True)
            with (RAW/(scene.replace("/","_")+"_"+ref+".log")).open("a") as log:
                subprocess.run([sys.executable,__file__,"group","--scene",scene,"--reference",ref,"--evidence",str(args.evidence),
                                "--subset",str(args.subset)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800,
                               env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,product(REPEATED,REFERENCES)))
    elif args.action=="collect":
        summary=summarize([read(f) for f in sorted(RAW.rglob("evaluation.json"))])
        if (PUBLIC/"summary.json").exists():raise FileExistsError("Existing sensitivity results")
        dump(PUBLIC/"summary.json",summary)
    else:
        summary=read(PUBLIC/"summary.json")
        if summarize(summary["rows"])!=summary:raise ValueError("Changed sensitivity analysis")
        print("Verified 24 additional scores and all matched repeat substitutions")


if __name__=="__main__":main()
