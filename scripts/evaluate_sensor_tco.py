"""Seal the in-loop calibration experiment and score both common references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np

from camera_depth_swap import Rig, relative_rig, median_baseline, unproject
from depth_trust import move_along_rays
from evaluation_stability import dump, sha
from fit_sensor_extrinsic import homogeneous
from pointmap_camera_diagnostic import recover
from prior_corruptions import tensor_hash
from run_sensor_tco import ROOT, NAME, MANIFEST, RAW, SCENES, CONDITIONS
from score_saved_geometry import read, store_map
from screen_sensor_extrinsic import make_reference, METRICS

PUBLIC = ROOT/"results"/NAME
REFS = ("released","registered_rgb")
METHODS = ["initial","direct_released","direct_fixed_rgb",*CONDITIONS,*["half_"+c for c in CONDITIONS]]


def completed(job):
    attempts = []
    for attempt in (1,2):
        path = RAW/(job["run_id"]+f"_a{attempt}.json")
        if path.exists():
            attempts.append(read(path))
    successes = [r for r in attempts if r["status"]=="complete"]
    if len(successes)!=1:
        raise ValueError(f"Expected one completed attempt for {job['run_id']}")
    success = successes[0]
    folder = ROOT/"runs"/(job["run_id"]+f"_a{success['attempt']}")
    if sha(folder/"run.json")!=success["record_sha256"]:
        raise ValueError("Run record changed")
    return folder,attempts


def prepare():
    import torch
    from run_repeat_calibration import tco_module
    from repeatability_capture import array_digest
    p = read(MANIFEST)
    current = sha(MANIFEST)
    physical = np.array(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    common, priors, maps, evidence, entries = {}, {}, {}, [], {}
    out = RAW/"sealed"
    out.mkdir(exist_ok=False)
    for job in p["jobs"]:
        scene,condition = job["scene"],job["condition"]
        folder,attempts = completed(job)
        run = read(folder/"run.json")
        cap = read(folder/"capture.json")["records"]
        if (run["sensor_condition"]!=condition or run["arguments"]["frames"]!=p["frames"] or run["manifest_sha256"]!=current
                or run["status"]!="complete" or run["base_parameters_modified"] or run["upstream_commit"]!=p["upstream_commit"]
                or run["arguments"]["scene"]!=scene or run["arguments"]["seed"]!=42):
            raise ValueError("Run disagrees with frozen job")
        for name in ("losses","gradients"):
            trace = read(folder/(name+".json"))
            if len(trace)!=40:
                raise ValueError("Incomplete adaptation trace")
        for name in ("initial_predictions","initial_adapters","final_adapters"):
            r = cap[name]
            if sha(folder/r["file"])!=r["file_sha256"]:
                raise ValueError("Corrupt capture")
        shared = dict(heads=cap["initial_predictions"]["tensor_sha256"],adapters=cap["initial_adapters"]["tensor_sha256"],
                      rng=cap["after_adapter_initialization_rng"]["sha256"],reference=run["evaluation_tensor_sha256"],
                      inputs=run["evaluation_inputs_relative"])
        if common.setdefault(scene,shared)!=shared:
            raise ValueError("Starting state, RNG or reference differs across conditions")
        r = cap["initial_predictions"]
        heads = dict(np.load(folder/r["file"],allow_pickle=False))
        if array_digest(heads)!=shared["heads"]:
            raise ValueError("Initial predictions changed")
        supplied = dict(np.load(folder/"supplied_priors.npz",allow_pickle=False))
        for k,value in supplied.items():
            if tensor_hash(value)!=run["prior_metadata"][k+"_sha256"]:
                raise ValueError("Prior tensor differs from original record")
        priors[scene,condition] = supplied
        z = heads["depth"][0,...,0]
        initial = relative_rig(heads["extrinsics"][0],heads["intrinsics"][0])
        result = np.load(folder/"predictions.npy",allow_pickle=False)
        fitted = [recover(view) for view in result]
        centers = [-c["extrinsics"][:3,:3].T@c["extrinsics"][:3,3] for c,_ in fitted]
        half, _ = move_along_rays(result,centers,z,np.stack([v for _,v in fitted]),"half_shape")
        outputs = {condition:result,"half_"+condition:half}
        if condition in ("released","fixed_rgb"):
            rig = relative_rig(supplied["extrinsics"],supplied["intrinsics"])
            scaled = Rig(rig.rotation,rig.center*median_baseline(initial)/median_baseline(rig),rig.intrinsic)
            outputs["direct_"+condition] = unproject(scaled,z)
        if condition=="released":
            with torch.no_grad(),torch.autocast("cpu",enabled=False):
                zero = tco_module.unproject_depth_to_world_points_torch(*[torch.from_numpy(heads[k]).float()
                              for k in ("depth","extrinsics","intrinsics")])
            outputs["initial"] = zero[0].numpy()
        for m,pts in outputs.items():
            maps.setdefault(scene,{})[m] = store_map(out,pts,scene,folder.name,entries)
        evidence.append(dict(**job,attempts=attempts,source_prediction_sha256=sha(folder/"predictions.npy"),
                             corrected_pose_sha256=tensor_hash(supplied["extrinsics"]),intrinsic_sha256=tensor_hash(supplied["intrinsics"]),
                             model_inference_adaptation_seconds=run["inference_and_adaptation_seconds"],
                             peak_allocated_bytes=run["peak_allocated_bytes"],peak_reserved_bytes=run["peak_reserved_bytes"],
                             research_commit=run["research_commit"],final_adapters_sha256=cap["final_adapters"]["tensor_sha256"]))
    for s in SCENES:
        origin = priors[s,"released"]
        fixed = (physical@homogeneous(origin["extrinsics"]))[:,:3]
        rgb_k = priors[s,"intrinsics_only"]["intrinsics"]
        if set(maps[s])!=set(METHODS) or np.array_equal(rgb_k,origin["intrinsics"]):
            raise ValueError("Missing method or intrinsic intervention")
        for c in CONDITIONS:
            np.testing.assert_allclose(priors[s,c]["extrinsics"],fixed if c in ("frame_only","fixed_rgb") else origin["extrinsics"],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(priors[s,c]["intrinsics"],rgb_k if c in ("intrinsics_only","fixed_rgb") else origin["intrinsics"])
    dump(PUBLIC/"seal.json",dict(manifest_sha256=current,evaluation_source_sha256=sha(Path(__file__)),
                                common=common,maps=maps,entries=entries,runs=evidence))


def score(scene,ref):
    import open3d as o3d
    import torch
    from run_repeat_calibration import evaluate
    from repeatability_capture import array_digest
    seal = read(PUBLIC/"seal.json")
    if seal["evaluation_source_sha256"]!=sha(Path(__file__)) or seal["manifest_sha256"]!=sha(MANIFEST):
        raise ValueError("Evaluation code or protocol changed")
    group = seal["common"][scene]
    source = dict(np.load(ROOT/group["inputs"],allow_pickle=False))
    if array_digest(source)!=group["reference"]:
        raise ValueError("Reference tensors changed")
    points = make_reference(source,scene,ref,None)
    mask = source["valid_mask"]
    digest = array_digest(dict(points=points,mask=mask))
    data = dict(images=torch.from_numpy(source["images"]),pointclouds=points,valid_mask=mask)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for method,key in seal["maps"][scene].items():
        record = seal["entries"][key]
        path = RAW/"sealed"/key/"points.npy"
        if sha(path)!=record["sha256"]:
            raise ValueError("Prediction changed")
        folder = RAW/"scores"/scene.replace("/","_")/ref/method
        folder.mkdir(parents=True,exist_ok=True)
        identity = dict(scene=scene,method=method,reference=ref,reference_sha256=digest,prediction_sha256=record["sha256"])
        output = folder/"evaluation.json"
        if output.exists():
            if any(read(output)[k]!=v for k,v in identity.items()):
                raise ValueError("Checkpoint changed")
            continue
        np.random.seed(42)
        torch.manual_seed(42)
        o3d.utility.random.seed(42)
        metrics = evaluate(np.load(path,allow_pickle=False),data,folder)
        dump(output,dict(**identity,metrics={k:metrics[k] for k in METRICS}))


def summarize(rows,seal):
    expected = {(s,r,m) for s in SCENES for r in REFS for m in METHODS}
    grid = {(r["scene"],r["reference"],r["method"]):r for r in rows}
    if set(grid)!=expected or len(rows)!=len(expected):
        raise ValueError("Incomplete factorial evaluation")
    tables,decisions,effects = {},{},{}
    for ref in REFS:
        table = {}
        for method in METHODS:
            scenes = {}
            for scene in SCENES:
                baseline = grid[scene,ref,"initial"]
                cell = grid[scene,ref,method]
                key = seal["maps"][scene][method]
                if (cell["reference_sha256"]!=baseline["reference_sha256"] or cell["prediction_sha256"]!=seal["entries"][key]["sha256"]
                        or not all(np.isfinite(v) for v in cell["metrics"].values())):
                    raise ValueError("Prediction, reference or metric invalid")
                ratios = [cell["metrics"][k]/baseline["metrics"][k] for k in METRICS[:2]]
                scenes[scene] = dict(score=mean(ratios),harmful=max(ratios)>1.01)
            table[method] = dict(scenes=scenes,score=mean(v["score"] for v in scenes.values()),
                                 harmful=sum(v["harmful"] for v in scenes.values()))
        best = min((m for m in METHODS if m!="fixed_rgb"),key=lambda m:table[m]["score"])
        delta = 1-table["fixed_rgb"]["score"]/table[best]["score"]
        loss = max(table["fixed_rgb"]["scenes"][s]["score"]/table[best]["scenes"][s]["score"]-1 for s in SCENES)
        harm = table["fixed_rgb"]["harmful"]<=table[best]["harmful"]
        decisions[ref] = dict(best_fixed=best,gain=delta,max_scene_relative_loss=loss,harm_pass=harm,
                              passed=delta>=.05 and loss<=.01 and harm)
        effects[ref] = {}
        for s in SCENES:
            y00,y10,y01,y11 = [table[c]["scenes"][s]["score"] for c in CONDITIONS]
            effects[ref][s] = dict(frame_main=((y10-y00)+(y11-y01))/2,k_main=((y01-y00)+(y11-y10))/2,
                                   interaction=y11-y10-y01+y00)
        tables[ref] = table
    return dict(rows=rows,tables=tables,decisions=decisions,effects=effects,seal_sha256=sha(PUBLIC/"seal.json"),
                motivates_repeat_and_noise=all(v["passed"] for v in decisions.values()),additional_cloud_usd=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","score","group","collect","verify"))
    p.add_argument("--scene",choices=SCENES)
    p.add_argument("--reference",choices=REFS)
    args = p.parse_args()
    if args.action=="prepare":
        prepare()
    elif args.action=="group":
        score(args.scene,args.reference)
    elif args.action=="score":
        def run(pair):
            scene,ref = pair
            log = RAW/("evaluation_"+scene.replace("/","_")+"_"+ref+".log")
            with log.open("a") as f:
                subprocess.run([sys.executable,__file__,"group","--scene",scene,"--reference",ref],stdout=f,stderr=subprocess.STDOUT,
                               check=True,timeout=3600,env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run,[(s,r) for r in REFS for s in SCENES]))
    elif args.action=="collect":
        data = summarize([read(f) for f in sorted((RAW/"scores").rglob("evaluation.json"))],read(PUBLIC/"seal.json"))
        if (PUBLIC/"summary.json").exists():
            raise FileExistsError("Existing summary")
        dump(PUBLIC/"summary.json",data)
    else:
        data = read(PUBLIC/"summary.json")
        if data!=summarize(data["rows"],read(PUBLIC/"seal.json")):
            raise ValueError("Decision cannot be reproduced")
        print("Verified",len(data["rows"]),"evaluations")


if __name__=="__main__":
    main()
