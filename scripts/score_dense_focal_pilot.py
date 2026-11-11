"""Score sealed GPU point maps on both crop-correct depth references, on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from statistics import mean

import numpy as np

from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import INPUT, PUBLIC, RAW, REF, jobs
from score_saved_geometry import read


REFS=("registered_rgb","released")
ARMS=("retain","consensus","adaptive")
METRICS=("accuracy_mean_m","completeness_mean_m","normal_consistency_mean")


def per_reference(rows,reference):
    def average(group):
        result={}
        for arm in ARMS:
            cells=[r["scores"][reference] for r in group if r["job"]["arm"]==arm]
            acc,comp=(mean(c["final"][k]/c["initial"][k] for c in cells) for k in METRICS[:2])
            result[arm]=dict(relative_accuracy=acc,relative_completeness=comp,score=(acc+comp)/2,
                             raw={k:mean(c["final"][k] for c in cells) for k in METRICS})
        result["frozen"]=dict(relative_accuracy=1.,relative_completeness=1.,score=1.,
                              raw={k:mean(r["scores"][reference]["initial"][k] for r in group) for k in METRICS})
        return result
    by_scene={s:average([r for r in rows if r["job"]["scene"]==s]) for s in sorted({r["job"]["scene"] for r in rows})}
    return dict(overall=average(rows),by_scene=by_scene,
                by_seed={str(s):average([r for r in rows if r["job"]["seed"]==s]) for s in (42,123)},
                by_case={s:average([r for r in rows if r["job"]["case"]==s]) for s in ("valid_zoom","metadata_fault")})


def summarize(rows):
    grid=jobs()
    if len(rows)!=len(grid) or any(a!=b["job"] for a,b in zip(grid,rows)):
        raise ValueError("A dense trajectory is missing or misplaced")
    results={ref:per_reference(rows,ref) for ref in REFS}
    physical=results["registered_rgb"]
    overall=physical["overall"]
    comparator=min(("retain","consensus","frozen"),key=lambda m:overall[m]["score"])
    gain=1-overall["adaptive"]["score"]/overall[comparator]["score"]
    released=results["released"]["overall"]
    checks=dict(gain=gain>=.05,seed_directions=all(v["adaptive"]["score"]<v[comparator]["score"] for v in physical["by_seed"].values()),
                distances=all(overall["adaptive"][m]<=1.02*overall[comparator][m] for m in ("relative_accuracy","relative_completeness")),
                released_reference=released["adaptive"]["score"]<=released["retain"]["score"])
    adaptive=[r for r in rows if r["job"]["arm"]=="adaptive"]
    return dict(references=results,comparator=comparator,registered_gain=gain,checks=checks,
                supports_transfer=all(checks.values()),
                adaptive_ordered_correct=sum(r["edited"]==(r["job"]["case"]=="metadata_fault") for r in adaptive),
                adaptive_false_edits=sum(r["edited"] for r in adaptive if r["job"]["case"]=="valid_zoom"),
                gpu_cpu_rule_agreement=sum(r["edited"]==r["cpu_fp32_decision"] for r in adaptive),
                adaptive_run_count=len(adaptive),peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows))


def validate_run(folder, r, job, spec):
    if (r["job"]!=job or r["status"]!="complete" or r["optimizer_updates"]!=40 or r["base_parameters_changed"]
            or r["study_sha256"]!=sha(spec) or not r["model_fresh"]):
        raise ValueError("Invalid trajectory: "+folder.name)
    for key,name in (("losses_sha256","losses.json"),("gradients_sha256","gradients.json"),("capture_sha256","capture.json")):
        if sha(folder/name)!=r[key]:
            raise ValueError("Changed trajectory record")
    for x in read(folder/"capture.json")["records"].values():
        if sha(folder/x["file"])!=x.get("file_sha256",x.get("sha256")):
            raise ValueError("Changed adapter or RNG file")
    if [v["step"] for v in read(folder/"losses.json")]!=list(range(40)):
        raise ValueError("Loss trace truncated")
    grads=read(folder/"gradients.json")
    if ([v["step"] for v in grads]!=list(range(40)) or
            not all(v["tensors"]==384 and np.isfinite(v["post_clip_norm"]) and v["post_clip_norm"]>0 for v in grads)):
        raise ValueError("Invalid adapter gradients")
    for key in ("initial","final"):
        p=folder/r[key]["file"]
        if p.name!=key+".npz" or sha(p)!=r[key]["sha256"]:
            raise ValueError("Point-map file changed")


def seal():
    from run_dense_focal_pilot import PLAN,RUNS
    terminals=RAW/"terminal"
    entries=[]
    for job in jobs():
        done=[]
        for attempt in (1,2):
            tag=job["run_id"]+f"_a{attempt}"
            terminal=terminals/(tag+".json")
            if not terminal.exists():
                continue
            t=read(terminal)
            if t["status"]=="interrupted":
                continue
            folder=RUNS/tag
            r=read(folder/"run.json")
            if t["job"]!=job or t["run_sha256"]!=sha(folder/"run.json"):
                raise ValueError("Unsealed run record")
            validate_run(folder,r,job,PLAN)
            done.append(dict(tag=tag,job=job,run_sha256=t["run_sha256"],initial=r["initial"],final=r["final"],
                             edited=r["decision"]["edited"],cpu_fp32_decision=r["cpu_fp32_decision"],
                             peak_allocated_bytes=r["peak_allocated_bytes"],instrumented_inference_seconds=r["instrumented_inference_seconds"],
                             gpu=r["gpu"],attempt=attempt))
        if len(done)!=1:
            raise ValueError("Expected exactly one successful execution: "+job["run_id"])
        entries.extend(done)
    path=PUBLIC/"seal.json"
    if path.exists():
        raise FileExistsError(path)
    dump(path,dict(manifest_sha256=sha(PLAN),rows=entries))


def evaluate(job_index):
    import open3d as o3d
    import torch
    from run_dense_focal_pilot import INPUT_PLAN,RUNS
    from run_repeat_calibration import evaluate as original_evaluate
    index=read(INPUT_PLAN)
    entry=read(PUBLIC/"seal.json")["rows"][job_index]
    k=entry["job"]["input_key"]
    rgb,reference=index["inputs"][k],index["references"][k]
    if sha(INPUT/rgb["file"])!=rgb["sha256"] or sha(REF/reference["file"])!=reference["sha256"]:
        raise ValueError("Evaluation references differ")
    torch.set_num_threads(2)
    with np.load(INPUT/rgb["file"],allow_pickle=False) as f:
        images=torch.from_numpy(f["images"])
    with np.load(REF/reference["file"],allow_pickle=False) as f:
        refs=dict(f)
    records={}
    for stage in ("initial","final"):
        loc=RUNS/entry["tag"]/entry[stage]["file"]
        if sha(loc)!=entry[stage]["sha256"]:
            raise ValueError("Changed reconstruction")
        with np.load(loc,allow_pickle=False) as f:
            points=f["points"]
        digest=hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        if digest!=entry[stage]["tensor_sha256"]:
            raise ValueError("Unexpected point map")
        for name in REFS:
            folder=RAW/"scores"/k/name/digest
            folder.mkdir(parents=True,exist_ok=True)
            path=folder/"verified.json"
            identity=dict(tensor_sha256=digest,reference_archive_sha256=reference["sha256"],reference=name)
            if path.exists():
                obs=read(path)
            else:
                np.random.seed(42)
                torch.manual_seed(42)
                o3d.utility.random.seed(42)
                metrics=original_evaluate(points,dict(images=images,pointclouds=refs[name],valid_mask=refs["valid_mask"]),folder)
                obs=dict(**identity,metrics={k:metrics[k] for k in METRICS},valid_points=metrics["valid_points"])
                dump(path,obs)
            if (any(obs[k]!=v for k,v in identity.items()) or obs["valid_points"]!=reference["valid_pixels"]
                    or not all(np.isfinite(x) for x in obs["metrics"].values())
                    or min(obs["metrics"][k] for k in METRICS[:2])<=0):
                raise ValueError("Invalid dense score")
            records.setdefault(name,{})[stage]=obs["metrics"]
    dump(RAW/"scored"/(entry["tag"]+".json"),dict(**entry,scores=records))


def publish():
    sealpath=PUBLIC/"seal.json"
    rows=[]
    for entry in read(sealpath)["rows"]:
        r=read(RAW/"scored"/(entry["tag"]+".json"))
        if any(r[k]!=v for k,v in entry.items()):
            raise ValueError("Evaluation record identity changed")
        rows.append(r)
    path=PUBLIC/"summary.json"
    if path.exists():
        raise FileExistsError(path)
    dump(path,dict(seal_sha256=sha(sealpath),rows=rows,results=summarize(rows)))


def score_group(indices):
    for i in indices:
        evaluate(i)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("seal","evaluate","publish","verify"))
    p.add_argument("--job",type=int,choices=range(72))
    p.add_argument("--workers",type=int,choices=range(1,5),default=1)
    args=p.parse_args()
    if args.action=="seal":
        seal()
    elif args.action=="evaluate":
        grid=jobs()
        selected=range(len(grid)) if args.job is None else [args.job]
        # Each image owns its own cache. No two processes ever write the same cache key.
        groups={}
        for i in selected:
            groups.setdefault(grid[i]["input_key"],[]).append(i)
        if args.workers==1:
            for group in groups.values():
                score_group(group)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                list(pool.map(score_group,groups.values()))
    elif args.action=="publish":
        publish()
    else:
        data=read(PUBLIC/"summary.json")
        if data["seal_sha256"]!=sha(PUBLIC/"seal.json") or data["results"]!=summarize(data["rows"]):
            raise ValueError("Released evidence does not replay")
        print("Verified all 72 dense trajectories and both reference comparisons.")
