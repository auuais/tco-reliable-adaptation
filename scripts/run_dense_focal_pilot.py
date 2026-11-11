"""Run independent full TCO trajectories from locally prepared camera inputs."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from evaluation_stability import dump, sha
from focal_zoom_observations import MODEL_SHA, ROOT, UPSTREAM, UPSTREAM_REV, digest_large, image_digest
from prepare_dense_focal_pilot import DESIGN, jobs
from score_saved_geometry import read


PLAN = ROOT/"manifests/dense_focal_pilot.json"
INPUT_PLAN = ROOT/"manifests/dense_focal_inputs.json"
REPLICATION = ROOT/"results/relative_focal_windows"
RUNS = ROOT/"runs/dense_focal_pilot"


def plan():
    inputs = read(INPUT_PLAN)
    if inputs["protocol_sha256"] != sha(DESIGN) or jobs() != inputs["jobs"]:
        raise ValueError("Dense study grid differs from local source")
    parent = read(REPLICATION/"summary.json")
    method = parent["results"]["dense_pilot_method"]
    if method not in ("relative","relative_with_folds"):
        raise ValueError("New-window replication did not advance a dense candidate")
    measured=read(ROOT/"results/dense_focal_pilot/geometric_support.json")
    if measured["replication_sha256"]!=sha(REPLICATION/"summary.json"):
        raise ValueError("Evidence for staged camera support changed")
    cases = {}
    for job in inputs["jobs"]:
        scene,kind,v,d = (job[k] for k in ("scene","case","view","direction"))
        key = job["prior_key"]+"_"+kind
        if key in cases:
            continue
        row, = [r for r in parent["rows"] if (r["scene"],r["case"],r["view"],r["direction"],r["noise"],r["seed"])
                == (scene,kind,v,d,"mild",137)]
        if row["rgb_sha256"] != inputs["inputs"][job["input_key"]]["rgb_sha256"]:
            raise ValueError("Dense RGB hash differs from the replication")
        support=measured["cases"][key]
        if (support["supplied_prior_sha256"]!=inputs["priors"][job["prior_key"]]["sha256"]
                or support["rgb_sha256"]!=row["rgb_sha256"]):
            raise ValueError("Staged support identity differs")
        cases[key] = dict(cpu_model_edit=row["edited"][method], two_folds=support["two_folds"],
                          matcher_seconds=support["matcher_seconds"],fold_scoring_seconds=support["fold_scoring_seconds"])
    files = [DESIGN,INPUT_PLAN,ROOT/"scripts/run_dense_focal_pilot.py",ROOT/"scripts/tco_focal_policy.py",
             ROOT/"scripts/relative_focal_confirmation.py",ROOT/"scripts/prepare_dense_focal_pilot.py",
             ROOT/"scripts/prior_ablation_config.py",ROOT/"scripts/repeatability_capture.py",
             ROOT/"scripts/score_dense_focal_pilot.py",ROOT/"scripts/dense_focal_support.py",
             ROOT/"results/dense_focal_pilot/geometric_support.json",ROOT/"configs/7scenes_baseline.yaml",
             ROOT/"patches/0001-bfloat16-numpy-export.patch"]
    return dict(method=method,cases=cases,steps=40,lr=.0003,model_sha256=MODEL_SHA,upstream_revision=UPSTREAM_REV,
                jobs=inputs["jobs"],sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},
                replication_sha256=sha(REPLICATION/"summary.json"),stage_cap_usd=3,additional_session_cap_usd=1.5)


def verified_inputs(folder, job):
    inputs = read(INPUT_PLAN)
    image, prior = inputs["inputs"][job["input_key"]],inputs["priors"][job["prior_key"]]
    arrays = {}
    for item, expected in ((image,{"images"}),(prior,{"extrinsics","intrinsics"})):
        file=folder/item["file"]
        if sha(file) != item["sha256"]:
            raise ValueError("Mismatched inference artifact")
        with np.load(file,allow_pickle=False) as data:
            if set(data.files)!=expected:
                raise ValueError("Unexpected field in device input")
            arrays.update(dict(data))
    if image_digest(arrays["images"]) != image["rgb_sha256"]:
        raise ValueError("Unexpected RGB tensor")
    return arrays


def write_points(folder, label, values):
    if values.shape != (5,392,518,3) or not np.isfinite(values).all():
        raise ValueError("Invalid dense reconstruction")
    path=folder/(label+".npz")
    with path.open("xb") as f:
        np.savez_compressed(f,points=values)
    return dict(file=path.name,sha256=sha(path),tensor_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())


def worker(args, spec, job, folder):
    import torch
    from omegaconf import OmegaConf
    from prior_ablation_config import effective_config, validate_loss_components
    from repeatability_capture import Capture
    from tco_focal_policy import initial_focal_policy
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/"base_models/vggt")]
    import tco_vggt_lora as tco
    from utils.geometry import unproject_depth_to_world_points_torch
    from utils.interfaces import infer_mv_pointclouds_w_priors_v2
    arr=verified_inputs(args.inputs,job)
    seed=job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    cfg=effective_config(OmegaConf.to_container(OmegaConf.load(ROOT/"configs/7scenes_baseline.yaml"),resolve=True),
                         1.,1.,spec["steps"],seed,spec["lr"])
    capture=Capture(folder)
    capture.rng("before_model")
    gpu=torch.cuda.get_device_name()
    start=time.perf_counter()
    model=tco.TCO_VGGT_LoRA.from_pretrained(str(args.checkpoint),enable_track=False).to("cuda").eval()
    torch.cuda.synchronize()
    record=dict(status="running",job=job,study_sha256=sha(PLAN),configuration=cfg,gpu=gpu,
                torch_version=str(torch.__version__),cuda=torch.version.cuda,device_capability=list(torch.cuda.get_device_capability()),
                research_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                load_seconds=time.perf_counter()-start,checkpoint_sha256=spec["model_sha256"],model_fresh=True)
    dump(folder/"run.json",record)
    original_parameters={id(p):p._version for p in model.parameters()}
    losses,grads=[],[]
    original_loss=tco.compute_tco_losses
    def loss(*a,**kw):
        used=kw["gt_intrinsics"][0].diagonal(dim1=-2,dim2=-1)[:,:2].detach().float().cpu().numpy()
        np.testing.assert_array_equal(used,record["decision"]["used_focal"])
        total,parts=original_loss(*a,**kw)
        values={k:float(v.detach()) for k,v in parts.items() if torch.is_tensor(v) and v.numel()==1}
        values.update(total=float(total.detach()),step=len(losses))
        validate_loss_components(cfg,values)
        losses.append(values)
        dump(folder/"losses.json",losses)
        return total,parts
    tco.compute_tco_losses=loss
    setup=model.backbone_peft_setup
    def instrument(*a,**kw):
        opt,params=setup(*a,**kw)
        capture.adapters("initial_adapters",model)
        capture.rng("before_updates")
        def before_step(optimizer,args,kwargs):
            all_g=[p.grad for p in params if p.grad is not None]
            if len(all_g)!=384:
                raise ValueError("Unexpected adapter gradient count")
            norm=float(torch.stack([g.detach().float().square().sum() for g in all_g]).sum().sqrt())
            if not np.isfinite(norm) or norm<=0:
                raise FloatingPointError("Missing or nonfinite gradient")
            grads.append(dict(step=len(grads),tensors=len(all_g),post_clip_norm=norm))
            dump(folder/"gradients.json",grads)
        opt.register_step_pre_hook(before_step)
        return opt,params
    model.backbone_peft_setup=instrument
    def observe(heads,decision):
        one=time.perf_counter()
        with torch.no_grad():
            frozen=unproject_depth_to_world_points_torch(heads["depth"],heads["extrinsics"],heads["intrinsics"])
        record["initial"]=write_points(folder,"initial",frozen[0].detach().cpu().float().numpy())
        record["decision"]=decision
        record["initial_capture_seconds"]=time.perf_counter()-one
        dump(folder/"run.json",record)
    case=spec["cases"][job["prior_key"]+"_"+job["case"]]
    support=case["two_folds"] if spec["method"]=="relative_with_folds" else None
    record["additional_precomputed_matching_seconds"]=(case["matcher_seconds"]+case["fold_scoring_seconds"]
        if support is not None and job["arm"]=="adaptive" else 0.)
    torch.cuda.reset_peak_memory_stats()
    capture.rng("before_inference")
    start=time.perf_counter()
    permitted=dict(images=torch.from_numpy(arr["images"]),extrs=arr["extrinsics"],intrs=arr["intrinsics"])
    with initial_focal_policy(model,job["arm"],spec["method"],support,observe):
        result=infer_mv_pointclouds_w_priors_v2(permitted,model,OmegaConf.create(cfg),sample_idx=1,
                                                scene_name=job["scene"].replace("/","-"))
    torch.cuda.synchronize()
    record["instrumented_inference_seconds"]=time.perf_counter()-start
    record["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    if len(losses)!=spec["steps"] or len(grads)!=spec["steps"]:
        raise ValueError("Incomplete optimizer trace")
    if any(p._version!=original_parameters[id(p)] for p in model.parameters() if id(p) in original_parameters):
        raise ValueError("Original model parameter changed")
    record["final"]=write_points(folder,"final",result)
    capture.adapters("final_adapters",model)
    capture.rng("after_inference")
    record.update(status="complete",optimizer_updates=len(grads),base_parameters_changed=False,
                  cpu_fp32_decision=case["cpu_model_edit"],capture_sha256=sha(folder/"capture.json"),
                  losses_sha256=sha(folder/"losses.json"),gradients_sha256=sha(folder/"gradients.json"))
    dump(folder/"run.json",record)


def check_source():
    args=["git","-c","core.fileMode=false","-c","core.autocrlf=input","-c","core.safecrlf=false","-C",str(UPSTREAM)]
    if subprocess.check_output(args+["rev-parse","HEAD"],text=True).strip()!=UPSTREAM_REV:
        raise ValueError("Wrong upstream TCO revision")
    changes=subprocess.check_output(args+["diff","--numstat","--ignore-cr-at-eol","HEAD"],text=True).splitlines()
    if [line.split("\t",2)[-1] for line in changes]!=["utils/interfaces.py"]:
        raise ValueError("Unexpected modifications to TCO")
    subprocess.run(args+["apply","--reverse","--check",str(ROOT/"patches/0001-bfloat16-numpy-export.patch")],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","dry-run","run","worker"))
    p.add_argument("--inputs",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--job",type=int,choices=range(72))
    p.add_argument("--attempt",type=int,choices=(1,2),default=1)
    args=p.parse_args()
    spec=plan()
    if args.action=="freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN,spec)
        return
    if spec!=read(PLAN):
        raise ValueError("Frozen dense pilot differs")
    if not args.inputs or not args.checkpoint:
        p.error("Supply local inference inputs and checkpoint")
    check_source()
    if args.action=="worker":
        if args.job is None:
            p.error("A worker needs a job index")
        job=spec["jobs"][args.job]
        folder=RUNS/(job["run_id"]+f"_a{args.attempt}")
        folder.mkdir(parents=True,exist_ok=False)
        worker(args,spec,job,folder)
        return
    if digest_large(args.checkpoint/"model.safetensors")!=MODEL_SHA:
        raise ValueError("Unexpected checkpoint file")
    checked=set()
    for job in spec["jobs"]:
        pair=(job["input_key"], job["prior_key"])
        if pair not in checked:
            verified_inputs(args.inputs,job)
            checked.add(pair)
    if args.action=="dry-run":
        print("Verified 72 prescribed trajectories, checkpoint, selected method and public source.")
        return
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise ValueError("Uncommitted research code")
    reports=ROOT/"reports/dense_focal_pilot/terminal"
    reports.mkdir(parents=True,exist_ok=True)
    for i,job in enumerate(spec["jobs"]):
        if args.job is not None and i!=args.job:
            continue
        tag=job["run_id"]+f"_a{args.attempt}"
        path=reports/(tag+".json")
        if args.attempt==2 and read(reports/(job["run_id"]+"_a1.json"))["status"]!="interrupted":
            raise ValueError("Only interrupted trajectories may be retried")
        if path.exists():
            prior=read(path)
            if prior["status"]!="complete" or prior["run_sha256"]!=sha(RUNS/tag/"run.json"):
                raise ValueError("Interrupted or inconsistent prior attempt: "+tag)
            continue
        with (reports/(tag+".log")).open("x") as log:
            try:
                subprocess.run([sys.executable,"-u",__file__,"worker","--inputs",str(args.inputs),"--checkpoint",
                                str(args.checkpoint),"--job",str(i),"--attempt",str(args.attempt)],check=True,timeout=360,
                               env=dict(os.environ,PYTHONHASHSEED=str(job["seed"]),OPENBLAS_NUM_THREADS="2",
                                        OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",WANDB_MODE="disabled"),stdout=log,stderr=subprocess.STDOUT)
                r=read(RUNS/tag/"run.json")
                if r["status"]!="complete" or r["optimizer_updates"]!=40 or r["job"]!=job:
                    raise ValueError("Worker did not produce a complete trajectory")
                dump(path,dict(status="complete",job=job,attempt=args.attempt,run_sha256=sha(RUNS/tag/"run.json")))
            except (ValueError,OSError,subprocess.SubprocessError) as exc:
                dump(path,dict(status="interrupted",job=job,attempt=args.attempt,reason=str(exc)))
                raise
        print(i+1,tag,"complete",flush=True)


if __name__=="__main__":
    main()
