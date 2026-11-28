"""Run one physically isolated five-view Track A case from its preregistered RGB/rig packet."""
import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np
import torch
import run_experiment as base
from run_profiled_d0 import sha, write_new
from run_tco_courtyard_camera import seed_model
from run_tco_final_ln_model import CAP, inputs, warm_local
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
from tco_frozen_prefix import place
from tco_one_percent_geometry import scene_scale
from tco_one_percent_protocol import ALLOCATOR, ROOT
from tco_native_stalls_quality_capacity_protocol import GSPLAT
from tco_two_track_hybrid import HybridAdamAudit, HYBRIDS, INITIALIZED
from utils.geometry import unproject_depth_to_world_points_torch


def run(job, packet, out, initialization):
    from gsplat.cuda._backend import _C
    profile = job.get("profile", "clean")
    arm, steps, seed = job["arm"], job["steps"], job["seed"]
    if (profile not in ("fixed", "stale", "real", "clean") or
            arm not in HYBRIDS | INITIALIZED | {"Full10","Full40","Full120","Full4","Full5","Cache10","ZERO","FullTime"}
            or len(set(job["frames"])) != 5 or sha(_C.__file__) != GSPLAT):
        raise ValueError("Invalid registered case, frames or rasterizer")
    if (initialization is not None) != (arm == "Distilled"):
        raise ValueError("Initialization required only for distilled arm")
    if initialization and sha(initialization) != job["initialization_sha256"]:
        raise ValueError("Initialization file digest disagrees with case")
    fixed = profile in ("fixed", "stale")
    if fixed:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    fraction = warm_local()
    seed_model(seed)
    supplied, input_id = inputs(packet, job["input_file_sha256"], fixed=False)
    if input_id != job["input_sha256"]:
        raise ValueError("Wrong array signature")
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, 0.0003, seed
    cold = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    loaded = time.perf_counter()
    captured, losses, schedule = {}, [], []
    model.register_forward_hook(lambda _m, _a, output: captured.update({k:output[k] for k in FIELDS}))
    native_loss = base.tco_module.compute_tco_losses
    native_randperm = torch.randperm

    def perm(*args, **kw):
        order = native_randperm(*args, **kw)
        if order.is_cuda and order.numel() == 5:
            schedule.append(order.detach())
        return order

    def measured_loss(*args, **kw):
        if (kw["mv_rgb_weight"] <= 0 or kw["mv_depth_weight"] or kw["mv_normal_weight"]
                or kw["depths"] is not None or kw["depth_masks"] is not None or kw["lambda_depth"]):
            raise ValueError("Unexpected change to RGB rendering, supervision, or isolation")
        with patch.object(torch, "randperm", perm):
            loss, terms = native_loss(*args, **kw)
        losses.append(loss.detach())
        if fixed:
            pred = kw["predictions"]
            probe = pred["depth"][:, :, ::19, ::19].float().square().mean() + pred["pose_enc"].float().square().mean()
            loss = loss.detach() + (probe - probe.detach())
        return loss, terms

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    start = time.perf_counter()
    audit = HybridAdamAudit(model, arm, seed, profile, initialization)
    place(model, audit.path)
    torch.cuda.synchronize()
    placement = time.perf_counter() - start
    torch.cuda.manual_seed(seed)
    with patch.object(base.tco_module, "compute_tco_losses", measured_loss):
        base.infer_mv_pointclouds_w_priors_v2(supplied, model, cfg, sample_idx=1, scene_name=job["room"].replace("/","-"))
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        depth, extr, intr = (captured[k].float() for k in ("depth","extrinsics","intrinsics"))
        canonical = unproject_depth_to_world_points_torch(depth, extr, intr)[0].cpu().numpy()
    torch.cuda.synchronize()
    end = time.perf_counter()
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    stats = torch.cuda.memory_stats()
    end_free, _ = torch.cuda.mem_get_info()
    arrays = dict(canonical=canonical, **{k:export(v) for k,v in captured.items()})
    prior_time, prior_scale = None, None
    if arm == "ZERO":
        begin = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            rig = torch.as_tensor(supplied["extrs"], dtype=torch.float32, device=depth.device)
            K = torch.as_tensor(supplied["intrs"], dtype=torch.float32, device=depth.device)
            scale, prior_scale = scene_scale(rig, extr[0])
            if scale is None:
                raise ValueError("Median non-degenerate baseline scale unavailable")
            if rig.shape[-2] == 3:
                bottom = rig.new_zeros((5,1,4))
                bottom[:,:,3] = 1
                rig = torch.cat((rig,bottom),dim=1)
            arrays["supplied_canonical"] = unproject_depth_to_world_points_torch(depth*scale,rig[None],K[None])[0].cpu().numpy()
        torch.cuda.synchronize()
        prior_time = time.perf_counter() - begin
    counters = audit.finish()
    orders = [v.cpu().tolist() for v in schedule]
    history = [float(v) for v in losses]
    if (len(history) != steps or audit.updates != steps or len(orders) != steps
            or any(sorted(p) != list(range(5)) for p in orders) or not all(map(math.isfinite, history))
            or not 0 < allocated <= CAP or any(not np.isfinite(v).all() for v in arrays.values())
            or canonical.shape != (5,*supplied["images"].shape[-2:],3) or canonical.dtype != np.float32):
        raise ValueError("Invalid reconstruction, update count, view permutation or cap")
    path = audit.path
    cache = None if path is None else dict(builds=path.builds,hits=path.hits,prefix_stages=path.prefix_stages,
               bytes=path.cached_bytes,generation=getattr(path,"generation",0),
               calls_by_generation=dict(getattr(path,"calls_by_generation",{})))
    stale = None
    if profile == "stale":
        adapter = next(p for n,p in audit.factors.items() if n.startswith("aggregator.global_blocks.11.") and n.endswith(".lora_B"))
        stale = path.stale_control(adapter)
    np.savez_compressed(out / "prediction.npz", **arrays)
    signed = None
    if audit.signed:
        np.savez_compressed(out / "fixed_signed_gradients.npz", **audit.signed)
        signed = sha(out / "fixed_signed_gradients.npz")
    result = dict(job)
    result.update(profile=profile, input_sha256=input_id, output_sha256=signature(arrays),
          predictions_sha256=sha(out / "prediction.npz"), fixed_signed_gradients_sha256=signed,
          lora_initial_sha256=audit.initial, adam_state_counts=counters, active_counts=audit.steps,
          completed_adam_updates=audit.updates, cache=cache, stale=stale, losses=history, view_permutations=orders,
          raster_shape=list(supplied["images"].shape), scene_seconds=end-start, cold_seconds=end-cold,
          checkpoint_seconds=loaded-cold, placement_seconds=placement, supplied_readout_seconds=prior_time,
          supplied_scale=prior_scale, peak_allocated=allocated, peak_reserved=reserved, cap_bytes=CAP,
          cap_mib=8960, cuda_free_before=free_before,cuda_free_after=end_free,cuda_device_total=total,
          allocation_retries=stats.get("num_alloc_retries"),oom_count=stats.get("num_ooms"),
          host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                       allocator=ALLOCATOR,fraction=fraction,gsplat_sha256=GSPLAT),
          rental_compute_usd=0)
    write_new(out / "complete.json",result)
    print(job["id"], "completed", steps,"updates in",round(end-start,3),"s",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("job","inputs","out","initialization"):
        parser.add_argument("--"+flag,type=Path,required=flag!="initialization")
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True,exist_ok=False)
    faulthandler.dump_traceback_later(150,repeat=True)
    try:
        run(job,args.inputs,args.out,args.initialization)
    except Exception as exc:
        write_new(args.out / "failed.json",dict(id=job["id"],error_type=type(exc).__name__,message=str(exc),rental_compute_usd=0))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
