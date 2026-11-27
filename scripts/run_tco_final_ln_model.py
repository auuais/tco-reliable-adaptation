"""Run one isolated registered five-view reconstruction on the physical RTX 3080."""
import argparse
from contextlib import ExitStack
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
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
from run_tco_courtyard_camera import seed_model
import run_tco_render_lifetime as lifetime
import tco_render_lifetime as runtime
from tco_final_ln_policy import ParameterAudit, resolve_norms, selected_names
from tco_final_ln_raster import RasterPair
from tco_frozen_prefix import TAPS, place
from tco_native_stalls_quality_capacity_protocol import GSPLAT
from tco_one_percent_protocol import ALLOCATOR, ROOT
from utils.geometry import unproject_depth_to_world_points_torch


CAP = 8960 * 2**20
PROFILES = ("clean", "diagnostic", "native_bridge", "fixed", "fixed_bridge")


def warm_local():
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 3080"
            or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != ALLOCATOR):
        raise ValueError("Expected exclusive local RTX 3080 and registered native allocator")
    total = torch.cuda.get_device_properties(0).total_memory
    if total <= CAP:
        raise ValueError("Insufficient physical GPU memory for registered cap")
    fraction = CAP / total
    torch.cuda.set_per_process_memory_fraction(fraction)
    if abs(torch.cuda.get_per_process_memory_fraction() - fraction) > 1e-7:
        raise ValueError("Absolute cap readback failed")
    warm = torch.ones((256, 256), device="cuda")
    (warm @ warm).sum().item()
    del warm
    torch.cuda.empty_cache()
    return fraction


def inputs(path, expected, fixed):
    file_sha = sha(path)
    if file_sha != expected:
        raise ValueError("Portable input differs from the locked case")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"images", "extrs", "intrs"}:
            raise ValueError("GPU input permits only RGB images and camera priors")
        arr = {k:np.ascontiguousarray(data[k]) for k in data.files}
    rgb, extr, intr = arr["images"], arr["extrs"], arr["intrs"]
    if (rgb.dtype != np.float32 or rgb.ndim != 4 or rgb.shape[:2] != (5,3)
            or rgb.shape[-1] not in ((294,518) if fixed else (518,)) or rgb.shape[-2] % 14
            or extr.shape not in ((5,3,4),(5,4,4)) or intr.shape != (5,3,3)
            or any(v.dtype.kind != "f" or not np.isfinite(v).all() for v in arr.values())):
        raise ValueError("Invalid five-view geometry packet")
    return dict(images=torch.from_numpy(rgb), extrs=extr, intrs=intr), signature(dict(rgb=rgb, poses=extr, intrinsics=intr))


def render_settings(original):
    def revised(*args, **kwargs):
        source = original(*args, **kwargs)
        return runtime._replace_once(source, "colors=dummy_cols_cam,",
                                     "colors=cols_rgb_b.unsqueeze(0).expand(viewmats_b.shape[0], N_b, cols_rgb_b.shape[-1]),")
    return revised


def run(job, packet, out):
    import gsplat
    from gsplat.cuda._backend import _C

    profile = job.get("profile") or ("diagnostic" if job.get("diagnostic") else "clean")
    if profile not in PROFILES:
        raise ValueError("Unknown worker profile " + str(profile))
    fixed = profile in ("fixed", "fixed_bridge")
    if fixed:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    steps, seed = job["steps"], job["seed"]
    if (steps not in (0, 2, 10, 20, 40) or len(job["frames"]) != 5 or len(set(job["frames"])) != 5
            or selected_names(job["ln_policy"]) and job["ln_lr"] not in (0.0, 1.e-5, 3.e-5)
            or not selected_names(job["ln_policy"]) and job["ln_lr"] != 0
            or not steps and job["ln_policy"] != "none"):
        raise ValueError("Invalid model or update configuration")
    fraction = warm_local()
    if sha(_C.__file__) != GSPLAT:
        raise ValueError("Unregistered rasterizer binary")
    seed_model(seed)
    supplied, input_id = inputs(packet, job["input_file_sha256"], fixed)
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, 0.0003, seed
    state = dict(updates=0, lora_tensors=0)
    cold = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    loaded = time.perf_counter()
    initial = {name:p.detach().cpu().clone() for name,p in resolve_norms(model).items()}
    captured = {}
    model.register_forward_hook(lambda _, _args, outputs: captured.update({name:outputs[name] for name in FIELDS}))
    batches_job = dict(job, policy="camera")
    path, batches, _ = lifetime.setup_execution(model, batches_job, state, None, None, None)
    original_loss = base.tco_module.compute_tco_losses
    pair = RasterPair(gsplat.rasterization_2dgs, "joint", lambda *_args, **_kw: None)
    losses, schedule = [], []
    native_randperm = torch.randperm

    def capture_permutation(*a, **kw):
        output = native_randperm(*a, **kw)
        if output.is_cuda and output.numel() == 5:
            schedule.append(output.detach())
        return output

    def loss(*a, **kw):
        if (kw["mv_rgb_weight"] <= 0 or kw["mv_depth_weight"] != 0 or kw["mv_normal_weight"] != 0
                or kw["depths"] is not None or kw["depth_masks"] is not None or kw["lambda_depth"] != 0):
            raise ValueError("Unexpected change to the registered RGB-only rendering loss or depth isolation")
        with patch.object(torch, "randperm", capture_permutation):
            value, terms = original_loss(*a, **kw)
        losses.append(value.detach())
        pair.check(len(losses))
        if fixed:
            pred = kw["predictions"]
            probe = pred["depth"][:, :, ::19, ::19].float().square().mean() + pred["pose_enc"].float().square().mean()
            value = value.detach() + (probe - probe.detach())
        return value, terms

    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime, "_objective_source", render_settings(runtime._objective_source)))
        effective = stack.enter_context(runtime.patched_runtime("camera"))
        stack.enter_context(patch.object(gsplat, "rasterization_2dgs", pair))
        stack.enter_context(patch.object(base.tco_module, "compute_tco_losses", loss))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        start = time.perf_counter()
        place(model, path)
        torch.cuda.synchronize()
        placement = time.perf_counter() - start
        policy = ParameterAudit(model, job, profile, initial)
        policy.install()
        stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", policy.clip))
        torch.cuda.manual_seed(seed)
        base.infer_mv_pointclouds_w_priors_v2(supplied, model, cfg, sample_idx=1, scene_name=job["room"].replace("/","-"))
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            canonical = unproject_depth_to_world_points_torch(
                *(captured[name].float() for name in ("depth","extrinsics","intrinsics")))[0].cpu().numpy()
        torch.cuda.synchronize()
        end = time.perf_counter()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        stats = torch.cuda.memory_stats()
        end_free, _ = torch.cuda.mem_get_info()

    pair.check(steps)
    optimizer = policy.finish()
    history = [float(x) for x in losses]
    orders = [p.cpu().tolist() for p in schedule]
    if (len(history) != steps or state["updates"] != steps or len(orders) != steps
            or any(sorted(p) != list(range(5)) for p in orders) or not all(map(math.isfinite, history))
            or sorted(batches) != [(5,)] or not 0 < allocated <= CAP
            or abs(torch.cuda.get_per_process_memory_fraction() - fraction) > 1e-7):
        raise ValueError("Native grouping, optimizer evidence, sampling schedule or memory cap differs")
    if canonical.dtype != np.float32 or canonical.shape != (5,*supplied["images"].shape[-2:],3):
        raise ValueError("Incorrect FP32 final reconstruction")
    arrays = dict(canonical=canonical, **{k:export(v) for k,v in captured.items()})
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite reconstruction")
    output = out / "prediction.npz"
    np.savez_compressed(output, **arrays)
    evidence = None
    if policy.gradients:
        for name, param in policy.lora.items():
            policy.gradients["final/" + name] = param.detach().cpu().numpy().copy()
        gradients = out / "fixed_signed_gradients.npz"
        np.savez_compressed(gradients, **policy.gradients)
        evidence = sha(gradients)
    clock = dict(checkpoint_seconds=loaded-cold, placement_seconds=placement, scene_seconds=end-start, cold_seconds=end-cold)
    result = dict(job)
    result.update(profile=profile, input_sha256=input_id, predictions_sha256=sha(output), output_sha256=signature(arrays),
                  optimizer=optimizer, fixed_signed_gradients_sha256=evidence, completed_adam_updates=state["updates"],
                  native_depth_batches=[list(b) for b in sorted(batches)], raster_shape=list(supplied["images"].shape),
                  readout=dict(dtype="float32", shape=list(canonical.shape)), lora_tensors=state["lora_tensors"],
                  render_calls=pair.calls, camera_checkpoint=True, head_taps=list(TAPS),
                  view_permutations=orders, losses=history, peak_allocated=allocated, peak_reserved=reserved,
                  cap_mib=8960, cap_bytes=CAP, cuda_free_before=free, cuda_free_after=end_free,
                  cuda_device_total=total, allocation_retries=stats.get("num_alloc_retries"), oom_count=stats.get("num_ooms"),
                  host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                               allocator=ALLOCATOR, fraction=fraction, gsplat_sha256=GSPLAT, **effective),
                  rental_compute_usd=0, **clock)
    write_new(out / "complete.json", result)
    print(f"{job['id']} completed {steps} updates and final FP32 readout in {end-start:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--inputs", "--input", dest="packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("Worker writes only to a fresh output directory")
    faulthandler.dump_traceback_later(150, repeat=True)
    try:
        run(job, args.packet, args.out)
    except Exception as exc:
        write_new(args.out / "failed.json", dict(id=job["id"], error_type=type(exc).__name__, message=str(exc),
                                                rental_compute_usd=0))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
