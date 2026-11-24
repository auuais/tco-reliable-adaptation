"""Run a registered full-decoder TCO case in the pinned local container."""

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import faulthandler
import json
import math
import os
import random
import re
import resource
import time
from unittest.mock import patch

import run_experiment as base
import numpy as np
import torch
from utils.geometry import unproject_depth_to_world_points_torch

from run_profiled_d0 import read, sha, write_new
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
from run_tco_one_percent import DATA_ROOT, install_variant, load_case, named_hash, observe_losses, tensor_hash
from tco_frozen_prefix import TAPS, place
from tco_full40_recompute import install_head
from tco_one_percent_protocol import ALLOCATOR, GPU_MEMORY_FRACTION, ROOT, check_case_inputs


STUDY = "tco_full40_recompute_v1"
PRIVATE = ROOT / "reports" / STUDY


class Timeline:
    """Sample device state without retaining activations or touching scored cases."""

    def __init__(self, stream, updates, steps):
        self.stream = stream
        self.state = updates
        self.sample_updates = {0, 1, steps // 2 - 1, steps - 1, steps}
        self.start = time.perf_counter()

    def event(self, phase, edge):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        row = dict(phase=phase, edge=edge, updates_done=self.state["updates"],
                   elapsed_seconds=time.perf_counter() - self.start, allocated=torch.cuda.memory_allocated(),
                   reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
                   peak_reserved=torch.cuda.max_memory_reserved(), device_free=free, device_total=total,
                   host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()

    @contextmanager
    def phase(self, name):
        if self.state["updates"] not in self.sample_updates:
            yield
            return
        self.event(name, "enter")
        try:
            yield
        except torch.OutOfMemoryError:
            self.event(name, "oom")
            raise
        except Exception as exc:
            # Non-reentrant checkpoint intentionally stops replay via this exception.
            if type(exc).__name__ == "_StopRecomputationError":
                self.event(name, "early_stop")
            raise
        else:
            self.event(name, "exit")


def instrument(function, scope, name):
    def wrapped(*args, **kwargs):
        with scope(name):
            return function(*args, **kwargs)
    return wrapped


def setup_model(model, job, state, audit, scope):
    path = install_variant(model, "four", job["seed"], state, audit)
    chunk, method = job["policy"][1:].split("_")
    if method not in ("on", "off") or int(chunk) not in (1, 2, 4, 8):
        raise ValueError("Unknown DPT policy")
    install_head(model, int(chunk), method == "on", scope)
    setup = model.backbone_peft_setup

    def track_inventory(*args, **kwargs):
        optimizer, params = setup(*args, **kwargs)
        if len(params) != 24 * 2 * 4 * 2:
            raise ValueError("Expected rank-four LoRA in every frame and global block")
        state["lora_tensors"] = len(params)
        if "adaptation_clock" in state:
            def mark_last_update(*_):
                if state["updates"] == job["steps"]:
                    state["adaptation_clock"][1].record()

            optimizer.register_step_post_hook(mark_last_update)
        if scope is not None:
            optimizer.step = instrument(optimizer.step, scope, "adam")
        if audit is not None:
            selected = {name: p for name, p in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}
            audit["per_factor_norms"] = {}

            def record_norms(*_):
                if any(p.grad is None for p in selected.values()):
                    raise ValueError("A full-decoder LoRA factor is missing its gradient")
                step = state["updates"]
                values = torch.stack([p.grad.detach().float().norm() for p in selected.values()]).tolist()
                audit["per_factor_norms"][str(step)] = dict(zip(selected, values))
                for name, param in selected.items():
                    state["factor_values"][f"grad/{step}/{name}"] = export(param.grad)

            def record_update(*_):
                for name, param in selected.items():
                    state["factor_values"][f"parameter/{state['updates']}/{name}"] = export(param)

            optimizer.register_step_pre_hook(record_norms)
            optimizer.register_step_post_hook(record_update)
        return optimizer, params

    model.backbone_peft_setup = track_inventory
    if "adaptation_clock" in state:
        forward = model.aggregate_with_patch_tokens

        def mark_first_update(*args, **kwargs):
            if kwargs.get("use_checkpoint") and not state["updates"]:
                state["adaptation_clock"][0].record()
            return forward(*args, **kwargs)

        model.aggregate_with_patch_tokens = mark_first_update
    if scope is not None:
        model.extract_patch_tokens = instrument(model.extract_patch_tokens, scope, "patch_tokens")
        model.forward_original_vggt = instrument(model.forward_original_vggt, scope, "initial_prediction")
        model.aggregate_with_patch_tokens = instrument(model.aggregate_with_patch_tokens, scope, "transformer")
        model.camera_head.forward = instrument(model.camera_head.forward, scope, "camera_head")
        model.depth_head.forward = instrument(model.depth_head.forward, scope, "depth_head")
        fusion = model.depth_head.scratch.refinenet1.out_conv
        fusion.forward = instrument(fusion.forward, scope, "depth.refinenet1.out_conv")
    return path


def warm_device():
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    device = torch.cuda.get_device_name()
    if (device != "NVIDIA GeForce RTX 3080" or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != ALLOCATOR
            or torch.cuda.get_per_process_memory_fraction() != GPU_MEMORY_FRACTION):
        raise ValueError("Use the registered local RTX 3080 and allocator")
    warm = torch.ones((256, 256), device="cuda")
    (warm @ warm).sum().item()
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()
    return device


def run(job, out):
    from gsplat.cuda._backend import _C

    steps, seed, width = (job[key] for key in ("steps", "seed", "width"))
    probe = job.get("probe")
    fixed = probe in ("fixed", "leak")
    audit = {} if fixed or probe == "identity" else None
    if fixed:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    faulthandler.dump_traceback_later(90, repeat=True)
    device = warm_device()
    out.mkdir(parents=True, exist_ok=False)
    supplied, reference = load_case(job, out, width)
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, 0.0003, seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.o3d.utility.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    state = {"updates": 0, "lora_tensors": 0}
    if job["cohort"] in ("screen", "screen5"):
        state["adaptation_clock"] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    factor_values = {}
    if audit is not None:
        state["factor_values"] = factor_values
    loss_history = []
    cold_start = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    loaded = time.perf_counter()
    predictions = {}

    def capture(_model, _inputs, output):
        predictions.update({name: output[name] for name in FIELDS})

    model.register_forward_hook(capture)
    with ExitStack() as stack:
        if job.get("profile"):
            stream = stack.enter_context((out / "profile.jsonl").open("x", encoding="utf-8"))
            scope = Timeline(stream, state, steps).phase
        else:
            scope = None
        path = setup_model(model, job, state, audit, scope)
        _, observed_loss = observe_losses(state, audit, fixed)
        mark = scope if scope is not None else lambda _name: nullcontext()

        def loss(*args, **kwargs):
            with mark("tco_loss_and_render"):
                result, terms = observed_loss(*args, **kwargs)
                loss_history.append(state["last_loss"])
                return result, terms

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        begin = time.perf_counter()
        with mark("placement"):
            place(model, path)
            torch.cuda.synchronize()
        placement = time.perf_counter() - begin
        frozen = tuple(model.named_parameters())
        versions = [param._version for _, param in frozen]
        if audit is not None:
            audit["frozen_start_sha256"] = named_hash(frozen)
        torch.cuda.manual_seed(seed)
        stack.enter_context(patch.object(base.tco_module, "compute_tco_losses", loss))
        if scope is not None:
            stack.enter_context(patch.object(torch.Tensor, "backward",
                                             instrument(torch.Tensor.backward, scope, "backward")))
            stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_",
                                             instrument(torch.nn.utils.clip_grad_norm_, scope, "clipping")))
        with mark("tco_model"):
            pipeline = base.infer_mv_pointclouds_w_priors_v2(
                supplied, model, cfg, sample_idx=1, scene_name=job["scene"].replace("/", "-"))
        with mark("final_fp32_unprojection"), torch.no_grad(), torch.autocast("cuda", enabled=False):
            points = unproject_depth_to_world_points_torch(
                *(predictions[name].float() for name in ("depth", "extrinsics", "intrinsics")))[0].cpu().numpy()
        torch.cuda.synchronize()
        end = time.perf_counter()
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        end_free, _ = torch.cuda.mem_get_info()
        adaptation_clock = state.get("adaptation_clock")
        step_seconds = (adaptation_clock[0].elapsed_time(adaptation_clock[1]) / (1_000 * steps)
                        if adaptation_clock else None)

    if state["updates"] != steps or versions != [param._version for _, param in frozen]:
        raise ValueError("Unexpected update count or base-weight change")
    lora_losses = torch.stack(loss_history).tolist() if loss_history else []
    if len(lora_losses) != steps or not all(map(math.isfinite, lora_losses)):
        raise ValueError("Unexpected loss count or nonfinite TCO loss")
    if audit is not None:
        audit["frozen_end_sha256"] = named_hash(frozen)
        if audit["frozen_end_sha256"] != audit["frozen_start_sha256"]:
            raise ValueError("Frozen weight bytes changed")
        audit["adapter_final"] = {name: tensor_hash(param) for name, param in model.named_parameters()
                                  if name.endswith((".lora_A", ".lora_B"))}
    arrays = dict(canonical=points, pipeline=pipeline, **{name: export(predictions[name]) for name in FIELDS})
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite reconstructed points or predictions")
    inputs = signature(dict(rgb=supplied["images"].numpy(), poses=supplied["extrs"], intrinsics=supplied["intrs"]))
    ref = signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"]))
    metric_start = time.perf_counter()
    metrics = base.evaluate(points, reference, out) if job.get("score") else None
    score_seconds = time.perf_counter() - metric_start if metrics is not None else None
    if metrics is not None and (metrics["valid_points"] <= 0 or not np.any(reference["valid_mask"])):
        raise ValueError("No valid evaluation geometry")
    if metrics is not None or audit is not None:
        np.savez_compressed(out / "prediction.npz", **arrays)
    if factor_values:
        np.savez_compressed(out / "lora_audit.npz", **factor_values)
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=device, allocator=ALLOCATOR,
                   gpu_memory_fraction=GPU_MEMORY_FRACTION, gsplat_sha256=sha(_C.__file__))
    write_new(out / "complete.json", dict(
        **job, input_sha256=inputs, reference_sha256=ref, output_sha256=signature(arrays),
        raster_shape=list(supplied["images"].shape), scene_seconds=end - begin, placement_seconds=placement,
        mean_step_seconds=step_seconds,
        checkpoint_seconds=loaded - cold_start, cold_seconds=end - cold_start, score_seconds=score_seconds,
        cuda_free_before=free, cuda_free_after=end_free, cuda_device_total=total,
        peak_allocated=peak, peak_reserved=reserved, host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        updates=state["updates"], lora_tensors=state["lora_tensors"], head_taps=list(TAPS),
        losses=lora_losses, metrics=metrics, runtime=runtime, audit=audit, rental_compute_usd=0))
    faulthandler.cancel_dump_traceback_later()
    print(f"{job['id']} {end - begin:.2f}s, {peak / 2**30:.3f} GiB peak allocated", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("job")
    cli.add_argument("attempt")
    cli.add_argument("--registration", choices=("initial", "locked"), required=True)
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.job) or not re.fullmatch(r"[a-z0-9]+", args.attempt):
        cli.error("Invalid job or attempt identity")
    filename = f"{STUDY}.json" if args.registration == "initial" else "tco_full40_recompute_locked_v1.json"
    plan = read(ROOT / "manifests" / filename)
    job = next(item for item in plan["jobs"] if item["id"] == args.job)
    out = PRIVATE / "cases" / args.job / args.attempt / "case"
    if job.get("profile") == "fixture":
        from tco_full40_recompute import fixture
        out.mkdir(parents=True, exist_ok=False)
        torch.set_num_threads(1)
        result = fixture()
        write_new(out / "complete.json", dict(**job, fixture=result, rental_compute_usd=0))
        return
    check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
    run(job, out)


if __name__ == "__main__":
    main()
