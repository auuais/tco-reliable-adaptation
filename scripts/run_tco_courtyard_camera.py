"""Execute a registered Courtyard trace or native-depth camera study case."""

import argparse
from contextlib import ExitStack, nullcontext
import faulthandler
import math
import os
import random
import re
import resource
import time
from unittest.mock import patch

import numpy as np
import torch
from utils.geometry import unproject_depth_to_world_points_torch

from run_profiled_d0 import read, sha, write_new
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
from run_tco_full40_recompute import Timeline, base, instrument, setup_model, warm_device
from run_tco_one_percent import DATA_ROOT, load_case, named_hash, observe_losses, tensor_hash
from tco_courtyard_camera import cpu_fixture, install_camera
from tco_courtyard_camera_protocol import CAMERA, PRIVATE, REGISTRATIONS, ROOT
from tco_frozen_prefix import TAPS, place
from tco_one_percent_protocol import ALLOCATOR, GPU_MEMORY_FRACTION, check_case_inputs


def successful_case(identifier):
    attempts = PRIVATE / "launches" / identifier
    matches = []
    for terminal in sorted(attempts.glob("attempt*/terminal.json")):
        receipt = read(terminal)
        if receipt["verdict"] == "completed":
            folder = PRIVATE / "cases" / identifier / terminal.parent.name / "case"
            if sha(folder / "complete.json") != receipt["complete_sha256"]:
                raise ValueError(f"Invalid trace completion hash for {identifier}")
            matches.append(folder)
    if len(matches) != 1:
        raise ValueError(f"Expected one successful trace for {identifier}")
    return matches[0]


def depth_layouts(model, expected):
    """Record actual native call spans during forward without storing tensors."""
    head = model.depth_head
    native_forward, native_chunk = head.forward, head._forward_impl
    layouts = set()
    current = []

    def observe_chunk(taps, images, patch_start, first=None, last=None):
        current.append((0 if first is None else first, images.shape[1] if last is None else last))
        return native_chunk(taps, images, patch_start, first, last)

    def observe_forward(*args, **kwargs):
        current.clear()
        result = native_forward(*args, **kwargs)
        spans = tuple(current)
        if spans != expected:
            raise ValueError(f"Unexpected native depth grouping {spans}, expected {expected}")
        layouts.add(tuple(end - start for start, end in spans))
        return result

    head._forward_impl = observe_chunk
    head.forward = observe_forward
    return layouts


def setup_execution(model, job, state, audit, scope, trace):
    camera = job["policy"] == CAMERA
    views = len(job["frames"])
    native = camera or job["policy"] == "c8_off"
    if camera:
        install_camera(model, True, scope)
    path = setup_model(model, dict(job, policy="c8_off") if camera else job, state, audit, scope)
    chunk = 8 if native else 1
    batches = depth_layouts(model, tuple((start, min(start + chunk, views)) for start in range(0, views, chunk)))
    if trace is not None:
        trace.bind(model)
        setup = model.backbone_peft_setup

        def setup_trace(*args, **kwargs):
            optimizer, params = setup(*args, **kwargs)
            optimizer.register_step_post_hook(lambda *_: trace.after_step())
            return optimizer, params

        model.backbone_peft_setup = setup_trace
    return path, batches


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.o3d.utility.random.seed(seed)
    torch.backends.cudnn.benchmark = False


def readout(job, out, supplied, reference):
    from tco_courtyard_trace import replay_snapshots

    warm_device()
    seed_model(job["seed"])
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = job["steps"], 0.0003, job["seed"]
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    path, batches = setup_execution(model, job, {"updates": 0, "lora_tensors": 0}, None, None, None)
    place(model, path)
    origin = successful_case(job["trace_job"])
    replay_snapshots(model, supplied, origin, out, cfg)
    inputs = signature(dict(rgb=supplied["images"].numpy(), poses=supplied["extrs"], intrinsics=supplied["intrs"]))
    ref = signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"]))
    write_new(out / "complete.json", dict(**job, updates=0, source_trace_sha256=sha(origin / "trace.json"),
                                        readout_sha256=sha(out / "readout.json"), input_sha256=inputs, reference_sha256=ref,
                                        depth_batches=sorted(batches), rental_compute_usd=0))


def run(job, out, supplied, reference):
    from gsplat.cuda._backend import _C

    seed, steps = job["seed"], job["steps"]
    probe = job.get("probe")
    fixed = probe in ("fixed", "leak")
    audit = {} if fixed or probe == "identity" else None
    trace = None
    if job["cohort"] == "trace":
        from tco_courtyard_trace import CourtyardTrace

        trace = CourtyardTrace(out)
    device = warm_device()
    seed_model(seed)
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, 0.0003, seed
    state = {"updates": 0, "lora_tensors": 0}
    if job["cohort"] in ("gate", "screen5") and not job.get("profile"):
        state["adaptation_clock"] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    if audit is not None:
        state["factor_values"] = {}
    loss_history = []
    cold_start = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    loaded = time.perf_counter()
    predictions = {}
    output_fields = FIELDS if trace is None else (*FIELDS, "pose_enc")
    model.register_forward_hook(lambda _, inputs, output: predictions.update(
        {name: output[name] for name in output_fields}))

    with ExitStack() as stack:
        if job.get("profile") == "timeline":
            stream = stack.enter_context((out / "profile.jsonl").open("x", encoding="utf-8"))
            scope = Timeline(stream, state, steps).phase
        else:
            scope = None
        mark = scope or (lambda _: nullcontext())
        path, batches = setup_execution(model, job, state, audit, scope, trace)
        _, observed_loss = observe_losses(state, audit, fixed)

        def loss(*args, **kwargs):
            with mark("tco_loss_and_render"):
                rng_before = torch.cuda.get_rng_state() if trace is not None else None
                result, terms = observed_loss(*args, **kwargs)
                if audit is not None:
                    audit.setdefault("pose_encoding_hashes", []).append(tensor_hash(kwargs["predictions"]["pose_enc"]))
                if trace is not None:
                    trace.loss(outputs=kwargs["predictions"], terms=terms, images=kwargs["images"],
                               total=result, rng_before=rng_before)
                loss_history.append(state["last_loss"])
                return result, terms

        def clip(*args, **kwargs):
            if trace is not None:
                trace.before_clip()
            with mark("clipping"):
                return native_clip(*args, **kwargs)

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
        if trace is not None or scope is not None:
            native_clip = torch.nn.utils.clip_grad_norm_
            stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", clip))

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
        clock = state.get("adaptation_clock")
        step_seconds = clock[0].elapsed_time(clock[1]) / (1000 * steps) if clock else None

    losses = torch.stack(loss_history).tolist() if loss_history else []
    if state["updates"] != steps or len(losses) != steps or not all(map(math.isfinite, losses)):
        raise ValueError("Invalid optimizer update or loss history")
    if versions != [param._version for _, param in frozen]:
        raise ValueError("Unexpected base-weight mutation")
    if audit is not None:
        audit["frozen_end_sha256"] = named_hash(frozen)
        if audit["frozen_end_sha256"] != audit["frozen_start_sha256"]:
            raise ValueError("Frozen weight bytes changed")
        audit["adapter_final"] = {name: tensor_hash(param) for name, param in model.named_parameters()
                                  if name.endswith((".lora_A", ".lora_B"))}
    arrays = dict(canonical=points, pipeline=pipeline, **{name: export(predictions[name]) for name in FIELDS})
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite reconstruction")
    if trace is not None:
        trace.final(predictions, points)
    inputs = signature(dict(rgb=supplied["images"].numpy(), poses=supplied["extrs"], intrinsics=supplied["intrs"]))
    ref = signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"]))
    metric_start = time.perf_counter()
    metrics = base.evaluate(points, reference, out) if job.get("score") else None
    score_seconds = time.perf_counter() - metric_start if metrics is not None else None
    if metrics is not None and metrics["valid_points"] <= 0:
        raise ValueError("No valid evaluation geometry")
    if metrics is not None or audit is not None:
        np.savez_compressed(out / "prediction.npz", **arrays)
    if state.get("factor_values"):
        np.savez_compressed(out / "lora_audit.npz", **state["factor_values"])
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=device, allocator=ALLOCATOR,
                   gpu_memory_fraction=GPU_MEMORY_FRACTION, gsplat_sha256=sha(_C.__file__))
    write_new(out / "complete.json", dict(
        **job, input_sha256=inputs, reference_sha256=ref, output_sha256=signature(arrays),
        raster_shape=list(supplied["images"].shape), native_depth_batches=sorted(batches),
        camera_checkpoint=job["policy"] == CAMERA,
        scene_seconds=end - begin, placement_seconds=placement, mean_step_seconds=step_seconds,
        checkpoint_seconds=loaded - cold_start, cold_seconds=end - cold_start, score_seconds=score_seconds,
        cuda_free_before=free, cuda_free_after=end_free, cuda_device_total=total,
        peak_allocated=peak, peak_reserved=reserved, host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        updates=state["updates"], lora_tensors=state["lora_tensors"], head_taps=list(TAPS),
        losses=losses, metrics=metrics, runtime=runtime, audit=audit,
        trace_sha256=sha(out / "trace.json") if trace is not None else None, rental_compute_usd=0))
    print(f"{job['id']} {end - begin:.2f}s, {peak / 2**30:.3f} GiB peak allocated", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("job")
    cli.add_argument("attempt")
    cli.add_argument("--registration", choices=tuple(REGISTRATIONS), required=True)
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.job) or not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        cli.error("Invalid job or attempt identity")
    plan = read(REGISTRATIONS[args.registration])
    job = next(item for item in plan["jobs"] if item["id"] == args.job)
    out = PRIVATE / "cases" / args.job / args.attempt / "case"
    out.mkdir(parents=True, exist_ok=False)
    if job.get("profile") == "fixture":
        torch.set_num_threads(1)
        write_new(out / "complete.json", dict(**job, fixture=cpu_fixture(), rental_compute_usd=0))
        return
    check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
    supplied, reference = load_case(job, out, job["width"])
    if job.get("profile") == "archived":
        from score_tco_courtyard_trace import rescore_archived

        result = rescore_archived(out, supplied, reference)
        write_new(out / "complete.json", dict(**job, archived=result, rental_compute_usd=0))
        return
    if job.get("profile") == "diagnosis":
        from score_tco_courtyard_trace import score_traces

        trace_cases, readout_cases = (
            {case["id"]: successful_case(case["id"]) for case in plan["jobs"] if case["cohort"] == cohort}
            for cohort in ("trace", "readout"))
        result = score_traces(out, supplied, reference, trace_cases, readout_cases, successful_case("trace_clean_777"))
        write_new(out / "complete.json", dict(**job, diagnosis=result, rental_compute_usd=0))
        return
    if job.get("probe") in ("fixed", "leak"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        if job["cohort"] == "readout":
            readout(job, out, supplied, reference)
        else:
            run(job, out, supplied, reference)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
