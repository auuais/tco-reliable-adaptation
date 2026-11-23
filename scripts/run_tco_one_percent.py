"""Run one registered TCO efficiency case inside the pinned local Docker image."""

import argparse
import contextlib
import faulthandler
import hashlib
import io
import os
from pathlib import Path
import random
import re
import resource
import time

import run_experiment as base
import numpy as np
import torch
from utils.geometry import unproject_depth_to_world_points_torch

from run_profiled_d0 import read, sha, write_new
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
from tco_frozen_prefix import TAPS, install, place
from tco_one_percent_geometry import BoundaryCorrection, correspondences, shuffle_edges
from tco_one_percent_protocol import ALLOCATOR, CONFIRM_MANIFEST, MANIFEST, METADATA, PRIVATE, ROOT, TIME_MANIFEST
from tco_one_percent_protocol import check_case_inputs


GEOMETRY_ARMS = ("own", "geometry", "shuffle")
DATA_ROOT = Path("/data/data")


def tensor_hash(tensor):
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(data).hexdigest()


def named_hash(parameters):
    digest = hashlib.sha256()
    for name, param in parameters:
        digest.update(repr((name, tuple(param.shape), str(param.dtype))).encode())
        digest.update(bytes.fromhex(tensor_hash(param)))
    return digest.hexdigest()


def install_variant(model, arm, seed, state, audit):
    baseline = {"own": "cached12", "geometry": "cached12", "shuffle": "cached12",
                "cached14": "cached12", "uncached14": "uncached12"}.get(arm, arm)
    path = install(model, baseline, seed)
    aggregator = model.aggregator
    if arm in ("cached14", "uncached14"):
        # The prior installer's closure consults this target function at setup time.
        path.start = 10
        model.get_lora_target_modules = lambda: [*aggregator.frame_blocks[10:], *aggregator.global_blocks[10:]]
    if arm in GEOMETRY_ARMS:
        setup_boundary(model, path, seed, arm, state, audit)
    if audit is not None:
        observe_forwards(model, path, audit)
    setup = model.backbone_peft_setup

    def track_optimizer(*args, **kwargs):
        optimizer, params = setup(*args, **kwargs)
        all_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
        if all_ids != [id(param) for param in params] or len(set(all_ids)) != len(all_ids):
            raise ValueError("Optimization and clipping do not use the same parameters")
        active = set(all_ids)
        if active != {id(p) for name, p in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}:
            raise ValueError("Unexpected trainable parameter inventory")
        if audit is not None:
            initial = {name: p for name, p in model.named_parameters() if id(p) in active}
            audit["adapter_initial"] = {name: tensor_hash(p) for name, p in initial.items()}
            audit["gradients"] = {}
            for name, parameter in initial.items():
                def record(grad, name=name):
                    step = str(state["updates"])
                    audit["gradients"].setdefault(step, {})[name] = dict(
                        sha256=tensor_hash(grad),
                        nonzero=(bool(torch.count_nonzero(grad)) if name.startswith("boundary_correction.") else None))
                parameter.register_hook(record)

        def update(*_):
            state["updates"] += 1

        optimizer.register_step_post_hook(update)
        return optimizer, params

    model.backbone_peft_setup = track_optimizer
    return path


def setup_boundary(model, path, seed, arm, state, audit):
    original = model.forward_original_vggt
    setup = model.backbone_peft_setup
    stage = path.stage
    aggregator = model.aggregator
    adapter = None

    def read_priors(_model, _args, kwargs):
        state["priors"] = kwargs["gt_extrinsics"], kwargs["gt_intrinsics"]

    model.register_forward_pre_hook(read_priors, with_kwargs=True)

    def initial(images):
        tokens, predictions = original(images)
        extrinsics, intrinsics = state.pop("priors")
        edges, info = correspondences(predictions["depth"][0, ..., 0], extrinsics[0],
                                      predictions["extrinsics"][0], intrinsics[0], aggregator.patch_size)
        if arm == "shuffle" and info["fallback"] is None:
            edges, summary = shuffle_edges(edges, seed + 3701)
            info.update(summary)
        state["matches"] = info
        if audit is not None:
            audit["donor_sha256"] = tensor_hash(edges)
        if info["fallback"] is None:
            state["edges"] = edges
        return tokens, predictions

    def setup_with_boundary(*args, **kwargs):
        nonlocal adapter
        optimizer, params = setup(*args, **kwargs)
        edges = state.pop("edges", None)
        if edges is not None:
            channels = path.source_tokens.shape[-1]
            adapter = BoundaryCorrection(channels, edges, arm, seed + 6101)
            model.boundary_correction = adapter
            factors = list(adapter.parameters())
            optimizer.add_param_group({"params": factors})
            params.extend(factors)
            state["matches"].update(
                learned_parameters=sum(p.numel() for p in factors),
                buffer_bytes=sum(t.numel() * t.element_size() for t in adapter.buffers()))
        return optimizer, params

    def adapt_stage(index, tokens, pos, outputs, batch, views, checkpoint):
        if adapter is not None and index == path.start:
            if batch != 1:
                raise ValueError("A boundary adapter belongs to exactly one scene")
            corrected = adapter(tokens, aggregator.patch_start_idx)
            if audit is not None:
                audit.setdefault("boundary_forwards", []).append(dict(
                    changed=not torch.equal(tokens, corrected), grad_enabled=torch.is_grad_enabled()))
            tokens = corrected
        return stage(index, tokens, pos, outputs, batch, views, checkpoint)

    model.forward_original_vggt = initial
    model.backbone_peft_setup = setup_with_boundary
    path.stage = adapt_stage


def observe_forwards(model, path, audit):
    original = model.forward_original_vggt

    def track_original(images):
        tokens, predictions = original(images)
        audit["initial_prediction"] = {name: tensor_hash(predictions[name]) for name in FIELDS}
        return tokens, predictions

    model.forward_original_vggt = track_original
    if path is not None:
        def aggregate(*args, **kwargs):
            outputs, offset = path(*args, **kwargs)
            audit.setdefault("taps", []).append({str(tap): tensor_hash(outputs[tap]) for tap in TAPS})
            return outputs, offset

        model.aggregate_with_patch_tokens = aggregate


def observe_losses(state, audit, fixed_gradient):
    original = base.tco_module.compute_tco_losses

    def loss(*args, **kwargs):
        if audit is not None:
            before = torch.cuda.get_rng_state()
        result, terms = original(*args, **kwargs)
        if audit is not None:
            after = torch.cuda.get_rng_state()
            generator = torch.Generator(device="cuda").set_state(before)
            group = torch.randperm(kwargs["images"].shape[1], device="cuda", generator=generator).tolist()
            if not torch.equal(generator.get_state(), after):
                raise ValueError("Render sampling no longer matches the pinned TCO permutation")
            audit.setdefault("render_orders", []).append(group)
            audit.setdefault("update_predictions", []).append(
                {name: tensor_hash(kwargs["predictions"][name]) for name in FIELDS})
        value = result.detach()
        state.setdefault("first_loss", value)
        state["last_loss"] = value
        if fixed_gradient:
            outputs = kwargs["predictions"]
            adjoint = outputs["depth"][:, :, ::19, ::19].float().square().mean()
            adjoint = adjoint + outputs["pose_enc"].float().square().mean()
            result = value + (adjoint - adjoint.detach())
        return result, terms

    return original, loss


def load_case(job, out, width):
    scene, frames = job["scene"], job["frames"]
    if job["dataset"] == "7scenes":
        metadata = METADATA
        if job["phase"] == "confirmation":
            metadata = out / "7scenes_metadata.npy"
            seq_dir = DATA_ROOT / "7scenes" / scene
            np.save(metadata, {scene: sum(1 for _ in seq_dir.glob("*.color.png"))})
        dataset = base.SevenScenes(SEVENSCENES_DIR=str(DATA_ROOT / "7scenes"), split="test",
                                   load_img_size=width, cache_file=str(metadata))
    elif job["dataset"] == "eth3d":
        from datasets.eth3d import ETH3D
        dataset = ETH3D(ETH3D_DIR=str(DATA_ROOT / "eth3d"), load_img_size=width,
                        cache_file=str(out / "eth3d_metadata.npy"))
    else:
        raise ValueError(f"Unknown dataset: {job['dataset']}")
    data = dataset.get_data(sequence_name=scene, ids=frames)
    images = data["images"]
    if (images.ndim != 4 or tuple(images.shape[:2]) != (len(frames), 3)
            or tuple(images.shape[-2:]) != data["valid_mask"].shape[-2:]):
        raise ValueError(f"Expected matched RGB and reference rasters, got {images.shape}")
    if job["dataset"] == "7scenes" and width == 518 and tuple(images.shape[-2:]) != (392, 518):
        raise ValueError("The 7-Scenes crop changed")
    if "images_expected" in job and [Path(path).name for path in data["image_paths"]] != job["images_expected"]:
        raise ValueError("The ETH3D loader selected different images")
    supplied = {name: data[name] for name in ("images", "extrs", "intrs")}
    reference = {name: data[name] for name in ("images", "valid_mask", "pointclouds")}
    if job.get("probe") == "leak":
        reference["valid_mask"] = np.zeros_like(reference["valid_mask"])
        reference["pointclouds"] = np.full_like(reference["pointclouds"], 1)
    return supplied, reference


def run(job, out, width):
    arm, seed, steps = (job[field] for field in ("arm", "seed", "steps"))
    probe = job.get("probe")
    fixed = probe in ("fixed", "leak")
    audit = {} if fixed or probe in ("repeat", "identity") else None
    if probe:
        faulthandler.dump_traceback_later(90, repeat=True)
    if fixed:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    device = torch.cuda.get_device_name()
    if device != "NVIDIA GeForce RTX 3080" or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("Use the registered local RTX 3080 and allocator")
    from gsplat.cuda._backend import _C

    warm = torch.ones((256, 256), device="cuda")
    (warm @ warm).sum().item()
    del warm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=False)
    supplied, reference = load_case(job, out, width)
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, 0.0003, seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.o3d.utility.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    state = {"updates": 0}
    torch.cuda.reset_peak_memory_stats()
    cold_start = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    if probe:
        print(f"Checkpoint ready on CPU for {width}px input", flush=True)
    path = install_variant(model, arm, seed, state, audit)
    predictions = {}

    def capture(_model, _inputs, output):
        predictions.update({name: output[name] for name in FIELDS})

    model.register_forward_hook(capture)
    before_loss, loss = observe_losses(state, audit, fixed)
    torch.cuda.synchronize()
    cold_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    begin = time.perf_counter()
    place(model, path)
    torch.cuda.synchronize()
    placed = time.perf_counter()
    if probe:
        print(f"CUDA placement took {placed - begin:.2f}s", flush=True)
    frozen = tuple(model.named_parameters())
    versions = [param._version for _, param in frozen]
    if audit is not None:
        audit["frozen_start_sha256"] = named_hash(frozen)
        print("Frozen weights hashed. Entering TCO.", flush=True)
    torch.cuda.manual_seed(seed)
    log = io.StringIO()
    base.tco_module.compute_tco_losses = loss
    try:
        with contextlib.redirect_stdout(log):
            pipeline = base.infer_mv_pointclouds_w_priors_v2(
                supplied, model, cfg, sample_idx=1, scene_name=job["scene"].replace("/", "-"))
    except BaseException:
        print(log.getvalue(), end="", flush=True)
        raise
    finally:
        base.tco_module.compute_tco_losses = before_loss
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        points = unproject_depth_to_world_points_torch(
            *(predictions[name].float() for name in ("depth", "extrinsics", "intrinsics")))[0].cpu().numpy()
    torch.cuda.synchronize()
    end = time.perf_counter()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    end_free, _ = torch.cuda.mem_get_info()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(log.getvalue(), end="", flush=True)
    if (state["updates"] != steps or versions != [param._version for _, param in frozen]
            or not np.isfinite(points).all()):
        raise ValueError("Unexpected update count, base-weight change or nonfinite output")
    assembly = None
    if path is not None and path.start:
        calls = steps + 2 if steps else 1
        if (path.reuse and path.builds + path.hits != calls
                or path.prefix_stages != path.start * (path.builds if path.reuse else calls)):
            raise ValueError("Wrong number of cache hits or stage executions")
        assembly = dict(start=path.start, builds=path.builds, hits=path.hits,
                        prefix_stages=path.prefix_stages, bytes_on_gpu=path.cached_bytes)
    inputs = signature(dict(rgb=supplied["images"].numpy(), poses=supplied["extrs"],
                            intrinsics=supplied["intrs"]))
    ref_hash = signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"]))
    if audit is not None:
        audit["frozen_end_sha256"] = named_hash(frozen)
        if audit["frozen_end_sha256"] != audit["frozen_start_sha256"]:
            raise ValueError("Frozen weight bytes changed")
        audit["adapter_final"] = {name: tensor_hash(param) for name, param in model.named_parameters()
                                  if name.endswith((".lora_A", ".lora_B"))}
    arrays = dict(canonical=points, pipeline=pipeline, **{name: export(predictions[name]) for name in FIELDS})
    outputs = signature(arrays)
    if probe not in ("memory", "timing"):
        np.savez_compressed(out / "prediction.npz", **arrays)
    metrics = base.evaluate(points, reference, out) if probe is None else None
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=device, allocator=ALLOCATOR,
                   gsplat_sha256=sha(_C.__file__))
    losses = ({key: float(state[key]) for key in ("first_loss", "last_loss")} if steps else None)
    write_new(out / "complete.json", dict(
        **job, input_sha256=inputs, reference_sha256=ref_hash, output_sha256=outputs,
        raster_shape=list(supplied["images"].shape), scene_seconds=end - begin,
        placement_seconds=placed - begin, cold_seconds=end - cold_start,
        cuda_free_before=free, cuda_free_after=end_free, cuda_device_total=total,
        peak_allocated=peak, peak_reserved=reserved, host_peak_bytes=rss, cold_peak_allocated=max(cold_peak, peak),
        cache=assembly, matches=state.get("matches"), updates=state["updates"], losses=losses,
        metrics=metrics, runtime=runtime, audit=audit, rental_compute_usd=0))
    print(f"{job['id']} {end - begin:.2f}s, {peak / 2**30:.3f} GiB peak allocated", flush=True)
    if probe:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("job")
    cli.add_argument("attempt")
    cli.add_argument("--registration", choices=("development", "time", "confirmation"), default="development")
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.job) or not re.fullmatch(r"[a-z0-9]+", args.attempt):
        cli.error("Invalid job or attempt identity")
    plan = read({"development": MANIFEST, "time": TIME_MANIFEST, "confirmation": CONFIRM_MANIFEST}
                [args.registration])
    job = next(item for item in [*plan["jobs"], *plan.get("probes", [])] if item["id"] == args.job)
    check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
    width = plan["fixed_adjoint_width"] if job.get("probe") in ("fixed", "leak") else 518
    run(job, PRIVATE / "cases" / args.job / args.attempt, width)
