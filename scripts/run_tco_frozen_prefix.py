"""Run one registered TCO variant with the original loss and a common evaluator."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import resource
import time
from pathlib import Path

import run_experiment as base
import numpy as np
import torch
from utils.geometry import unproject_depth_to_world_points_torch

from tco_frozen_prefix import install, place
from tco_frozen_prefix_protocol import ARMS, FRAMES, MANIFEST, PRIVATE, ROOT, SCENES
from run_profiled_d0 import sha


FIELDS = ("depth", "depth_conf", "extrinsics", "intrinsics")
CHECKPOINT = "/data/checkpoints/VGGT-1B"


def export(tensor):
    return tensor.detach().cpu().float().numpy().copy()


def signature(arrays):
    sha = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        sha.update(repr((name, value.shape, value.dtype.str)).encode())
        sha.update(value.tobytes())
    return sha.hexdigest()


def observe(model, arrays, path, fixed_gradient):
    original_aggregate = model.aggregate_with_patch_tokens
    original_extract = model.extract_patch_tokens
    original_setup = model.backbone_peft_setup
    original_loss = base.tco_module.compute_tco_losses
    if path is not None and path.start:
        def boundary(tokens):
            count = sum(name.startswith("boundary") for name in arrays)
            arrays[f"boundary{count}"] = export(tokens)

        path.boundary_observer = boundary

    def aggregate(*args, **kwargs):
        taps, offset = original_aggregate(*args, **kwargs)
        key = f"tap{sum(name.startswith('tap') for name in arrays)//4}"
        for index in (4, 11, 17, 23):
            arrays[f"{key}.{index}"] = export(taps[index])
        print(f"Captured {key} at {torch.cuda.max_memory_allocated()/2**30:.3f} GiB", flush=True)
        return taps, offset

    def extract(images):
        print("Extracting full-resolution image tokens", flush=True)
        result = original_extract(images)
        print("Image tokens ready and patch encoder offloaded", flush=True)
        return result

    def setup(*args, **kwargs):
        opt, params = original_setup(*args, **kwargs)

        def retain(grad, name):
            arrays["grad."+name] = export(grad)

        for name, param in model.named_parameters():
            if name.endswith((".lora_A", ".lora_B")):
                param.register_hook(lambda grad, name=name: retain(grad, name))

        def save_after(*_):
            for name, param in model.named_parameters():
                if name.endswith((".lora_A", ".lora_B")):
                    arrays["adam."+name] = export(param)

        opt.register_step_post_hook(save_after)
        return opt, params

    def loss(*args, **kwargs):
        arrays["sampler.before"] = torch.cuda.get_rng_state().cpu().numpy()
        result = original_loss(*args, **kwargs)
        arrays["sampler.after"] = torch.cuda.get_rng_state().cpu().numpy()
        arrays["loss.total"] = export(result[0])
        for name, term in result[1].items():
            if isinstance(term, torch.Tensor):
                arrays["loss."+name] = export(term)
        if fixed_gradient:
            outputs = kwargs["predictions"]
            # Exercise both heads with deterministic adjoints while checking the real loss value above.
            depth = outputs["depth"][:, :, ::19, ::19].float()
            pose = outputs["pose_enc"].float()
            probe = depth.square().mean() + pose.square().mean()
            return result[0].detach() + (probe-probe.detach()), result[1]
        return result

    model.aggregate_with_patch_tokens = aggregate
    model.extract_patch_tokens = extract
    model.backbone_peft_setup = setup
    base.tco_module.compute_tco_losses = loss


def run(job, out, engineering, fixed_gradient=False):
    scene, seed, arm, steps = (job[k] for k in ("scene", "seed", "arm", "steps"))
    out.mkdir(parents=True, exist_ok=False)
    if fixed_gradient:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_name()
    if "RTX 3080" not in gpu:
        raise ValueError("Use the local RTX 3080")
    from gsplat.cuda._backend import _C

    # Exercise CUDA dispatch without a measured scene or live model.
    warm = torch.ones((256, 256), device="cuda")
    (warm @ warm).sum().item()
    del warm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dataset = base.SevenScenes(SEVENSCENES_DIR="/data/data/7scenes", split="test",
                               load_img_size=518, cache_file=str(ROOT/"environment/development_dataset_cache_v7.npy"))
    data = dataset.get_data(sequence_name=scene, ids=FRAMES)
    if tuple(data["images"].shape) != (5, 3, 392, 518):
        raise ValueError(f"Unexpected full-resolution input {data['images'].shape}")
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, .0003, seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.o3d.utility.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    cold = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    path = install(model, arm, seed)
    captures = {}
    if engineering:
        observe(model, captures, path, fixed_gradient)
    values = {}

    def capture(_mod, _inputs, output):
        values.update({name: output[name] for name in FIELDS})

    model.register_forward_hook(capture)
    torch.cuda.synchronize()
    prior_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin_free, device_total = torch.cuda.mem_get_info()
    begin = time.perf_counter()
    place(model, path)
    torch.cuda.synchronize()
    placed = time.perf_counter()
    if engineering:
        print(f"Device placement completed in {placed-begin:.2f}s", flush=True)
    versions = {name: (p, p._version) for name, p in model.named_parameters()}
    # Model loading and adapter setup must not move the renderer's RNG draw schedule.
    torch.cuda.manual_seed(seed)
    permitted = {"images": data["images"], "extrs": data["extrs"], "intrs": data["intrs"]}
    output_context = contextlib.nullcontext() if engineering else contextlib.redirect_stdout(io.StringIO())
    with output_context:
        pipeline = base.infer_mv_pointclouds_w_priors_v2(permitted, model, cfg, sample_idx=1,
                                                         scene_name=scene.replace("/", "-"))
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        canonical = unproject_depth_to_world_points_torch(
            *(values[name].float() for name in ("depth", "extrinsics", "intrinsics")))[0].cpu().numpy()
    torch.cuda.synchronize()
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    end_free, _ = torch.cuda.mem_get_info()
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if any(p._version != version for p, version in versions.values()):
        raise ValueError("Frozen base parameters changed")
    if not np.isfinite(canonical).all():
        raise ValueError("Nonfinite prediction")
    inputs = signature(dict(images=data["images"].numpy(), mask=data["valid_mask"], ref=data["pointclouds"],
                            intrinsics=data["intrs"], extrinsics=data["extrs"]))
    heads = {name: export(values[name]) for name in FIELDS}
    arrays = dict(canonical=canonical, pipeline=pipeline, **heads)
    if engineering:
        arrays.update(captures)
    save_prediction = np.savez if engineering else np.savez_compressed
    save_prediction(out/"prediction.npz", **arrays)
    if engineering:
        scores = None
    else:
        scores = base.evaluate(canonical, data, out)
    cache = None if path is None else dict(builds=path.builds, hits=path.hits,
                                         prefix_stages=path.prefix_stages, bytes_on_gpu=path.cached_bytes)
    if path and path.start:
        calls = steps+2 if steps else 1
        expected_prefix_runs = path.builds if path.reuse else calls
        if (path.prefix_stages != path.start*expected_prefix_runs or
                path.reuse and path.builds+path.hits != calls or
                not path.reuse and (path.builds or path.hits)):
            raise ValueError("Measured prefix execution does not match the configured reuse")
    base.dump(out/"complete.json", dict(**job, input_sha256=inputs, output_sha256=signature(arrays),
              scene_seconds=end-begin, forward_seconds=end-placed, placement_seconds=placed-begin,
              cold_seconds=end-cold, peak_allocated=peak, peak_reserved=reserved,
              cold_peak_allocated=max(prior_peak, peak), host_peak_bytes=host_peak,
              cuda_free_before=begin_free, cuda_free_after=end_free, cuda_device_total=device_total,
              gpu=gpu, torch=torch.__version__,
              cuda=torch.version.cuda, gsplat=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
              cache=cache, metrics=scores, rental_compute_usd=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--attempt", default="attempt01")
    parser.add_argument("--engineering", choices=ARMS)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--fixed-gradient", action="store_true")
    args = parser.parse_args()
    if not args.job.replace("_", "").isalnum() or not args.attempt.isalnum():
        parser.error("Use alphanumeric job and attempt identifiers")
    if args.engineering:
        job = dict(id=args.job, arm=args.engineering, scene=SCENES[0], seed=42, steps=args.steps)
        job["fixed_gradient"] = args.fixed_gradient
    else:
        if args.fixed_gradient:
            parser.error("Fixed gradients are an engineering diagnostic only")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        job = next(item for item in manifest["jobs"] if item["id"] == args.job)
        for name, digest in manifest["input_source_sha256"].items():
            if name.startswith(job["scene"]+"/") and sha(Path("/data/data/7scenes")/name) != digest:
                raise ValueError(f"Registered source bytes changed: {name}")
    mode = "engineering" if args.engineering else "cases"
    run(job, PRIVATE/mode/args.job/args.attempt, bool(args.engineering), args.fixed_gradient)


if __name__ == "__main__":
    main()
