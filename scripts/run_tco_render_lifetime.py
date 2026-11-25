"""Execute one immutable case of the local TCO render-lifetime study."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import faulthandler
import json
import math
import os
from pathlib import Path
import re
import resource
import time
from unittest.mock import patch

import numpy as np
import torch

from run_profiled_d0 import read, sha, write_new
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
import run_tco_full40_recompute as full40
from run_tco_courtyard_camera import depth_layouts, seed_model
from run_tco_one_percent import DATA_ROOT, load_case, named_hash, observe_losses, tensor_hash
from tco_courtyard_camera import install_camera
from tco_frozen_prefix import TAPS, place
from tco_one_percent_protocol import ALLOCATOR, GPU_MEMORY_FRACTION, check_case_inputs
from tco_render_lifetime import DEPTH_RECOMPUTE, LifetimeMonitor, patched_runtime
from tco_render_lifetime_protocol import PRIVATE, REGISTRATIONS, ROOT
from utils.geometry import unproject_depth_to_world_points_torch


base = full40.base


class LifetimeTimeline(full40.Timeline):
    def __init__(self, stream, state, steps, monitor):
        super().__init__(stream, state, steps)
        self.monitor = monitor

    def event(self, phase, edge):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        host_pages = int(Path("/proc/self/statm").read_text().split()[1])
        row = dict(
            phase=phase, edge=edge, updates_done=self.state["updates"],
            elapsed_seconds=time.perf_counter() - self.start,
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            device_free=free, device_total=total, host_rss_bytes=host_pages * os.sysconf("SC_PAGE_SIZE"),
            host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            inventory=self.monitor.inventory(),
        )
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()


def source_case(identifier, filename):
    attempts = PRIVATE / "launches" / identifier
    successes = []
    for receipt_path in attempts.glob("attempt*/terminal.json"):
        receipt = read(receipt_path)
        if receipt["job"] != identifier or receipt["attempt"] != receipt_path.parent.name:
            raise ValueError(f"Invalid source receipt for {identifier}")
        if receipt["verdict"] == "completed":
            successes.append(receipt)
    if len(successes) != 1:
        raise ValueError(f"Expected one completed source case for {identifier}")
    receipt = successes[0]
    folder = PRIVATE / "cases" / identifier / receipt["attempt"] / "case"
    complete = folder / "complete.json"
    if sha(complete) != receipt["complete_sha256"]:
        raise ValueError(f"Changed source receipt for {identifier}")
    result = read(complete)
    artifact = folder / filename
    if result["artifact_sha256"].get(filename) != sha(artifact):
        raise ValueError(f"Changed source artifact for {identifier}/{filename}")
    return artifact, result


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def install_real_audit(model, job, state, audit, out):
    original_setup = model.backbone_peft_setup
    active = {}
    destination = out / "next_update.pt"
    source = job.get("matched_state_from")
    if source is not None:
        source_path, source_result = source_case(source, "next_update.pt")
        expected = ("dataset", "scene", "frames", "width", "seed")
        if any(job[field] != source_result[field] for field in expected):
            raise ValueError("Matched-state probe must use the source inputs")
        snapshot = torch.load(source_path, map_location="cpu", weights_only=True)
        audit["matched_state_sha256"] = sha(source_path)

    def setup(*args, **kwargs):
        optimizer, params = original_setup(*args, **kwargs)
        parameter_ids = {id(param) for param in params}
        selected = {name: param for name, param in model.named_parameters() if id(param) in parameter_ids}
        if len(selected) != 384:
            raise ValueError("Unexpected optimizer inventory for real-loss probe")
        active.update(optimizer=optimizer, selected=selected)

        def capture_update(*_):
            step = state["updates"]
            for name, param in selected.items():
                statistics = optimizer.state[param]
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    key_output = "adam_step" if key == "step" else key
                    state["factor_values"][f"{key_output}/{step}/{name}"] = export(statistics[key])
            if step == 1:
                torch.save(dict(
                    factors=cpu_copy(selected), optimizer=cpu_copy(optimizer.state_dict()),
                    cuda_rng=torch.cuda.get_rng_state(), cpu_rng=torch.get_rng_state(),
                ), destination)

        optimizer.register_step_post_hook(capture_update)
        return optimizer, params

    model.backbone_peft_setup = setup
    if source is not None:
        aggregate = model.aggregate_with_patch_tokens

        def restore(*args, **kwargs):
            if kwargs.get("use_checkpoint") and state["updates"] == 1:
                with torch.no_grad():
                    for name, param in active["selected"].items():
                        param.copy_(snapshot["factors"][name])
                active["optimizer"].load_state_dict(snapshot["optimizer"])
                torch.set_rng_state(snapshot["cpu_rng"])
                torch.cuda.set_rng_state(snapshot["cuda_rng"])
                audit["matched_restored"] = {name: tensor_hash(p) for name, p in active["selected"].items()}
            return aggregate(*args, **kwargs)

        model.aggregate_with_patch_tokens = restore
    return active


def setup_execution(model, job, state, audit, scope, monitor):
    policy = job["policy"]
    recompute = policy in DEPTH_RECOMPUTE
    views = len(job["frames"])
    spans = tuple((start, min(start + 8, views)) for start in range(0, views, 8))
    seen_recompute_spans = set()
    if policy != "native":
        install_camera(model, True, scope)
    if recompute:
        native_installer = full40.install_head

        @contextmanager
        def depth_scope(name):
            if name.startswith("depth.recompute"):
                seen_recompute_spans.add(name.removeprefix("depth.recompute"))
            with scope(name) if scope is not None else nullcontext():
                yield

        def install(model, chunk, enabled, diagnostics=None):
            return native_installer(model, chunk, enabled, depth_scope)

        with patch.object(full40, "install_head", install):
            path = full40.setup_model(model, dict(job, policy="c8_on"), state, audit, scope)
    else:
        path = full40.setup_model(model, dict(job, policy="c8_off"), state, audit, scope)
    layouts = depth_layouts(model, spans)
    if monitor is not None:
        original_setup = model.backbone_peft_setup

        def setup(*args, **kwargs):
            optimizer, params = original_setup(*args, **kwargs)
            monitor.optimizer = optimizer
            return optimizer, params

        model.backbone_peft_setup = setup
        if not recompute:
            forward = model.depth_head.forward

            def depth_saved_storage(*args, **kwargs):
                with torch.autograd.graph.saved_tensors_hooks(monitor.pack_depth, lambda value: value):
                    return forward(*args, **kwargs)

            model.depth_head.forward = depth_saved_storage
    return path, layouts, seen_recompute_spans


def run(job, out, supplied, reference):
    from gsplat.cuda._backend import _C

    seed, steps, probe = job["seed"], job["steps"], job.get("probe")
    fixed = probe in {"fixed", "leak"}
    audit = {} if fixed or probe in {"identity", "real"} else None
    device = full40.warm_device()
    seed_model(seed)
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = steps, 0.0003, seed
    state = {"updates": 0, "lora_tensors": 0}
    if steps and audit is None and not job.get("profile"):
        state["adaptation_clock"] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    if audit is not None:
        state["factor_values"] = {}
    loss_history = []
    cold_start = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    loaded = time.perf_counter()
    predictions = {}
    model.register_forward_hook(lambda _, inputs, output: predictions.update(
        {name: output[name] for name in FIELDS}))
    policy = job["policy"]

    with ExitStack() as stack:
        monitor = None
        scope = None
        if job.get("profile") == "timeline":
            monitor = LifetimeMonitor()
            stream = stack.enter_context((out / "profile.jsonl").open("x", encoding="utf-8"))
            scope = LifetimeTimeline(stream, state, steps, monitor).phase
        mark = scope or (lambda _: nullcontext())
        effective = stack.enter_context(patched_runtime(policy, scope, monitor))
        path, batches, recompute_spans = setup_execution(model, job, state, audit, scope, monitor)
        if probe == "real":
            install_real_audit(model, job, state, audit, out)
        _, observed_loss = observe_losses(state, audit, fixed)
        native_clip = torch.nn.utils.clip_grad_norm_

        def loss(*args, **kwargs):
            with mark("tco_loss_and_render"):
                if probe == "capture":
                    from check_tco_render_lifetime import capture_renderer_inputs

                    filename = "renderer_inputs.pt" if not state["updates"] else f"renderer_inputs_{state['updates']}.pt"
                    capture_renderer_inputs(kwargs, out / filename)
                if monitor is not None:
                    monitor.remember("adaptation_predictions", kwargs["predictions"])
                result, terms = observed_loss(*args, **kwargs)
                if audit is not None:
                    update = str(state["updates"] + 1)
                    audit.setdefault("pose_encoding_hashes", []).append(tensor_hash(kwargs["predictions"]["pose_enc"]))
                    audit.setdefault("loss_components", {})[update] = {
                        **{key: float(value.detach()) for key, value in terms.items()}, "total": float(result.detach())
                    }
                loss_history.append(state["last_loss"])
                return result, terms

        def clip(*args, **kwargs):
            with mark("clipping"):
                if audit is not None:
                    update = state["updates"] + 1
                    for name, param in model.named_parameters():
                        if name.endswith((".lora_A", ".lora_B")):
                            if param.grad is None:
                                raise ValueError(f"Missing gradient for {name}")
                            state["factor_values"][f"preclip/{update}/{name}"] = export(param.grad)
                norm = native_clip(*args, **kwargs)
                if audit is not None:
                    audit.setdefault("preclip_total_norm", {})[str(update)] = float(norm)
                return norm

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
                                             full40.instrument(torch.Tensor.backward, scope, "backward")))
        if audit is not None or scope is not None:
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
        step_seconds = clock[0].elapsed_time(clock[1]) / (1_000 * steps) if clock else None

    losses = torch.stack(loss_history).tolist() if loss_history else []
    if state["updates"] != steps or len(losses) != steps:
        raise ValueError("Invalid optimizer update or loss count")
    if not all(map(math.isfinite, losses)):
        raise ValueError("Nonfinite optimizer loss")
    if versions != [param._version for _, param in frozen]:
        raise ValueError("Unexpected base-weight mutation")
    if audit is not None:
        audit["frozen_end_sha256"] = named_hash(frozen)
        if audit["frozen_end_sha256"] != audit["frozen_start_sha256"]:
            raise ValueError("Frozen weight bytes changed")
        audit["adapter_final"] = {name: tensor_hash(param) for name, param in model.named_parameters()
                                  if name.endswith((".lora_A", ".lora_B"))}
    if policy in DEPTH_RECOMPUTE and steps:
        spans = {f"[{start}:{min(start + 8, len(job['frames']))}]" for start in range(0, len(job["frames"]), 8)}
        if recompute_spans != spans:
            raise ValueError(f"Depth checkpoint replay spans differ: {recompute_spans} vs {spans}")
    arrays = dict(canonical=points, pipeline=pipeline, **{name: export(predictions[name]) for name in FIELDS})
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite reconstruction")
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
        if any(not np.isfinite(value).all() for value in state["factor_values"].values()):
            raise ValueError("Nonfinite gradient, factor, or optimizer state")
        np.savez_compressed(out / "lora_audit.npz", **state["factor_values"])
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=device, allocator=ALLOCATOR,
                   gpu_memory_fraction=GPU_MEMORY_FRACTION, gsplat_sha256=sha(_C.__file__), **effective)
    artifacts = {path.name: sha(path) for path in out.iterdir()
                 if path.name in {"prediction.npz", "lora_audit.npz", "next_update.pt", "profile.jsonl"}
                 or path.name.startswith("renderer_inputs")}
    write_new(out / "complete.json", dict(
        **job, input_sha256=inputs, reference_sha256=ref, output_sha256=signature(arrays),
        raster_shape=list(supplied["images"].shape), native_depth_batches=sorted(batches),
        camera_checkpoint=policy != "native", depth_recompute_spans=sorted(recompute_spans),
        scene_seconds=end - begin, placement_seconds=placement, mean_step_seconds=step_seconds,
        checkpoint_seconds=loaded - cold_start, cold_seconds=end - cold_start, score_seconds=score_seconds,
        cuda_free_before=free, cuda_free_after=end_free, cuda_device_total=total,
        peak_allocated=peak, peak_reserved=reserved,
        host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        updates=state["updates"], lora_tensors=state["lora_tensors"], head_taps=list(TAPS),
        losses=losses, metrics=metrics, runtime=runtime, audit=audit, artifact_sha256=artifacts,
        rental_compute_usd=0))
    print(f"{job['id']} {end - begin:.2f}s, {peak / 2**30:.3f} GiB peak allocated", flush=True)


def rescore(job, out, reference):
    source, previous = source_case(job["source_job"], "prediction.npz")
    if any(job[key] != previous[key] for key in ("dataset", "scene", "frames", "width", "seed")):
        raise ValueError("Rescore case must use the source inputs")
    seed_model(job["seed"])
    reference_sha = signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"]))
    if reference_sha != previous["reference_sha256"]:
        raise ValueError("Rescore did not load the original evaluation geometry")
    with np.load(source, allow_pickle=False) as data:
        if signature({key: data[key] for key in data.files}) != previous["output_sha256"]:
            raise ValueError("Archive reconstruction signature changed")
        metrics = base.evaluate(data["canonical"], reference, out)
    write_new(out / "complete.json", dict(
        **job, metrics=metrics, original_metrics=previous["metrics"], reference_sha256=reference_sha,
        output_sha256=previous["output_sha256"], source_prediction_sha256=sha(source), rental_compute_usd=0))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("job")
    cli.add_argument("attempt")
    cli.add_argument("--registration", choices=tuple(REGISTRATIONS), required=True)
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.job) or not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        cli.error("Invalid job or attempt identity")
    plan = read(REGISTRATIONS[args.registration])
    job, = (item for item in plan["jobs"] if item["id"] == args.job)
    out = PRIVATE / "cases" / args.job / args.attempt / "case"
    out.mkdir(parents=True, exist_ok=False)
    probe, profile = job.get("probe"), job.get("profile")
    if probe == "cpu" or profile == "fixture":
        from check_tco_render_lifetime import cpu_fixture

        torch.set_num_threads(1)
        write_new(out / "complete.json", dict(**job, fixture=cpu_fixture(), rental_compute_usd=0))
        return
    if probe in {"renderer", "synthetic"}:
        from check_tco_render_lifetime import renderer_fixture

        full40.warm_device()
        capture = None
        if job.get("capture_job"):
            capture, _ = source_case(job["capture_job"], "renderer_inputs.pt")
            source_case(job["capture_job"], "renderer_inputs_1.pt")
        write_new(out / "complete.json", dict(**job, renderer=renderer_fixture(capture), rental_compute_usd=0))
        return
    check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
    supplied, reference = load_case(job, out, job["width"])
    if probe == "rescore" or profile == "rescore":
        rescore(job, out, reference)
        return
    if probe in {"fixed", "leak"}:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        run(job, out, supplied, reference)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
