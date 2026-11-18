"""Run one registered profiled-LoRA state, seal predictions, and never score it.

Dry-run verifies the scientific registry and selected job without importing
PyTorch, constructing a model, reserving attempt directories, or using a GPU.
Real runs require --image (or PROFILED_DOCKER_IMAGE) supplied by the launcher.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import subprocess
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/run_profiled_lora.py"


def sha(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inside_root(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output or input path escapes the repository: {name}")
    return path


def preflight(args):
    """CPU-only authoritative gate; no attempt is reserved here."""
    from register_profiled_execution import verify_registry

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.attempt):
        raise ValueError("Attempt must be 1-64 letters/digits/hyphens/underscores, beginning with a letter or digit")
    manifest = inside_root(args.manifest)
    verify_registry(manifest, "scientific")
    registry = read(manifest)
    if registry.get("scientific_ready_at_registration") is not True:
        raise ValueError("Scientific registry was not ready when sealed")
    if registry["source_sha256_lf"].get(RUNNER) != sha(Path(__file__), text=True):
        raise ValueError("This runner is not the source registered for scientific execution")
    selected = [job for job in registry["jobs"] if job["run_id"] == args.run_id]
    if len(selected) != 1:
        raise ValueError("Run ID must identify exactly one registered scientific job")
    job = selected[0]
    plan = registry["study_settings"]
    optimization = plan["optimization"]
    fixed = {"macro_steps": 8, "all_directed_pairs_per_step": 20,
             "adam_learning_rate": .0001, "gradient_clip_norm": 1,
             "seed": 42, "head_compute_dtype": "float32",
             "decoder_compute_dtype": "bfloat16", "small_solve_dtype": "float64", "tf32": True}
    if any(optimization.get(key) != value for key, value in fixed.items()):
        raise ValueError("Registered optimization differs from this runner's validated implementation")
    if any(plan["adapter"].get(key) != value for key, value in {"rank": 4, "alpha": 1, "dropout": 0}.items()):
        raise ValueError("Registered adapter configuration differs")
    if job["adapt_lora"] != (job["arm"] != "P0") or job["optimizer_steps"] != (8 if job["adapt_lora"] else 0):
        raise ValueError("Job adaptation/step contract differs")
    raw = inside_root(job["raw_attempt_parent"]) / args.attempt
    public = inside_root(job["public_attempt_parent"]) / args.attempt
    if raw.exists() or public.exists():
        raise FileExistsError("Preserve existing completed, failed, or partial attempts; choose a fresh attempt ID")
    return manifest, registry, job, raw, public


@contextmanager
def scientific_gpu_lock(identity):
    """Serialize this runner's jobs; never remove another job's existing lock."""
    path = ROOT / "reports/lora_profiled_v1/scientific_gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(identity, pid=os.getpid(), hostname=socket.gethostname())
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    try:
        yield
    finally:
        if path.exists() and read(path) == payload:
            path.unlink()


def counted_decoder_call(model, function, *args, **kwargs):
    """Count real outer decoder invocations, including calls inside helpers."""
    original = model._decoder
    count = 0

    def observed(*values, **options):
        nonlocal count
        count += 1
        return original(*values, **options)

    model._decoder = observed
    try:
        result = function(*args, **kwargs)
    finally:
        model._decoder = original
    return result, count


def run(args, checked):
    """Execute only after scientific preflight; all heavy imports stay here."""
    manifest, registry, job, raw, public = checked
    image = args.image or os.environ.get("PROFILED_DOCKER_IMAGE")
    if image is None:
        raise ValueError("Real execution needs host-inspected --image or PROFILED_DOCKER_IMAGE")
    import numpy as np
    import torch
    from mast3r.model import AsymmetricMASt3R
    from mast3r_network_priors import digest, install
    from check_profiled_endpoints import capture_endpoint
    from profiled_network import (PAIRS, collect_points, fuse, input_hashes, load_case,
                                  output_adjoints, prepare_training, replay_backward,
                                  training_features, views_for)

    environment = read(ROOT / "results/lora_profiled_v1/environment.json")
    if image != registry["docker_image"] or image != environment["docker_image"]:
        raise ValueError("Host-inspected Docker image differs from the scientific registry")
    if torch.__version__ != environment["torch"] or torch.version.cuda != environment["cuda"]:
        raise ValueError("Current PyTorch/CUDA version differs from the registered runtime")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != environment["gpu"]:
        raise ValueError("Registered local GPU is unavailable; this runner does not authorize rental compute")
    checkpoint = sha(args.weights / "model.safetensors")
    if checkpoint != registry["checkpoint_sha256"] or checkpoint != environment["checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint differs from the registered scientific checkpoint")
    available = shutil.disk_usage(ROOT).free
    if available < 1024**3:
        raise OSError("Less than 1 GiB of local free space; preserve space for this attempt's endpoint cache")
    execution_commit = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    identity = dict(run_id=job["run_id"], attempt=args.attempt)
    runtime = dict(**identity, schema="profiled_scientific_attempt_v1", job=job,
        execution_commit=execution_commit, registry_execution_commit=registry["execution_commit"],
        manifest=manifest.relative_to(ROOT).as_posix(), manifest_sha256=sha(manifest),
        source_sha256_lf=registry["source_sha256_lf"], artifact_sha256=registry["artifact_sha256"],
        input_sha256=input_hashes(), docker_image=image,
        image_identity_supplied_by="--image" if args.image else "PROFILED_DOCKER_IMAGE",
        checkpoint_sha256=checkpoint, torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), seed=42, allow_tf32=True,
        raw_attempt=raw.relative_to(ROOT).as_posix(), public_attempt=public.relative_to(ROOT).as_posix(),
        disk_free_bytes_before=available, rental_compute_usd=0,
        held_source_development_arrays_loaded=False, target_depth_arrays_loaded=False,
        accuracy_scoring_performed=False)

    def save_arrays(path, values):
        with path.open("xb") as stream:
            np.savez_compressed(stream, **values)

    def gradient_summary(named):
        values = {name: parameter.grad.detach().cpu() for name, parameter in named if parameter.grad is not None}
        if not values or not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise FloatingPointError("Missing or nonfinite LoRA gradients")
        norm2 = sum(float(value.double().square().sum()) for value in values.values())
        active = [name for name, value in values.items() if bool(value.any())]
        if not active or norm2 <= 0:
            raise FloatingPointError("All LoRA gradients are zero")
        return dict(norm=float(norm2**.5), parameters_with_gradient=len(values),
                    parameters_with_nonzero_gradient=len(active), active_names=active,
                    missing_names=[name for name, _ in named if name not in values])

    def to_device(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [to_device(item, device) for item in value]
        return value

    with scientific_gpu_lock(identity):
        # This lock serializes this runner; the launcher also checks for external GPU jobs.
        if public.exists() or raw.exists():
            raise FileExistsError("Another process reserved this attempt after preflight")
        public.mkdir(parents=True, exist_ok=False)
        raw.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        steps = []
        original_hash = None
        optimizer_updates = 0
        try:
            dump_new(public / "started.json", runtime)
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats()
            case_cpu = load_case(job["scene"], job["adaptation_source"], device="cpu")
            if case_cpu["packet_sha256"] != sha(ROOT / job["input_packet"]):
                raise ValueError("Loaded adaptation packet differs from the selected scientific job")
            model = AsymmetricMASt3R.from_pretrained(
                str(args.weights), local_files_only=True).eval().cuda().requires_grad_(False)
            original_state = list(model.state_dict().items())
            original_hash = digest(original_state)
            bridge_name = "results/mast3r_neural_transfer/zero_adapter_bridge.json"
            graph_audit = read(ROOT / "results/lora_profiled_v1/baseline_graph.json")["scenes"][job["scene"]]
            bridge_sources = [item for item in graph_audit["sources"] if item["file"] == bridge_name]
            if len(bridge_sources) != 1 or sha(ROOT / bridge_name) != bridge_sources[0]["sha256"]:
                raise ValueError("Historical zero-adapter identity changed from the registered graph provenance")
            expected_original = read(ROOT / bridge_name)["original_state_hash"]
            if original_hash != expected_original:
                raise ValueError("Fresh pretrained model state differs from the validated zero-adapter model")
            named_adapters, modules = [], []
            teacher_calls = 0
            adapter_initial_hash = None
            preparation_peak_allocated = torch.cuda.max_memory_allocated()/2**20
            preparation_peak_reserved = torch.cuda.max_memory_reserved()/2**20
            training_started = time.perf_counter()
            if job["adapt_lora"]:
                case = to_device(case_cpu, "cuda")
                _, _, views = views_for(job["scene"] + "_base")
                features = training_features(model, views)
                (initial, teacher_confidence), teacher_calls = counted_decoder_call(
                    model, collect_points, model, features)
                if teacher_calls != 20 or set(initial) != set(PAIRS):
                    raise ValueError("Pristine teacher did not visit every registered directed pair once")
                torch.manual_seed(42)
                modules = install(model, ROOT / "third_party/TCO/peft/lora.py")
                model.eval()
                named_adapters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
                if len(modules) != 144 or sum(p.numel() for _, p in named_adapters) != 1032192:
                    raise ValueError("LoRA placement or parameter count differs from the registered adapter")
                if any(not name.endswith((".lora_A", ".lora_B")) for name, _ in named_adapters):
                    raise ValueError("A non-LoRA model parameter is trainable")
                if any(bool(parameter.detach().any()) for name, parameter in named_adapters if name.endswith(".lora_B")):
                    raise ValueError("LoRA must start with exactly zero B factors")
                adapter_initial_hash = digest(named_adapters)
                prepared = prepare_training(case, initial, teacher_confidence)
                initial_depth, initial_confidence = fuse(initial, teacher_confidence)
                save_arrays(raw / "teacher_initial.npz", dict(
                    depth=initial_depth.detach().cpu().numpy(),
                    confidence=initial_confidence.detach().cpu().numpy()))
                teacher_record = dict(
                    decoder_calls=teacher_calls, checkpoint_sha256=checkpoint,
                    initial_adapter_state_sha256=adapter_initial_hash,
                    fused_native_file=(raw / "teacher_initial.npz").relative_to(ROOT).as_posix(),
                    fused_native_sha256=sha(raw / "teacher_initial.npz"),
                    full_pair_pointmaps_archived=False, graph_audit=prepared["graph_audit"],
                    states=[dict(support_fold=fold,
                        gains=state["gains"].cpu().tolist(),
                        sensor_normalizers=state["sensor_normalizers"].cpu().tolist(),
                        ridge=float(state["ridge"]), support_counts=[len(p["xy"]) for p in state["support"]],
                        query_counts=[len(p["xy"]) for p in prepared["queries"][fold]],
                        graph_filtering=state["graph_filtering"])
                        for fold, state in enumerate(prepared["states"])],
                    legacy_pair_gauges={f"{i}_{j}": float(value) for (i,j), value in prepared["gauges"].items()},
                    confidence_and_graph_frozen=True)
                dump_new(public / "teacher.json", teacher_record)
                del initial, initial_depth, initial_confidence
                preparation_peak_allocated = torch.cuda.max_memory_allocated()/2**20
                preparation_peak_reserved = torch.cuda.max_memory_reserved()/2**20
                parameters = [parameter for _, parameter in named_adapters]
                optimizer = torch.optim.Adam(parameters, lr=.0001)
                optimizer.zero_grad(set_to_none=True)
                for step in range(8):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    step_start = time.perf_counter()
                    before = {name: parameter.detach().cpu().clone() for name, parameter in named_adapters}
                    (points, unused_confidence), collect_calls = counted_decoder_call(
                        model, collect_points, model, features)
                    del unused_confidence
                    loss, scales, adjoints = output_adjoints(points, prepared, job["arm"], step)
                    _, replay_calls = counted_decoder_call(model, replay_backward, model, features, adjoints)
                    if collect_calls != 20 or replay_calls != 20:
                        raise ValueError("Macro-step must collect and replay exactly 20 directed pairs")
                    gradient = gradient_summary(named_adapters)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer_updates += 1
                    optimizer.zero_grad(set_to_none=True)
                    if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
                        raise FloatingPointError("Optimizer produced nonfinite LoRA factors")
                    update_norm2 = sum(float((parameter.detach().cpu().double()-before[name].double()).square().sum())
                                       for name, parameter in named_adapters)
                    if update_norm2 <= 0:
                        raise FloatingPointError("Macro-step did not change any LoRA factor")
                    torch.cuda.synchronize()
                    row = dict(step=step, support_fold=None if job["arm"] == "P1" else step % 2,
                        loss_before_update=float(loss), scales_before_update=None if scales is None else scales.cpu().tolist(),
                        graph_free_decoder_calls=collect_calls, differentiable_decoder_calls=replay_calls,
                        optimizer_steps=1, optimizer_updates_cumulative=optimizer_updates,
                        gradient=gradient, unclipped_gradient_norm=float(norm), gradient_clip_norm=1.,
                        adapter_update_l2=float(update_norm2**.5), seconds=time.perf_counter()-step_start,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        adapter_state_sha256=digest(named_adapters))
                    dump_new(public / f"step_{step:02d}.json", row)
                    steps.append(row)
                    print(json.dumps(dict(run_id=job["run_id"], step=step, loss=row["loss_before_update"],
                                          seconds=row["seconds"], peak_allocated_mib=row["peak_allocated_mib"])), flush=True)
                    del points, adjoints, loss, scales, before
                save_arrays(raw / "adapters.npz", {name: parameter.detach().cpu().numpy() for name, parameter in named_adapters})
                del optimizer, prepared, teacher_confidence, features, views, case
            training_seconds = time.perf_counter() - training_started
            if optimizer_updates != job["optimizer_steps"]:
                raise ValueError("Completed optimizer count differs from the registered job")
            before_endpoint_hash = digest(original_state)
            if before_endpoint_hash != original_hash:
                raise ValueError("Original model weights or buffers changed during adaptation")
            adapter_final_hash = digest(named_adapters) if named_adapters else None
            pre_endpoint_peak = torch.cuda.max_memory_allocated()/2**20
            torch.cuda.empty_cache()
            endpoint_start = time.perf_counter()
            endpoint, endpoint_calls = counted_decoder_call(
                model, capture_endpoint, model, case_cpu, raw / "endpoint_cache", raw / "endpoint", public / "endpoint")
            evidence = endpoint[4]
            endpoint_seconds = time.perf_counter() - endpoint_start
            endpoint_peak_reserved = torch.cuda.max_memory_reserved()/2**20
            after_hash = digest(original_state)
            if after_hash != original_hash:
                raise ValueError("Original model weights or buffers changed during final capture")
            if named_adapters and digest(named_adapters) != adapter_final_hash:
                raise ValueError("Final reconstruction changed the learned LoRA factors")
            for name, expected in runtime["source_sha256_lf"].items():
                if sha(ROOT / name, text=True) != expected:
                    raise ValueError(f"Registered source changed during execution: {name}")
            for name, expected in runtime["artifact_sha256"].items():
                if sha(ROOT / name) != expected:
                    raise ValueError(f"Registered artifact changed during execution: {name}")
            if input_hashes() != runtime["input_sha256"] or sha(manifest) != runtime["manifest_sha256"]:
                raise ValueError("Registered inputs or scientific manifest changed during execution")
            summary = dict(**runtime, status="predictions_sealed_without_accuracy_scoring",
                original_state_sha256=original_hash, pre_endpoint_original_state_sha256=before_endpoint_hash,
                final_original_state_sha256=after_hash, original_state_unchanged=True,
                adapter_module_count=len(modules), trainable_parameters=sum(p.numel() for _, p in named_adapters),
                adapter_initial_sha256=adapter_initial_hash, adapter_final_sha256=adapter_final_hash,
                optimizer_updates=optimizer_updates, teacher_decoder_calls=teacher_calls,
                training_graph_free_decoder_calls=sum(row["graph_free_decoder_calls"] for row in steps),
                training_differentiable_decoder_calls=sum(row["differentiable_decoder_calls"] for row in steps),
                endpoint_decoder_calls=endpoint_calls, steps=steps, endpoint=evidence,
                training_and_teacher_seconds=training_seconds, endpoint_seconds=endpoint_seconds,
                seconds=time.perf_counter()-started,
                preparation_peak_allocated_mib=preparation_peak_allocated,
                preparation_peak_reserved_mib=preparation_peak_reserved,
                endpoint_peak_reserved_mib=endpoint_peak_reserved,
                peak_allocated_mib=max([preparation_peak_allocated, pre_endpoint_peak, evidence["capture"]["peak_allocated_mib"]]
                    + [row["peak_allocated_mib"] for row in steps]),
                peak_reserved_mib=max([preparation_peak_reserved, endpoint_peak_reserved]
                    + [row["peak_reserved_mib"] for row in steps]),
                raw_files={path.relative_to(raw).as_posix(): sha(path) for path in sorted(raw.rglob("*")) if path.is_file()},
                public_files={path.relative_to(public).as_posix(): sha(path) for path in sorted(public.rglob("*")) if path.is_file()},
                pass_all=True, accuracy_improvement_established=False)
            dump_new(public / "complete.json", summary)
            print(json.dumps(dict(run_id=job["run_id"], attempt=args.attempt, pass_all=True,
                                  optimizer_updates=optimizer_updates, accuracy_scoring_performed=False,
                                  complete=(public / "complete.json").relative_to(ROOT).as_posix())), flush=True)
            return summary
        except Exception as error:
            dump_new(public / "failure.json", dict(**runtime, error_type=type(error).__name__,
                message=str(error), traceback=traceback.format_exc(), original_state_sha256=original_hash,
                optimizer_updates_completed=optimizer_updates, sealed_step_count=len(steps),
                seconds=time.perf_counter()-started, pass_all=False,
                recovery="Preserve this attempt and all partial files; investigate before using a new attempt ID"))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/lora_profiled_execution_v1.json"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--weights", type=Path, default=Path("/weights"))
    parser.add_argument("--image", help="Docker image ID inspected by the launcher; alternatively PROFILED_DOCKER_IMAGE")
    parser.add_argument("--dry-run", action="store_true", help="Verify scientific readiness and the selected job, without PyTorch/model/GPU work")
    args = parser.parse_args()
    try:
        checked = preflight(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Scientific preflight refused; no job launched: {error}\n")
    if args.dry_run:
        manifest, registry, job, raw, public = checked
        print(json.dumps(dict(dry_run=True, scientific_ready=True, job=job,
            manifest=manifest.relative_to(ROOT).as_posix(), manifest_sha256=sha(manifest),
            raw_attempt=raw.relative_to(ROOT).as_posix(), public_attempt=public.relative_to(ROOT).as_posix(),
            model_initialized=False, attempt_directories_created=False, gpu_work_performed=False,
            rental_compute_usd=0, accuracy_scoring_performed=False), indent=2))
        return
    run(args, checked)


if __name__ == "__main__":
    main()
