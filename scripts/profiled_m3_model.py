"""One fresh M3 model state; no CLI, launch permission or score release.

The verified wrapper owns nominee/stage authorization, the shared GPU lock and
provider/local process control. This core verifies actual registered bytes,
constructs one fresh frozen model, trains only LoRA using unchanged M2 helpers,
and records native/final model outputs in M3-only unused attempt directories.
CPU mocks exercise orchestration only; real centered/clean endpoint acceptance
and the future byte-verifying wrapper remain required before scientific use.
"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
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
from types import SimpleNamespace
import uuid

from profiled_artifacts import logical_inventory, resolve_logical
from profiled_m3_gate import state_id
from profiled_runtime import EXPECTED_RUNTIME
from run_profiled_lora import counted_decoder_call, dump_new, sha

ROOT = Path(__file__).resolve().parents[1]
CORE = "scripts/profiled_m3_model.py"
INPUT_PLAN = "manifests/lora_profiled_m3_inputs_v2.json"
INPUT_ATTEMPT = "inputs02"
BRIDGE = "results/mast3r_neural_transfer/zero_adapter_bridge.json"
ENVIRONMENT = "results/lora_profiled_v1/environment.json"
RGB_MANIFEST = "manifests/dense_focal_inputs.json"
_EXECUTION_ATTEMPTED = False


@dataclass(frozen=True)
class ModelContext:
    job: dict
    attempt: str
    registry_file: str
    registry_sha256: str
    source_sha256_lf: dict
    artifact_sha256: dict
    input_sha256: dict
    case_input_digest: str
    docker_image: str
    checkpoint_sha256: str
    checkpoint_config_sha256: str
    original_state_sha256: str
    weights: Path
    execution_commit: str
    runtime_identity: dict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA256: " + label)


def _file(root, name):
    return resolve_logical(root, name, must_exist=True)


def _maps(root, mapping, text=False):
    require(isinstance(mapping, dict) and mapping, "Missing registered file identities")
    for name, expected in sorted(mapping.items()):
        _digest(expected, name)
        require(sha(_file(root, name), text=text) == expected, "Registered bytes changed: " + name)


def _job(job):
    require(job.get("scene") in ("heads", "chess", "fire") and job.get("side") in ("earlier", "later")
            and job.get("grid") in ("centered", "shifted") and job.get("pose") in ("clean", "noisy"),
            "Unsupported scene/source/grid/pose")
    require(type(job.get("repeat")) is int and job["repeat"] in (0, 1, 2) and job.get("arm") in
            ("P0", "P1", "P2", "P3", "P4", "P5"), "Unsupported arm or repeat")
    require(job.get("job_id") == state_id(job["scene"], job["side"], job["grid"], job["pose"],
                                         job["repeat"], job["arm"]), "Noncanonical job ID")
    require(job.get("case_id") == "_".join(job[key] for key in ("scene", "side", "grid", "pose")),
            "Noncanonical case ID")
    trained = job["arm"] != "P0"
    expected = dict(fresh_pretrained_process=True, original_weights_frozen=True, seed=42,
                    lora_enabled=trained, rank=4 if trained else None, alpha=1 if trained else None,
                    zero_function_initialization=trained, macrosteps=8 if trained else 0,
                    ordered_pairs_per_macrostep=20 if trained else 0,
                    collect_replay_passes_per_macrostep=2 if trained else 0,
                    optimizer="Adam" if trained else "none", learning_rate=.0001 if trained else None,
                    clip_gradient_norm=1. if trained else None)
    require(job.get("training") == expected, "Training contract differs from the validated M2 schedule")


def _verify_bindings(context, root):
    """Read/hash identities only; no model, fitting arrays or evaluation labels."""
    require(isinstance(context, ModelContext), "Expected ModelContext")
    _job(context.job)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", context.attempt) is not None,
            "Invalid fresh attempt ID")
    _digest(context.registry_sha256, "registry")
    registry_path = _file(root, context.registry_file)
    require(sha(registry_path) == context.registry_sha256, "Execution registry bytes changed")
    registry = _read(registry_path)
    require(registry.get("schema") == "profiled_m3_execution_v1", "Unexpected M3 execution schema")
    jobs = [job for job in registry.get("jobs", []) if job.get("job_id") == context.job["job_id"]]
    require(len(jobs) == 1 and jobs[0] == context.job, "Job differs from exact registered descriptor")
    core = registry["model_core"]
    require(core.get("input_plan") == INPUT_PLAN and core.get("input_attempt") == INPUT_ATTEMPT,
            "Only the preserved v2/inputs02 preparation is supported")
    for key in ("source_sha256_lf", "artifact_sha256", "docker_image", "checkpoint_sha256", "checkpoint_config_sha256",
                "original_state_sha256", "runtime_identity"):
        require(core.get(key) == getattr(context, key), "Context differs from registry model_core: " + key)
    require(core["cases"][context.job["case_id"]] == dict(input_sha256=context.input_sha256,
            case_input_digest=context.case_input_digest), "Case identities differ from registered model inputs")
    from register_profiled_execution import SOURCES
    required = set(SOURCES) | {CORE, "scripts/run_profiled_lora.py", "scripts/profiled_m3_network.py",
                               "scripts/prepare_profiled_m3_inputs.py", "scripts/profiled_artifacts.py",
                               "scripts/profiled_m3_gate.py"}
    require(required.issubset(context.source_sha256_lf), "Required model implementation source is unbound")
    require(context.source_sha256_lf[CORE] == sha(Path(__file__), text=True), "This model core is not registered")
    require({INPUT_PLAN, BRIDGE, ENVIRONMENT, RGB_MANIFEST}.issubset(context.artifact_sha256),
            "Required model provenance is unbound")
    _maps(root, context.source_sha256_lf, text=True)
    _maps(root, context.artifact_sha256)
    _maps(root, context.input_sha256)
    _digest(context.case_input_digest, "prepared case")
    _digest(context.checkpoint_sha256, "checkpoint")
    _digest(context.checkpoint_config_sha256, "checkpoint config")
    _digest(context.original_state_sha256, "original model state")
    require(context.job.get("checkpoint_sha256") == context.checkpoint_sha256,
            "Job checkpoint identity differs from the actual registered checkpoint")
    require(_read(_file(root, BRIDGE))["original_state_hash"] == context.original_state_sha256,
            "Original-model identity differs from the preserved zero-adapter bridge")
    environment = _read(_file(root, ENVIRONMENT))
    require(context.docker_image == environment["docker_image"] and
            context.checkpoint_sha256 == environment["checkpoint_sha256"], "Image/checkpoint differs from M2 environment")
    expected_runtime = dict(torch=environment["torch"], cuda=environment["cuda"], gpu=environment["gpu"],
                            policy=EXPECTED_RUNTIME)
    require(context.runtime_identity == expected_runtime, "Runtime differs from pinned model arithmetic/environment")
    require(sha(Path(context.weights) / "model.safetensors") == context.checkpoint_sha256,
            "Actual model checkpoint bytes differ")
    require(sha(Path(context.weights) / "config.json") == context.checkpoint_config_sha256,
            "Actual model configuration bytes differ")
    rgb = _read(_file(root, RGB_MANIFEST))["inputs"][context.job["scene"] + "_base"]
    require(isinstance(rgb.get("file"), str) and Path(rgb["file"]).name == rgb["file"]
            and rgb["file"].endswith(".npz"), "RGB input must use its registered archive basename")
    expected_scene = {"heads": "heads/seq-01", "chess": "chess/seq-03", "fire": "fire/seq-03"}[context.job["scene"]]
    require(rgb.get("scene") == expected_scene and rgb.get("batch") == "base", "Wrong registered RGB scene/window")
    _digest(rgb["sha256"], "RGB archive")
    _digest(rgb["rgb_sha256"], "RGB tensor identity")
    rgb_name = "reports/dense_focal_pilot/inference/" + rgb["file"]
    require(context.artifact_sha256.get(rgb_name) == rgb["sha256"],
            "Selected RGB archive is not bound for this model state")
    require(sha(_file(root, rgb_name)) == rgb["sha256"], "Actual selected RGB archive bytes differ")
    require(re.fullmatch(r"[0-9a-f]{40}", context.execution_commit) is not None, "Invalid actual execution commit")
    return registry


def _paths(context, root):
    suffix = context.job["job_id"] + "/" + context.attempt
    raw = resolve_logical(root, "reports/lora_profiled_m3_v1/cases/" + suffix)
    public = resolve_logical(root, "results/lora_profiled_m3_v1/cases/" + suffix)
    if raw.exists() or public.exists():
        raise FileExistsError("Preserve completed, failed or partial M3 attempts; choose a fresh attempt")
    return raw, public


def _production_backend():
    """Heavy imports are deferred until registered bytes have passed preflight."""
    import numpy as np
    import torch
    from mast3r.model import AsymmetricMASt3R
    from mast3r_network_priors import digest, install
    from profiled_runtime import configure_runtime, verify_runtime
    from profiled_network import (PAIRS, collect_points, fuse, output_adjoints, prepare_training,
                                  replay_backward, training_features, views_for)
    from profiled_m3_network import load_case, input_digest, capture_endpoint

    def to_device(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [to_device(item, device) for item in value]
        return value

    def model_factory(weights):
        return AsymmetricMASt3R.from_pretrained(str(weights), local_files_only=True).eval().cuda().requires_grad_(False)

    def git_commit(root):
        return subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(root),
                                        "rev-parse", "HEAD"], text=True).strip()

    return SimpleNamespace(np=np, torch=torch, configure_runtime=configure_runtime, verify_runtime=verify_runtime,
        digest=digest, install=install, PAIRS=PAIRS, collect_points=collect_points, fuse=fuse,
        output_adjoints=output_adjoints, prepare_training=prepare_training, replay_backward=replay_backward,
        training_features=training_features, views_for=views_for, load_case=load_case, input_digest=input_digest,
        capture_endpoint=capture_endpoint, to_device=to_device, model_factory=model_factory, git_commit=git_commit)


def _original_objects(model):
    return list(model.parameters()) + list(model.buffers())


def _check_original_objects(model, originals, named_adapters):
    adapters = {id(parameter) for _, parameter in named_adapters}
    current = [value for value in _original_objects(model) if id(value) not in adapters]
    require({id(value) for value in current} == {id(value) for value in originals},
            "An original parameter/buffer object was replaced or removed")
    require(not any(value.requires_grad for value in originals), "An original parameter/buffer became trainable")


def _run(context, backend, root):
    """Private injected backend exists only for explicit CPU orchestration fixtures."""
    context = deepcopy(context)
    registry = _verify_bindings(context, root)
    raw, public = _paths(context, root)
    ROOT = root
    job = context.job
    np, torch = backend.np, backend.torch
    (digest, install, PAIRS, collect_points, fuse, output_adjoints, prepare_training,
     replay_backward, training_features, views_for, to_device) = (
        backend.digest, backend.install, backend.PAIRS, backend.collect_points, backend.fuse,
        backend.output_adjoints, backend.prepare_training, backend.replay_backward,
        backend.training_features, backend.views_for, backend.to_device)
    arithmetic = backend.configure_runtime()
    require(arithmetic == context.runtime_identity["policy"], "Actual configured arithmetic differs")
    require(torch.__version__ == context.runtime_identity["torch"] and torch.version.cuda == context.runtime_identity["cuda"]
            and torch.cuda.is_available() and torch.cuda.get_device_name() == context.runtime_identity["gpu"],
            "Actual model runtime or local GPU differs")
    require(backend.git_commit(root) == context.execution_commit, "Actual execution commit differs")
    available = shutil.disk_usage(root).free
    require(available >= 1024 ** 3, "Less than 1 GiB free for this model attempt")
    checkpoint = context.checkpoint_sha256
    process = dict(pid=os.getpid(), hostname=socket.gethostname(), process_run_uuid=uuid.uuid4().hex,
                   model_initialization_uuid=uuid.uuid4().hex, model_constructions=0)
    runtime = dict(schema="profiled_m3_model_attempt_v1", job=job, job_id=job["job_id"], attempt=context.attempt,
        execution_commit=context.execution_commit, registry_file=context.registry_file,
        registry_sha256=context.registry_sha256, source_sha256_lf=context.source_sha256_lf,
        artifact_sha256=context.artifact_sha256, input_sha256=context.input_sha256,
        case_input_digest=context.case_input_digest, input_plan=INPUT_PLAN, input_attempt=INPUT_ATTEMPT,
        docker_image=context.docker_image, checkpoint_sha256=checkpoint,
        checkpoint_config_sha256=context.checkpoint_config_sha256, runtime_identity=context.runtime_identity,
        runtime_policy=arithmetic, seed=42, raw_attempt=raw.relative_to(root).as_posix(),
        public_attempt=public.relative_to(root).as_posix(), process=process,
        disk_free_bytes_before=available, rental_compute_usd=0,
        held_source_development_arrays_loaded=False, target_depth_arrays_loaded=False,
        accuracy_scoring_performed=False, stage_release_authorized=False)

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

    public.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    runtime["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    steps, original_hash, optimizer_updates = [], None, 0
    try:
        dump_new(public / "started.json", runtime)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        case_cpu = backend.load_case(root / INPUT_PLAN, INPUT_ATTEMPT, job["scene"], job["side"],
                                     job["grid"], job["pose"], device="cpu")
        require(all(case_cpu[key] == job[job_key] for key, job_key in (
            ("scene", "scene"), ("side", "side"), ("grid", "grid"), ("pose_condition", "pose"))),
            "Loaded a different selected M3 case")
        require(case_cpu["input_sha256"] == context.input_sha256 and
                case_cpu["prepared_input_digest"] == backend.input_digest(case_cpu) == context.case_input_digest,
                "Actual fitting files/tensors differ from the registered selected case")
        process["model_constructed_at_utc"] = datetime.now(timezone.utc).isoformat()
        model = backend.model_factory(context.weights)
        process["model_constructions"] += 1
        original_state = list(model.state_dict().items())
        original_objects = _original_objects(model)
        _check_original_objects(model, original_objects, [])
        original_hash = digest(original_state)
        require(original_hash == context.original_state_sha256, "Fresh model differs from validated original state")
        named_adapters, modules = [], []
        teacher_calls = 0
        adapter_initial_hash = None
        preparation_peak_allocated = torch.cuda.max_memory_allocated()/2**20
        preparation_peak_reserved = torch.cuda.max_memory_reserved()/2**20
        training_started = time.perf_counter()
        if job["training"]["lora_enabled"]:
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
                print(json.dumps(dict(run_id=job["job_id"], step=step, loss=row["loss_before_update"],
                                      seconds=row["seconds"], peak_allocated_mib=row["peak_allocated_mib"])), flush=True)
                del points, adjoints, loss, scales, before
            save_arrays(raw / "adapters.npz", {name: parameter.detach().cpu().numpy() for name, parameter in named_adapters})
            del optimizer, prepared, teacher_confidence, features, views, case
        training_seconds = time.perf_counter() - training_started
        require(optimizer_updates == job["training"]["macrosteps"], "Optimizer update count differs")
        before_endpoint_hash = digest(original_state)
        require(before_endpoint_hash == original_hash, "Original weights/buffers changed during adaptation")
        _check_original_objects(model, original_objects, named_adapters)
        adapter_final_hash = digest(named_adapters) if named_adapters else None
        pre_endpoint_peak = torch.cuda.max_memory_allocated()/2**20
        torch.cuda.empty_cache()
        endpoint_start = time.perf_counter()
        endpoint, endpoint_calls = counted_decoder_call(model, backend.capture_endpoint, model, case_cpu,
            raw / "endpoint_cache", raw / "endpoint", public / "endpoint")
        require(endpoint_calls == 20, "Fresh native endpoint must invoke all 20 directed pairs exactly once")
        evidence = endpoint[4]
        endpoint_seconds = time.perf_counter() - endpoint_start
        endpoint_peak_reserved = torch.cuda.max_memory_reserved()/2**20
        after_hash = digest(original_state)
        require(after_hash == original_hash, "Original weights/buffers changed during endpoint capture")
        _check_original_objects(model, original_objects, named_adapters)
        if named_adapters:
            require(digest(named_adapters) == adapter_final_hash, "Endpoint capture changed learned LoRA factors")
        require(backend.input_digest(case_cpu) == context.case_input_digest,
                "Endpoint capture changed prepared fitting tensors")
        _verify_bindings(context, root)
        require(backend.verify_runtime() == arithmetic, "Runtime changed during model execution")
        require(process["model_constructions"] == 1, "Expected exactly one fresh model construction")
        summary = dict(**runtime, status="model_outputs_complete_without_scoring", completed_at_utc=datetime.now(timezone.utc).isoformat(),
            original_state_sha256=original_hash, pre_endpoint_original_state_sha256=before_endpoint_hash,
            final_original_state_sha256=after_hash, original_state_unchanged=True,
            adapter_module_count=len(modules), trainable_parameters=sum(p.numel() for _, p in named_adapters),
            adapter_initial_sha256=adapter_initial_hash, adapter_final_sha256=adapter_final_hash,
            optimizer_updates=optimizer_updates, teacher_decoder_calls=teacher_calls,
            training_graph_free_decoder_calls=sum(row["graph_free_decoder_calls"] for row in steps),
            training_differentiable_decoder_calls=sum(row["differentiable_decoder_calls"] for row in steps),
            endpoint_decoder_calls=endpoint_calls, steps=steps, endpoint=evidence,
            training_and_teacher_seconds=training_seconds, endpoint_seconds=endpoint_seconds,
            seconds=time.perf_counter()-started, preparation_peak_allocated_mib=preparation_peak_allocated,
            preparation_peak_reserved_mib=preparation_peak_reserved, endpoint_peak_reserved_mib=endpoint_peak_reserved,
            peak_allocated_mib=max([preparation_peak_allocated, pre_endpoint_peak, evidence["capture"]["peak_allocated_mib"]]
                + [row["peak_allocated_mib"] for row in steps]),
            peak_reserved_mib=max([preparation_peak_reserved, endpoint_peak_reserved]
                + [row["peak_reserved_mib"] for row in steps]),
            raw_files={name: sha(path) for name, path in logical_inventory(raw).items()},
            public_files={name: sha(path) for name, path in logical_inventory(public).items()},
            pass_all=True, accuracy_improvement_established=False)
        dump_new(public / "complete.json", summary)
        return summary
    except Exception as error:
        dump_new(public / "failure.json", dict(**runtime, error_type=type(error).__name__,
            failed_at_utc=datetime.now(timezone.utc).isoformat(), message=str(error), traceback=traceback.format_exc(),
            original_state_sha256=original_hash, optimizer_updates_completed=optimizer_updates,
            sealed_step_count=len(steps), seconds=time.perf_counter()-started, pass_all=False,
            recovery="Preserve this attempt and all partial files; classify before a fresh process/attempt"))
        raise


def execute(context):
    """Execute one actual model state only when called by the verified wrapper.

    The registry bindings are checked here; nominee/stage/lock authorization is
    separately the wrapper's responsibility. There is intentionally no CLI,
    alternate backend, arbitrary output path, or success/approval boolean.
    """
    global _EXECUTION_ATTEMPTED
    require(not _EXECUTION_ATTEMPTED, "One model attempt per process; start a fresh registered process")
    context = deepcopy(context)
    _verify_bindings(context, ROOT)
    _paths(context, ROOT)
    _EXECUTION_ATTEMPTED = True
    return _run(context, _production_backend(), ROOT)
