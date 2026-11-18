"""Validate real frozen-backbone LoRA gradients before accuracy experiments."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R

from capture_mast3r_transfer import WEIGHTS
from mast3r_network_priors import digest, install
from profiled_network import (ROOT, PAIRS, collect_points, input_hashes, load_case,
                              output_adjoints, pair_forward, prepare_training,
                              replay_backward, sha, source_hashes, training_features,
                              training_loss, views_for)
from register_profiled_execution import verify_registry


PUBLIC = ROOT / "results/lora_profiled_v1"
SETTINGS = ROOT / "manifests/lora_profiled_bridge_checks_v1.json"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def gradients(model):
    return {name:parameter.grad.detach().cpu().clone() if parameter.grad is not None
            else torch.zeros_like(parameter, device="cpu")
            for name,parameter in model.named_parameters() if parameter.requires_grad}


def gradient_summary(values):
    norm2 = sum(float(value.double().square().sum()) for value in values.values())
    if not all(torch.isfinite(value).all() for value in values.values()) or norm2 <= 0:
        raise ValueError("LoRA gradient is nonfinite or identically zero")
    return dict(norm=float(np.sqrt(norm2)), parameters_with_nonzero_gradient=sum(bool(v.any()) for v in values.values()),
                active_names=[name for name,value in values.items() if bool(value.any())])


def retained_gradient(model, features, constants, prepared, varying, precision):
    model.zero_grad(set_to_none=True)
    points = dict(constants)
    for pair in varying:
        points[pair], _ = pair_forward(model, features, pair, precision)
    loss, _ = training_loss(points, prepared, "P4", 0)
    loss.backward()
    result = gradients(model)
    return float(loss.detach()), result


def compare_gradients(first, second):
    aa = bb = cross = error = 0.
    maximum = 0.
    for name in first:
        a, b = first[name].double(), second[name].double()
        aa += float(a.square().sum())
        bb += float(b.square().sum())
        cross += float((a*b).sum())
        error += float((a-b).square().sum())
        maximum = max(maximum, float((a-b).abs().max()))
    return dict(relative_l2=float(np.sqrt(error/max(aa,1e-30))),
                cosine=float(cross/max(np.sqrt(aa*bb),1e-30)), max_absolute=maximum)


def run(args, attempt):
    from profiled_runtime import configure_runtime, verify_runtime
    arithmetic = configure_runtime()
    spec = json.loads(SETTINGS.read_text())
    verify_registry(args.manifest, "engineering")
    environment = json.loads((PUBLIC / "environment.json").read_text())
    if args.image != environment["docker_image"] or torch.__version__ != environment["torch"] or torch.version.cuda != environment["cuda"]:
        raise ValueError("Current image/PyTorch/CUDA differs from registered runtime")
    sources = source_hashes(("scripts/check_profiled_lora_bridge.py",))
    inputs = input_hashes()
    if sha(args.weights / "model.safetensors") != WEIGHTS:
        raise ValueError("Checkpoint changed")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AsymmetricMASt3R.from_pretrained(str(args.weights), local_files_only=True).eval().cuda().requires_grad_(False)
    original = list(model.state_dict().items())
    original_hash = digest(original)
    case = load_case(spec["scene"], spec["adaptation_source"])
    _, _, views = views_for(spec["scene"]+"_base")
    features = training_features(model, views)
    pristine, teacher_confidence = collect_points(model, features)
    torch.manual_seed(42)
    modules = install(model, ROOT / "third_party/TCO/peft/lora.py")
    trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if len(modules) != 144 or sum(p.numel() for _,p in trainable) != 1032192:
        raise ValueError("Registered LoRA placement or trainable parameter count changed")
    if any(not name.endswith((".lora_A", ".lora_B")) for name,_ in trainable):
        raise ValueError("A non-LoRA parameter is trainable")
    zero, zero_confidence = collect_points(model, features)
    zero_exact = all(torch.equal(pristine[p][branch],zero[p][branch]) for p in PAIRS for branch in (0,1))
    zero_exact = zero_exact and all(torch.equal(teacher_confidence[p],zero_confidence[p]) for p in PAIRS)
    if not zero_exact:
        raise ValueError("Zero-LoRA pair outputs differ from the original frozen model")
    del zero, zero_confidence
    prepared = prepare_training(case, pristine, teacher_confidence)
    varying = [tuple(pair) for pair in spec["varying_pairs"]]
    print("Zero adapter and initial geometric states verified", flush=True)
    loss_reference, retained = retained_gradient(model, features, pristine, prepared, varying, "bfloat16")
    active = gradient_summary(retained)
    model.zero_grad(set_to_none=True)
    replay_points = dict(pristine)
    fresh, _ = collect_points(model, features, pairs=varying)
    replay_points.update(fresh)
    loss_replay, _, adjoints = output_adjoints(replay_points, prepared, "P4", 0, varying)
    replay_backward(model, features, adjoints)
    replayed = gradients(model)
    comparison = compare_gradients(retained,replayed)
    comparison.update(loss_retained=loss_reference, loss_replay=float(loss_replay))
    if float(loss_replay) != loss_reference or comparison["relative_l2"] > spec["replay_relative_gradient_l2_max"] or comparison["cosine"] < spec["replay_cosine_min"]:
        dump(attempt / "replay_failure.json", comparison)
        raise ValueError("Retained-graph and exact-state replay gradients disagree")
    del retained, replayed, adjoints, fresh, replay_points
    model.zero_grad(set_to_none=True)
    print("Real-model retained/replay gradients verified", flush=True)

    configure_runtime(allow_tf32=False)
    _, analytical = retained_gradient(model, features, pristine, prepared, varying, "float32")
    gradient_summary(analytical)
    selected = max(analytical, key=lambda name:float(analytical[name].abs().max()))
    index = int(analytical[selected].abs().argmax())
    derivative = float(analytical[selected].reshape(-1)[index])
    parameter = dict(model.named_parameters())[selected]
    original_value = parameter.detach().reshape(-1)[index].clone()
    fd_rows = []
    model.zero_grad(set_to_none=True)
    del analytical
    for epsilon in spec["finite_difference_steps"]:
        losses = []
        for sign in (-1,1):
            with torch.no_grad():
                parameter.reshape(-1)[index].copy_(original_value+sign*epsilon)
                point_values = dict(pristine)
                perturbation, _ = collect_points(model,features,precision="float32",pairs=varying)
                point_values.update(perturbation)
                value, _ = training_loss(point_values, prepared, "P4", 0)
                losses.append(float(value))
            del point_values, perturbation, value
        with torch.no_grad():
            parameter.reshape(-1)[index].copy_(original_value)
        difference = (losses[1]-losses[0])/(2*epsilon)
        absolute = abs(difference-derivative)
        tolerance = spec["finite_difference_absolute_tolerance"]+spec["finite_difference_relative_tolerance"]*abs(derivative)
        fd_rows.append(dict(epsilon=epsilon, analytical=derivative, finite_difference=difference,
                            absolute_error=absolute, tolerance=tolerance, pass_check=absolute<=tolerance))
    finite_pass = sum(row["pass_check"] for row in fd_rows)>=spec["finite_difference_minimum_passes"]
    finite_pass = finite_pass and (fd_rows[-1]["pass_check"] or not spec["finite_difference_smallest_step_must_pass"])
    dump(attempt / "finite_differences.json", dict(parameter=selected, flat_index=index, rows=fd_rows, pass_all=finite_pass))
    if not finite_pass:
        raise ValueError("Real LoRA directional finite differences failed the declared tolerances")
    print("Real LoRA directional derivative verified", flush=True)

    configure_runtime()
    model.zero_grad(set_to_none=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    before_adapters = [p.detach().cpu().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters,lr=spec["full_macro_step"]["learning_rate"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    full_points, _ = collect_points(model, features)
    full_loss, scales, full_adjoints = output_adjoints(full_points, prepared, "P4", 0)
    replay_backward(model, features, full_adjoints)
    full_gradients = gradients(model)
    full_summary = gradient_summary(full_gradients)
    unclipped = torch.nn.utils.clip_grad_norm_(parameters, spec["full_macro_step"]["gradient_clip_norm"],error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    changed = sum(not torch.equal(old,new.detach().cpu()) for old,new in zip(before_adapters,parameters))
    if changed == 0 or not all(torch.isfinite(p).all() for p in parameters):
        raise ValueError("Full macro-step did not update finite LoRA factors")
    final_hash = digest(original)
    if final_hash != original_hash:
        raise ValueError("Original model state changed during LoRA diagnostics")
    if sources != source_hashes(("scripts/check_profiled_lora_bridge.py",)) or inputs != input_hashes():
        raise ValueError("Source or inputs changed during the gradient bridge")
    verify_runtime()
    report = dict(pass_all=True, scope="Engineering model-gradient bridge; no reconstruction accuracy conclusion",
        runtime_policy=arithmetic, finite_difference_matmul_and_cudnn_tf32=False,
        original_state_unchanged=True, zero_adapter_pass=zero_exact, replay_gradient_pass=True,
        finite_difference_pass=finite_pass, full_macro_step_pass=True,
        original_state_sha256=original_hash, final_original_state_sha256=final_hash,
        adapter_modules=len(modules), trainable_parameters=sum(p.numel() for p in parameters),
        replay=comparison, two_pair_gradient=active, finite_difference_parameter=selected,
        finite_difference_flat_index=index, finite_difference_rows=fd_rows,
        full_step=dict(loss=float(full_loss), scales=scales.cpu().tolist(), seconds=seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, parameters_changed=changed,
            unclipped_gradient_norm=float(unclipped), gradient=full_summary,
            graph_free_decoder_calls=20, differentiable_decoder_calls=20, optimizer_steps=1),
        source_sha256_lf=sources,
        input_sha256=inputs, threshold_manifest_sha256=sha(SETTINGS),
        engineering_manifest_sha256=sha(args.manifest),
        execution_commit=subprocess.check_output(["git","-c","safe.directory=*","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        docker_image=environment["docker_image"], checkpoint_sha256=WEIGHTS,
        attempt=attempt.relative_to(ROOT).as_posix(), rental_compute_usd=0)
    dump(attempt / "report.json", report)
    dump(PUBLIC / "gradient_bridge.json", report)
    print(json.dumps(dict(pass_all=True,replay=comparison,finite_differences=fd_rows,
        full_step_seconds=seconds,peak_allocated_mib=report["full_step"]["peak_allocated_mib"])),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,default=Path("/weights"))
    parser.add_argument("--manifest",type=Path,default=ROOT/"manifests/lora_profiled_engineering_v1.json")
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--image",required=True,help="Docker image ID inspected by the launching host")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.attempt):
        parser.error("Attempt ID must contain only letters, digits, underscore or hyphen")
    attempt = PUBLIC / "engineering" / args.attempt
    if (PUBLIC / "gradient_bridge.json").exists():
        raise FileExistsError("A successful gradient bridge already exists; verify its provenance before rerunning")
    attempt.mkdir(parents=True,exist_ok=False)
    try:
        run(args,attempt)
    except Exception as error:
        dump(attempt/"failure.json",dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),rental_compute_usd=0))
        raise
