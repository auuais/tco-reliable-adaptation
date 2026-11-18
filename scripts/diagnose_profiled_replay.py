"""Separate decoder arithmetic and backward accumulation in a failed bridge.

This diagnostic does not relax acceptance thresholds or update model weights.
It preserves the original joint-backward failure and compares all relevant paths.
"""
import argparse
import inspect
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R

from check_profiled_lora_bridge import compare_gradients, dump, gradients, retained_gradient
from mast3r_network_priors import digest, install
from profiled_network import (ROOT, collect_points, input_hashes, load_case, output_adjoints,
                              pair_forward, prepare_training, replay_backward, sha, source_hashes,
                              training_features, training_loss, views_for)
from capture_mast3r_transfer import WEIGHTS


def tensors_equal(first, second):
    if first is None or second is None:
        return dict(exact=first is second, both_unused=first is None and second is None)
    first, second = first.detach().cpu(), second.detach().cpu()
    if first.shape != second.shape or first.dtype != second.dtype:
        return dict(exact=False, shape_or_dtype_differs=True)
    delta = first.double()-second.double()
    return dict(exact=bool(torch.equal(first, second)), dtype=str(first.dtype), shape=list(first.shape),
                max_absolute=float(delta.abs().max()),
                relative_l2=float(delta.norm()/max(float(first.double().norm()),1e-30)))


def snapshot(values):
    return tuple(None if value is None else value.detach().cpu().clone() for value in values)


def sum_gradients(values):
    # Explicit FP32 host sum, separately identified from production GPU .grad
    # accumulation. Per-pair comparisons do not depend on this sum.
    return {name:sum((item[name] for item in values), torch.zeros_like(values[0][name]))
            for name in values[0]}


def backward_with_checked_adjoints(outputs, adjoints):
    supplied = tuple(None if value is None else value.detach().to(outputs[i].device).clone()
                     for i,value in enumerate(adjoints))
    before = snapshot(supplied)
    active = [(output,gradient) for output,gradient in zip(outputs,supplied) if gradient is not None]
    if active:
        torch.autograd.backward([output for output,_ in active], [gradient for _,gradient in active])
    return [tensors_equal(old,new) for old,new in zip(before,supplied)]


def sequential_retained(model, features, constants, prepared, varying, precision):
    """Collect graph-cut evidence and each retained VJP, then release the graphs."""
    model.zero_grad(set_to_none=True)
    points = dict(constants)
    outputs = []
    for pair in varying:
        points[pair], _ = pair_forward(model, features, pair, precision)
        outputs.extend(points[pair])
    loss, _ = training_loss(points, prepared, "P4", 0)
    adjoints = torch.autograd.grad(loss, outputs, allow_unused=True, retain_graph=True)
    value = float(loss.detach())
    point_copies = {pair:snapshot(points[pair]) for pair in varying}
    adjoint_copies = {pair:snapshot(adjoints[2*i:2*i+2]) for i,pair in enumerate(varying)}
    # The coupled loss graph is no longer needed. Decoder graphs are independent
    # apart from leaf parameters and detached encoder features.
    del loss, adjoints, outputs
    per_pair, mutations = {}, {}
    for index, pair in enumerate(varying):
        model.zero_grad(set_to_none=True)
        mutations[pair] = backward_with_checked_adjoints(points[pair],adjoint_copies[pair])
        per_pair[pair] = gradients(model)
        del points[pair]
    del points
    model.zero_grad(set_to_none=True)
    # Only CPU copies leave this function, so fresh replay cannot overlap a
    # retained decoder graph and inflate the 3080 memory requirement.
    return value, sum_gradients(list(per_pair.values())), point_copies, adjoint_copies, per_pair, mutations


def replay(model, features, constants, prepared, varying, precision):
    model.zero_grad(set_to_none=True)
    points = dict(constants)
    current, _ = collect_points(model, features, precision=precision, pairs=varying)
    points.update(current)
    loss, _, adjoints = output_adjoints(points, prepared, "P4", 0, varying)
    value = float(loss)
    point_copies = {pair:snapshot(current[pair]) for pair in varying}
    adjoint_copies = {pair:snapshot(adjoints[pair]) for pair in varying}
    supplied = {pair:tuple(None if gradient is None else gradient.detach().clone() for gradient in values)
                for pair,values in adjoints.items()}
    replay_backward(model, features, supplied, precision=precision)
    aggregate = gradients(model)
    aggregate_mutations = {pair:[tensors_equal(old,new) for old,new in zip(adjoint_copies[pair],supplied[pair])]
                           for pair in varying}
    del loss, points, current, adjoints, supplied
    model.zero_grad(set_to_none=True)
    per_pair, mutations, forward_equal = {}, {}, {}
    for pair in varying:
        model.zero_grad(set_to_none=True)
        outputs, _ = pair_forward(model, features, pair, precision)
        forward_equal[pair] = [tensors_equal(old,new) for old,new in zip(point_copies[pair],outputs)]
        mutations[pair] = backward_with_checked_adjoints(outputs,adjoint_copies[pair])
        per_pair[pair] = gradients(model)
        del outputs
    model.zero_grad(set_to_none=True)
    return value,aggregate,point_copies,adjoint_copies,per_pair,mutations,aggregate_mutations,forward_equal


def run(args):
    target = ROOT / "results/lora_profiled_v1/engineering" / args.attempt
    target.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    environment = json.loads((ROOT/"results/lora_profiled_v1/environment.json").read_text())
    if args.image != environment["docker_image"] or sha(args.weights/"model.safetensors") != WEIGHTS:
        raise ValueError("Runtime image or checkpoint differs")
    sources = source_hashes(("scripts/check_profiled_lora_bridge.py", "scripts/diagnose_profiled_replay.py"))
    inputs = input_hashes()
    torch.manual_seed(42); np.random.seed(42)
    # Match the failed bridge's pretrained BF16 teacher and frozen normalizers.
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AsymmetricMASt3R.from_pretrained(str(args.weights),local_files_only=True).eval().cuda().requires_grad_(False)
    original = list(model.state_dict().items()); base_hash = digest(original)
    case = load_case("heads","earlier")
    _,_,views = views_for("heads_base")
    features = training_features(model,views)
    initial,confidence = collect_points(model,features)
    torch.manual_seed(42); install(model,ROOT/"third_party/TCO/peft/lora.py")
    adapter_state = [(name,value) for name,value in model.named_parameters() if value.requires_grad]
    adapter_hash = digest(adapter_state)
    prepared = prepare_training(case,initial,confidence)
    varying = [(0,1),(0,2)]
    report = dict(scope="Arithmetic diagnostic only; no parameter update or acceptance-threshold change",
                  source_sha256_lf=sources,input_sha256=inputs,comparisons={},
                  initial_teacher=dict(precision="bfloat16 decoder / float32 heads",allow_tf32=True,collections=1),
                  rope=dict(module=type(model.dec_blocks[0].attn.rope).__module__,
                            class_name=type(model.dec_blocks[0].attn.rope).__qualname__,
                            source_file=inspect.getfile(type(model.dec_blocks[0].attn.rope))),
                  per_pair_sum="CPU float32 diagnostic sum; production replay uses its original GPU .grad accumulation")
    for precision in ("bfloat16","float32"):
        torch.backends.cuda.matmul.allow_tf32 = precision == "bfloat16"
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        loss_a, first = retained_gradient(model,features,initial,prepared,varying,precision)
        loss_b, repeated = retained_gradient(model,features,initial,prepared,varying,precision)
        retained_details = sequential_retained(model,features,initial,prepared,varying,precision)
        loss_c,sequential,retained_points,retained_adjoints,retained_pairs,retained_mutations = retained_details
        torch.cuda.empty_cache()
        replay_details = replay(model,features,initial,prepared,varying,precision)
        loss_d,replayed,replay_points,replay_adjoints,replay_pairs,replay_mutations,aggregate_mutations,replay_forward = replay_details
        local = dict(losses=[loss_a,loss_b,loss_c,loss_d],
                     joint_repeat=compare_gradients(first,repeated),
                     joint_vs_replay=compare_gradients(first,replayed),
                     sequential_vs_replay=compare_gradients(sequential,replayed),
                     joint_vs_sequential=compare_gradients(first,sequential),
                     replay_gpu_accumulation_vs_cpu_pair_sum=compare_gradients(replayed,sum_gradients(list(replay_pairs.values()))),
                     per_pair={f"{i}_{j}":compare_gradients(retained_pairs[i,j],replay_pairs[i,j]) for i,j in varying},
                     graph_cut_outputs={f"{i}_{j}":[tensors_equal(a,b) for a,b in zip(retained_points[i,j],replay_points[i,j])] for i,j in varying},
                     graph_cut_adjoints={f"{i}_{j}":[tensors_equal(a,b) for a,b in zip(retained_adjoints[i,j],replay_adjoints[i,j])] for i,j in varying},
                     supplied_adjoint_unchanged={
                         "retained_per_pair":{f"{i}_{j}":retained_mutations[i,j] for i,j in varying},
                         "replayed_per_pair":{f"{i}_{j}":replay_mutations[i,j] for i,j in varying},
                         "production_replay":{f"{i}_{j}":aggregate_mutations[i,j] for i,j in varying}},
                     recomputed_grad_enabled_outputs={f"{i}_{j}":replay_forward[i,j] for i,j in varying})
        by_parameter=[]
        for name in first:
            difference = float((first[name].double()-replayed[name].double()).square().sum())
            if difference:
                by_parameter.append(dict(name=name,squared_difference=difference,
                                         reference_norm=float(first[name].double().norm())))
        local["largest_parameter_differences"] = sorted(by_parameter,key=lambda x:-x["squared_difference"])[:12]
        report["comparisons"][precision] = local
        dump(target / (precision+".json"), local)
        print(json.dumps({precision:local}),flush=True)
        del first,repeated,sequential,replayed,retained_details,replay_details
        del retained_points,retained_adjoints,retained_pairs,retained_mutations
        del replay_points,replay_adjoints,replay_pairs,replay_mutations,aggregate_mutations,replay_forward
    report.update(original_state_unchanged=digest(original)==base_hash,
                  adapter_state_unchanged=digest(adapter_state)==adapter_hash,
                  seconds=time.perf_counter()-start,docker_image=args.image,checkpoint_sha256=WEIGHTS,
                  execution_commit=subprocess.check_output(["git","-c","safe.directory=*","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
                  rental_compute_usd=0)
    if report["source_sha256_lf"] != source_hashes(("scripts/check_profiled_lora_bridge.py", "scripts/diagnose_profiled_replay.py")) or report["input_sha256"] != input_hashes():
        raise ValueError("Source or inputs changed during diagnostic")
    dump(target/"report.json",report)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--weights",type=Path,default=Path("/weights"))
    parser.add_argument("--image",required=True)
    args=parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.attempt):
        parser.error("Invalid attempt ID")
    try:
        run(args)
    except Exception as error:
        target = ROOT / "results/lora_profiled_v1/engineering" / args.attempt
        if target.is_dir() and not (target/"failure.json").exists():
            dump(target/"failure.json",dict(error_type=type(error).__name__,message=str(error),
                                           traceback=traceback.format_exc(),rental_compute_usd=0))
        raise
