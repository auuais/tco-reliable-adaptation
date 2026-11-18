"""Generate one P3 direction and select its scale with reserved adaptation depth."""
import argparse
import math
import os
import time

import numpy as np
import torch

from mast3r_network_capture import training_features
from mast3r_network_priors import digest
from native_lora_step_inputs import (EXPECTED_RUNTIME, RAW, ROOMS, SIDES, STATES, subset,
                                     verify_registration)
from prepare_profiled_inputs import read, sha
from profiled_calibration import prepare_calibration
from profiled_network import PAIRS, collect_points, fuse, output_adjoints, partition, replay_backward
from profiled_runtime import configure_runtime, verify_runtime
from run_native_lora_direction import (WEIGHTS, assign, model_and_factors, native_endpoint, save_arrays,
                                       snapshot)
from run_profiled_d0 import write_new
from run_profiled_lora import counted_decoder_call, scientific_gpu_lock
from score_native_lora_direction import data_packet, evaluate, means
from score_native_lora_step import choose


def load_permitted(registration, env, side):
    record = registration["inputs"]["environments"][env]
    inputs = RAW / "intake" / env
    image_path = inputs / "images.npz"
    if sha(image_path) != record["image_sha256"]:
        raise ValueError("Sealed RGB archive changed")
    with np.load(image_path, allow_pickle=False) as archive:
        frames, k, pose = (archive[key] for key in ("images", "K", "c2w"))
    rgb = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
    names = [f"step_{env}_{record['image_sha256'][:12]}_{f}" for f in record["target_frames"]]
    views = [dict(img=image[None], true_shape=np.array([[384, 512]], dtype=np.int32),
                  idx=i, instance=names[i]) for i, image in enumerate(rgb)]
    packet = data_packet(inputs / f"{side}.npz", record["packets"][side]["sha256"])
    observed = [[int((p["fold"] == fold).sum()) for p in packet] for fold in range(3)]
    if observed != record["packets"][side]["counts"]:
        raise ValueError("Permitted observation counts changed")
    fitting = subset(packet, [p["fold"] < 2 for p in packet])
    samples = [{key: torch.from_numpy(p[key]) for key in ("xy", "depth")} for p in fitting]
    case = dict(samples=samples, folds=[torch.from_numpy(p["fold"]) for p in fitting],
                K=torch.from_numpy(k), c2w=torch.from_numpy(pose), w2c=torch.from_numpy(np.linalg.inv(pose)))
    return case, (dict(sha256=record["image_sha256"]), names, views), packet


def learn_direction(model, factors, case, views, output):
    gpu_case = dict(samples=[{key: value.cuda() for key, value in sample.items()} for sample in case["samples"]],
                    folds=[f.cuda() for f in case["folds"]], K=case["K"].cuda(), c2w=case["c2w"].cuda())
    support, query = partition(gpu_case, 0)
    features = training_features(model, views)
    (points, conf), teacher_calls = counted_decoder_call(model, collect_points, model, features)
    teacher, confidence = fuse(points, conf)
    # The sensor-only P3 loss gives the graph term zero weight.
    calibration = prepare_calibration(teacher, gpu_case["K"], gpu_case["c2w"], support, [], confidence)
    loss, _, gradients = output_adjoints(points, dict(states=[calibration], queries=[query], confidence=conf),
                                         "P3", 0)
    del points, conf, teacher, confidence
    _, replay_calls = counted_decoder_call(model, replay_backward, model, features, gradients)
    if teacher_calls != len(PAIRS) or replay_calls != len(PAIRS):
        raise ValueError("Direction must use all 20 directed pairs for both passes")
    parameters = [p for _, p in factors]
    adam = torch.optim.Adam(parameters, lr=.0001)
    before = snapshot(factors)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    adam.step()
    adam.zero_grad(set_to_none=True)
    after = snapshot(factors)
    displacement = math.fsum(float((after[name].double() - value.double()).square().sum())
                             for name, value in before.items())
    if displacement <= 0 or not math.isfinite(displacement):
        raise ValueError("Invalid first-step direction")
    for name, state in (("before", before), ("after", after)):
        save_arrays(output / f"{name}.npz", {key: value.numpy() for key, value in state.items()})
    evidence = dict(teacher_calls=teacher_calls, replay_calls=replay_calls, optimizer_steps=1,
                    loss=float(loss), gradient_norm=float(norm), adapter_l2=math.sqrt(displacement),
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    factor_files={name: sha(output / f"{name}.npz") for name in ("before", "after")})
    return before, after, evidence


def run(env, side, attempt):
    configuration = verify_registration(committed=True)
    if sha(WEIGHTS / "model.safetensors") != configuration["checkpoint_sha256"] or (
            sha(WEIGHTS / "config.json") != configuration["config_sha256"]):
        raise ValueError("Pinned checkpoint or config changed")
    if os.environ.get("PROFILED_D0_IMAGE_ID") != configuration["image"]:
        raise ValueError("Expected the pinned local runtime")
    configure_runtime()
    ref = read(EXPECTED_RUNTIME)
    if (torch.__version__, torch.version.cuda, torch.cuda.get_device_name()) != (ref["torch"], ref["cuda"], ref["gpu"]):
        raise ValueError("Unexpected Torch, CUDA or GPU")
    output = RAW / "cases" / f"{env}_{side}" / attempt
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    fitting, bundle, permitted = load_permitted(configuration, env, side)
    model, factors, original, original_sha = model_and_factors()
    before, after, direction = learn_direction(model, factors, fitting, bundle[2], output)
    torch.cuda.empty_cache()
    rows = {}
    for state, multiplier in STATES.items():
        assign(factors, before, after, multiplier)
        initial_factors = digest(factors)
        dest = output / state
        dest.mkdir()
        torch.cuda.reset_peak_memory_stats()
        g0, tree, graph = native_endpoint(model, fitting, bundle, dest / "cache", dest)
        if digest(factors) != initial_factors or digest(original) != original_sha:
            raise ValueError("Adapter or frozen backbone changed during native alignment")
        with np.load(dest / "native.npz", allow_pickle=False) as archive:
            depth = archive["depth"]
            if float(archive["g0"]) != g0:
                raise ValueError("Native metric gauge changed")
        selector = evaluate(depth, permitted, [p["fold"] == 2 for p in permitted], g0,
                            configuration["min_partition"])
        if any(row["mae_mm"] is None for row in selector):
            raise ValueError("A fixed selector depth is invalid or missing")
        record = dict(state=state, multiplier=multiplier, adapter_sha256=initial_factors, g0=g0,
                      native_sha256=sha(dest / "native.npz"), trace_sha256=sha(dest / "trace.json"),
                      tree=tree, match_edges=graph, selector_rows=selector, selector_mae_mm=means(selector),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        write_new(dest / "complete.json", record)
        rows[state] = record
        print(env, side, state, "native alignment complete", flush=True)
    selected = choose({state: row["selector_mae_mm"] for state, row in rows.items()})
    verify_runtime()
    write_new(output / "choice.json", dict(env=env, side=side, attempt=attempt, chosen=selected,
              direction=direction, states=rows, original_sha256=original_sha, frozen_weights=True,
              role="selection uses this direction's reserved adaptation partition only",
              wall_seconds=time.monotonic()-start, rental_compute_usd=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", choices=ROOMS)
    parser.add_argument("side", choices=SIDES)
    parser.add_argument("--attempt", choices=("attempt01", "attempt02"), default="attempt01")
    args = parser.parse_args()
    with scientific_gpu_lock(dict(study="native-lora-step", room=args.env, side=args.side)):
        run(args.env, args.side, args.attempt)
