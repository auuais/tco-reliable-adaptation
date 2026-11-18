"""Execute a registered frozen-MASt3R one-step direction and native endpoints."""
import argparse
import gc
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from check_profiled_endpoints import load_fresh_graph
from download_stanford12_direction import ENVIRONMENTS
from mast3r_alignment_hook import optimizer_hook
from mast3r_calibrated_alignment import CalibratedPriors
from mast3r_network_capture import training_features
from mast3r_network_priors import digest, install
from native_lora_direction_inputs import RAW, ROOT, SIDES, STATES, WIDTH, HEIGHT
from prepare_profiled_inputs import read, sha
from profiled_calibration import prepare_calibration
from profiled_network import (PAIRS, collect_points, fuse, output_adjoints, partition, replay_backward,
                              training_loss)
from profiled_runtime import configure_runtime, verify_runtime
from run_profiled_d0 import write_new
from run_profiled_lora import counted_decoder_call, scientific_gpu_lock


MANIFEST = ROOT / "manifests/native_lora_direction_execution_v1.json"
REFERENCE = ROOT / "results/lora_profiled_v1/environment.json"
WEIGHTS = Path("/weights")
BRIDGE_REFERENCE = ROOT / "results/mast3r_neural_transfer/zero_adapter_bridge.json"


def save_arrays(path, values):
    with path.open("xb") as output:
        np.savez_compressed(output, **values)


def model_and_factors():
    from mast3r.model import AsymmetricMASt3R

    torch.manual_seed(42)
    np.random.seed(42)
    model = AsymmetricMASt3R.from_pretrained(str(WEIGHTS), local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    original = list(model.state_dict().items())
    expected = read(BRIDGE_REFERENCE)["original_state_hash"]
    if digest(original) != expected:
        raise ValueError("Loaded backbone differs from the historical verified model")
    torch.manual_seed(42)
    modules = install(model, ROOT / "third_party/TCO/peft/lora.py")
    model.eval()
    factors = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if (len(modules) != 144 or sum(p.numel() for _, p in factors) != 1032192 or
            any(not n.endswith((".lora_A", ".lora_B")) for n, _ in factors) or
            any(bool(p.any()) for n, p in factors if n.endswith(".lora_B"))):
        raise ValueError("Unexpected decoder LoRA inventory or nonzero initial B")
    return model, factors, original, expected


def role_case(registration, environment, side, device):
    entry = registration["inputs"]["environments"][environment]
    inputs = RAW / registration["intake"] / environment
    image_path, packet_path = inputs / "images.npz", inputs / f"{side}.npz"
    if sha(image_path) != entry["image_sha256"] or sha(packet_path) != entry["packets"][side]["sha256"]:
        raise ValueError("Bound image or permitted temporal packet changed")
    with np.load(image_path, allow_pickle=False) as data:
        frames, intrinsic, c2w = (data[k] for k in ("images", "K", "c2w"))
    rgb = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
    names = [f"{environment}_{entry['image_sha256'][:12]}_{n}" for n in entry["target_frames"]]
    views = [dict(img=image[None], true_shape=np.array([[HEIGHT, WIDTH]], dtype=np.int32),
                  idx=i, instance=names[i]) for i, image in enumerate(rgb)]
    samples, folds = [], []
    with np.load(packet_path, allow_pickle=False) as data:
        for i in range(5):
            xy, depth, fold = (data[f"{k}_{i}"] for k in ("xy", "depth", "fold"))
            samples.append(dict(xy=torch.as_tensor(xy, device=device),
                                depth=torch.as_tensor(depth, device=device)))
            folds.append(torch.as_tensor(fold, device=device))
    case = dict(samples=samples, folds=folds, K=torch.as_tensor(intrinsic, device=device),
                c2w=torch.as_tensor(c2w, device=device), w2c=torch.as_tensor(np.linalg.inv(c2w), device=device))
    bundle = (dict(sha256=entry["image_sha256"]), names, views)
    return case, bundle


def snapshot(factors):
    return {name: value.detach().cpu().clone() for name, value in factors}


@torch.no_grad()
def assign(factors, before, after, multiplier):
    for name, value in factors:
        a, b = before[name], after[name]
        if multiplier == 0:
            changed = a
        elif multiplier == 1:
            changed = b
        else:
            changed = a.double() + multiplier * (b.double() - a.double())
        value.copy_(changed)


def displacement(factors, before):
    after = snapshot(factors)
    squared_factor, squared_update = 0., 0.
    for name, value in after.items():
        squared_factor += float((value.double() - before[name].double()).square().sum())
        if name.endswith(".lora_B"):
            a = name.removesuffix("B") + "A"
            old_a, old_b = before[a].double(), before[name].double()
            new_a, new_b = after[a].double(), value.double()
            # First-step B starts at zero. Trace of low-rank Gram products.
            if bool(old_b.any()):
                raise ValueError("Expected pristine zero B for effective update diagnostic")
            squared_update += float(((new_b.T @ new_b) * (new_a @ new_a.T)).sum()) / 16
    return dict(adapter_l2=math.sqrt(squared_factor), effective_weight_l2=math.sqrt(max(0., squared_update)),
                adapter_sha256=digest(factors))


def train(registration, environment, side, output):
    case, (_, _, views) = role_case(registration, environment, side, "cuda")
    support, query = partition(case, 0)
    model, factors, original, expected = model_and_factors()
    features = training_features(model, views)
    (points, conf), calls = counted_decoder_call(model, collect_points, model, features)
    if calls != len(PAIRS):
        raise ValueError("Pristine teacher did not visit every directed pair")
    teacher, confidence = fuse(points, conf)
    # P3's registered graph weight is zero. The graph does not enter its solve or loss.
    state = prepare_calibration(teacher, case["K"], case["c2w"], support, [], confidence)
    prepared = dict(states=[state], queries=[query], confidence=conf)
    loss, _, adjoints = output_adjoints(points, prepared, "P3", 0)
    del points, teacher, confidence
    _, replay_calls = counted_decoder_call(model, replay_backward, model, features, adjoints)
    del adjoints
    if replay_calls != len(PAIRS):
        raise ValueError("Gradient replay did not visit every directed pair")
    params = [parameter for _, parameter in factors]
    optimizer = torch.optim.Adam(params, lr=.0001)
    before = snapshot(factors)
    gradients = {name: p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(before[name])
                 for name, p in factors}
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = snapshot(factors)
    delta_dot = math.fsum(float((gradients[name].double() * (after[name].double() - value.double())).sum())
                          for name, value in before.items())
    if not math.isfinite(delta_dot) or not any(not torch.equal(after[name], p) for name, p in before.items()):
        raise ValueError("Nonfinite or zero optimizer direction")
    if not all(bool(torch.isfinite(p).all()) for p in after.values()):
        raise ValueError("Nonfinite updated factor state")
    for label, arrays in (("before", before), ("after", after), ("unclipped_gradient", gradients)):
        save_arrays(output / f"{label}.npz", {name: value.numpy() for name, value in arrays.items()})
    metrics = {}
    for label, multiplier in STATES.items():
        if label.startswith("repeat"):
            continue
        assign(factors, before, after, multiplier)
        with torch.no_grad():
            (fresh, _), forward_calls = counted_decoder_call(model, collect_points, model, features)
            objective, _ = training_loss(fresh, prepared, "P3", 0)
        del fresh
        if forward_calls != len(PAIRS) or not bool(torch.isfinite(objective)):
            raise ValueError("Incomplete or nonfinite training-objective reevaluation")
        metrics[label] = dict(J=float(objective), calls=forward_calls, **displacement(factors, before))
    if digest(original) != expected or metrics["zero"]["J"] != float(loss):
        raise ValueError("Backbone changed or identical pristine BF16 loss differs")
    verify_runtime()
    write_new(output / "direction.json", dict(environment=environment, side=side, model_sha256=expected,
              model_unchanged=True, optimizer_updates=1, teacher_calls=calls, replay_calls=replay_calls,
              unclipped_gradient_norm=float(grad_norm), dot=delta_dot, states=metrics,
              files={f"{k}.npz": sha(output / f"{k}.npz") for k in ("before", "after", "unclipped_gradient")},
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def legacy_prior_rows(samples):
    """Encode native center coordinates for the existing JointPriors input grid."""
    result = []
    for sample in samples:
        xy = sample["xy"].cpu().numpy()
        encoded = (xy + .5) * [518/WIDTH, 392/HEIGHT] - .5
        np.testing.assert_array_equal(np.rint((encoded+.5)*[WIDTH/518, HEIGHT/392]-.5), xy)
        result.append(np.column_stack((encoded, sample["depth"].cpu().numpy())))
    return result


def native_endpoint(model, case, bundle, cache, destination):
    import mast3r.cloud_opt.sparse_ga as alignment
    from dust3r.image_pairs import make_pairs

    _, names, views = bundle
    pairs = make_pairs(views, scene_graph="complete", prefilter=None, symmetrize=True)
    cache.mkdir(exist_ok=False)
    tree = {}

    def factory(arguments):
        root, edges = arguments["mst"]
        tree.update(root=int(root), edges=[[int(a), int(b)] for a, b in edges])
        return CalibratedPriors(alignment, arguments, legacy_prior_rows(case["samples"]),
                                case["w2c"].numpy(), case["K"].numpy(), .01)

    with optimizer_hook(alignment, factory) as callbacks:
        torch.manual_seed(42)
        np.random.seed(42)
        result = alignment.sparse_global_alignment(
            names, pairs, str(cache), model, device="cuda", lr1=.07, niter1=300, lr2=.01,
            niter2=300, opt_depth=True, shared_intrinsics=False, matching_conf_thr=0.)
    if (len(callbacks) != 1 or len(callbacks[0].trace) != 300 or
            callbacks[0].counts != [len(s["xy"]) for s in case["samples"]] or
            not all(r.get("finite_gradient") and r.get("received_gradient") == 1. for r in callbacks[0].trace)):
        raise ValueError("Native alignment prior coverage or fine gradients changed")
    with torch.no_grad():
        points, depths, _ = result.get_dense_pts3d(clean_depth=False)
        depth = torch.stack([z.reshape(HEIGHT, WIDTH) for z in depths])
        cameras, intrinsics = result.cam2w.detach(), result.intrinsics.detach()
        for i, p in enumerate(points):
            actual = (p.reshape(HEIGHT, WIDTH, 3) - cameras[i,:3,3]) @ cameras[i,:3,2]
            torch.testing.assert_close(actual, depth[i], rtol=2e-4, atol=2e-5)
    if not all(bool(t.isfinite().all()) for t in (depth, cameras, intrinsics)):
        raise ValueError("Nonfinite native output")
    array = depth.detach().cpu().numpy()
    ratios = []
    for raster, sample in zip(array, case["samples"]):
        xy = sample["xy"].numpy()
        predicted = raster[xy[:,1], xy[:,0]].astype(np.float64)
        if np.any(predicted <= 0):
            raise ValueError("Nonpositive native depth on permitted gauge samples")
        ratios.append(sample["depth"].numpy() / predicted)
    gauge = float(np.median(np.concatenate(ratios)))
    save_arrays(destination / "native.npz", dict(depth=array, c2w=cameras.cpu().numpy(),
                intrinsics=intrinsics.cpu().numpy(), g0=np.asarray(gauge, dtype=np.float64)))
    _, _, graph = load_fresh_graph(cache, names)
    write_new(destination / "trace.json", dict(tree=tree, fine=callbacks[0].trace, graph=graph))
    return gauge, tree, [dict(i=r["i"], j=r["j"], count=r["count"],
                            sha256=r["pixel_pairs_sha256_int64_le"]) for r in graph["edges"]]


def capture(registration, environment, side, output, state_name):
    case, bundle = role_case(registration, environment, side, "cpu")
    model, factors, original, expected = model_and_factors()
    direction = read(output / "direction.json")
    arrays = []
    for name in ("before", "after"):
        path = output / f"{name}.npz"
        if sha(path) != direction["files"][path.name]:
            raise ValueError("Saved optimizer factor bytes changed")
        with np.load(path, allow_pickle=False) as contents:
            arrays.append({n: torch.from_numpy(contents[n]) for n, _ in factors})
    multiplier = STATES[state_name]
    assign(factors, *arrays, multiplier)
    target = "zero" if state_name == "repeat_zero" else "plus_full" if state_name == "repeat_full" else state_name
    factor_sha = digest(factors)
    if factor_sha != direction["states"][target]["adapter_sha256"]:
        raise ValueError("Perturbed endpoint factors differ from the signed training evaluation")
    if state_name.startswith("repeat"):
        gpu_case, (_, _, views) = role_case(registration, environment, side, "cuda")
        support, query = partition(gpu_case, 0)
        features = training_features(model, views)
        assign(factors, *arrays, 0)
        teacher_points, conf = collect_points(model, features)
        teacher, confidence = fuse(teacher_points, conf)
        state = prepare_calibration(teacher, gpu_case["K"], gpu_case["c2w"], support, [], confidence)
        assign(factors, *arrays, multiplier)
        with torch.no_grad():
            points, _ = collect_points(model, features)
            surrogate_repeat = float(training_loss(points, dict(states=[state], queries=[query],
                                                                  confidence=conf), "P3", 0)[0])
        del gpu_case, support, query, features, teacher_points, conf, teacher, confidence, state, points
        gc.collect()
        torch.cuda.empty_cache()
    else:
        surrogate_repeat = None
    destination = output / state_name
    destination.mkdir(exist_ok=False)
    gauge, tree, edges = native_endpoint(model, case, bundle, destination / "cache", destination)
    if digest(original) != expected or digest(factors) != factor_sha:
        raise ValueError("Alignment changed frozen model or adapter weights")
    verify_runtime()
    write_new(destination / "complete.json", dict(state=state_name, multiplier=multiplier, g0=gauge,
              model_unchanged=True, optimizer_updates=0, surrogate_repeat=surrogate_repeat, tree=tree,
              match_edges=edges, adapter_sha256=factor_sha, native_sha256=sha(destination / "native.npz"),
              trace_sha256=sha(destination / "trace.json"),
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def bridge(attempt):
    """Verify the new invocation against a sealed late-stage engineering endpoint."""
    from mast3r_network_capture import views_for
    from profiled_network import load_case

    destination = RAW / attempt
    destination.mkdir(parents=True, exist_ok=False)
    model, factors, original, expected = model_and_factors()
    factor_sha = digest(factors)
    case = load_case("heads", "earlier", device="cpu")
    with scientific_gpu_lock(dict(study="native-lora-direction-bridge", attempt=attempt)):
        gauge, tree, edges = native_endpoint(model, case, views_for("heads_base"),
                                            destination / "cache", destination)
    if digest(original) != expected or digest(factors) != factor_sha:
        raise ValueError("Engineering endpoint changed model or zero adapter")
    reference = ROOT / "reports/lora_profiled_v1/endpoint_attempts/endpoint01/zero/native.npz"
    if sha(reference) != "4dbf75a2f159168bf83dcf0ed2db6f7e13054857be8821e79f1816d401f5911f":
        raise ValueError("Original engineering endpoint bytes do not match the release")
    with np.load(reference, allow_pickle=False) as old, np.load(destination / "native.npz", allow_pickle=False) as new:
        differences = {name: float(np.max(np.abs(old[name] - new[name]))) for name in old.files}
    if any(differences.values()):
        raise ValueError(f"Zero-LoRA native endpoint differs from historical reference: {differences}")
    verify_runtime()
    write_new(destination / "bridge.json", dict(exact=True, differences=differences, tree=tree, g0=gauge,
              match_edges=edges, original_state_sha256=expected, optimizer_updates=0,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              native_sha256=sha(destination / "native.npz"), reference_sha256=sha(reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("case", "direction", "endpoint", "bridge"))
    parser.add_argument("environment", nargs="?", choices=ENVIRONMENTS)
    parser.add_argument("side", nargs="?", choices=tuple(SIDES))
    parser.add_argument("--state", choices=tuple(STATES))
    parser.add_argument("--bridge-attempt", choices=("bridge01", "bridge02", "bridge03"), default="bridge01")
    args = parser.parse_args()
    if args.mode == "bridge":
        configure_runtime()
        bridge(args.bridge_attempt)
        return
    if not args.environment or not args.side:
        parser.error("A registered room and permitted temporal side are required")
    output = RAW / "cases" / f"{args.environment}_{args.side}"
    manifest = read(MANIFEST)
    if args.mode == "case":
        from native_lora_direction_registration import verify_registration
        verify_registration()
        if sha(WEIGHTS / "model.safetensors") != manifest["checkpoint_sha256"] or (
                sha(WEIGHTS / "config.json") != manifest["checkpoint_config_sha256"]):
            raise ValueError("Checkpoint or config digest differs")
        output.mkdir(parents=True, exist_ok=True)
        commands = [("direction", None)] + [("endpoint", state) for state in STATES]
        with scientific_gpu_lock(dict(study="native-lora-direction", room=args.environment, side=args.side)):
            for mode, state in commands:
                receipt = output / state / "complete.json" if state else output / "direction.json"
                if receipt.exists():
                    record = read(receipt)
                    if state and (sha(receipt.with_name("native.npz")) != record["native_sha256"]
                                  or sha(receipt.with_name("trace.json")) != record["trace_sha256"]):
                        raise ValueError("Completed endpoint digest differs")
                    if not state and any(sha(output / p) != v for p, v in record["files"].items()):
                        raise ValueError("Completed direction digest differs")
                    continue
                argv = [sys.executable, __file__, mode, args.environment, args.side]
                if state:
                    argv += ["--state", state]
                subprocess.run(argv, check=True)
        return
    configure_runtime()
    ref = read(REFERENCE)
    if torch.__version__ != ref["torch"] or torch.version.cuda != ref["cuda"] or (
            torch.cuda.get_device_name() != ref["gpu"]):
        raise ValueError("Local model runtime differs from the pinned reference")
    if args.mode == "direction":
        train(manifest, args.environment, args.side, output)
    elif args.state:
        capture(manifest, args.environment, args.side, output, args.state)
    else:
        parser.error("endpoint requires --state")


if __name__ == "__main__":
    main()
