"""Real MASt3R adapters and exact same-state output-adjoint replay.

The encoder and original decoder/head weights are frozen. Only LoRA is updated.
No evaluation packet is loaded by these training helpers.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from mast3r_network_capture import training_features, views_for
from mast3r_network_priors import objective as legacy_objective
from profiled_baseline_graph import load_baseline_graph
from profiled_calibration import canonical_depth, prepare_calibration, profiled_objective
from run_mast3r_neural_transfer import intrinsic, schedule


ROOT = Path(__file__).resolve().parents[1]
PAIRS = tuple(itertools.permutations(range(5), 2))
FRAMES = (25, 225, 425, 625, 825)


def sha(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case(scene, side, device="cuda"):
    """Load exactly one permitted temporal packet and its sealed spatial folds."""
    if scene not in ("heads", "chess", "fire") or side not in ("earlier", "later"):
        raise ValueError("Unknown registered scene/side")
    public = ROOT / "results/lora_profiled_v1"
    record = json.loads((public / "inputs.json").read_text())
    registration = record["registration"]
    split_path = public / "splits.npz"
    if sha(split_path) != record["splits_binary_sha256"]:
        raise ValueError("Sealed spatial folds changed")
    for name, expected in registration["immutable_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Immutable input changed: {name}")
    packet_path = ROOT / f"results/mast3r_partner_shape/{side}.npz"
    if sha(packet_path) != registration["packet_sha256"][side]:
        raise ValueError("Permitted adaptation packet changed")
    samples, folds = [], []
    with np.load(packet_path, allow_pickle=False) as packet, np.load(split_path, allow_pickle=False) as split:
        for frame in FRAMES:
            prefix = f"{scene}_{side}_{frame}"
            rows, xy = split[prefix+"__rows"], split[prefix+"__xy"]
            values = packet[f"{scene}_{frame}"][rows]
            samples.append(dict(xy=torch.as_tensor(xy.copy(), dtype=torch.long, device=device),
                                depth=torch.as_tensor(values[:,2].copy(), dtype=torch.float64, device=device)))
            folds.append(torch.as_tensor(split[prefix+"__fold"].copy(), dtype=torch.long, device=device))
    with np.load(ROOT / "results/mast3r_neural_transfer/pose_packets.npz") as poses:
        w2c = poses[scene+"_clean"].copy()
    return dict(scene=scene, side=side, samples=samples, folds=folds,
                K=torch.as_tensor(intrinsic(), dtype=torch.float64, device=device),
                c2w=torch.as_tensor(np.linalg.inv(w2c), dtype=torch.float64, device=device),
                w2c=torch.as_tensor(w2c, dtype=torch.float64, device=device),
                packet_sha256=sha(packet_path), splits_sha256=sha(split_path))


def partition(case, support_fold):
    support, query = [], []
    for observation, labels in zip(case["samples"], case["folds"]):
        for result, mask in ((support, labels == support_fold), (query, labels != support_fold)):
            if int(mask.sum()) < 8:
                raise ValueError("Insufficient sealed support/query observations")
            result.append({key: value[mask] for key, value in observation.items()})
    return support, query


def pair_forward(model, features, pair, precision="bfloat16"):
    """Return both point branches; decoder arithmetic is explicit for diagnostics."""
    i, j = pair
    first, pfirst, sfirst = features[i]
    second, psecond, ssecond = features[j]
    if precision == "float32":
        first, second = first.float(), second.float()
    elif precision != "bfloat16":
        raise ValueError("Unsupported registered precision")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
        decoder1, decoder2 = model._decoder(first, pfirst, second, psecond)
    with torch.autocast("cuda", enabled=False):
        out1 = model._downstream_head(1, [v.float() for v in decoder1], sfirst)
        out2 = model._downstream_head(2, [v.float() for v in decoder2], ssecond)
    return (out1["pts3d"][0], out2["pts3d"][0]), out1["conf"][0]


def collect_points(model, features, precision="bfloat16", pairs=PAIRS):
    points, confidence = {}, {}
    with torch.no_grad():
        for pair in pairs:
            outputs, conf = pair_forward(model, features, pair, precision)
            # Clone narrow tensors so a view cannot keep the descriptor buffer.
            points[pair] = tuple(value.detach().clone() for value in outputs)
            confidence[pair] = conf.detach().clone()
    return points, confidence


def fuse(points, confidence):
    depths, fused_confidence = [], []
    for i in range(5):
        partners = [(i,j) for j in range(5) if j != i]
        depth, conf = canonical_depth(torch.stack([points[p][0] for p in partners]),
                                      torch.stack([confidence[p] for p in partners]))
        depths.append(depth)
        fused_confidence.append(conf)
    return torch.stack(depths), torch.stack(fused_confidence)


def prepare_training(case, initial_points, confidence):
    initial_depth, fused_confidence = fuse(initial_points, confidence)
    matches, graph_audit = load_baseline_graph(case["scene"], device=initial_depth.device)
    states, queries = [], []
    for fold in range(2):
        support, query = partition(case, fold)
        states.append(prepare_calibration(initial_depth, case["K"], case["c2w"],
                                          support, matches, fused_confidence))
        queries.append(query)
    # P1 deliberately preserves the historical full-packet float32 median rule.
    gauges, units, support = {}, [], []
    for i, sample in enumerate(case["samples"]):
        xy, measured = sample["xy"], sample["depth"].float()
        initial = [initial_points[i,j][0][xy[:,1],xy[:,0],2] for j in range(5) if j != i]
        if not torch.stack([torch.isfinite(z) & (z > 0) for z in initial]).all():
            raise ValueError("Nonpositive initial P1 sensor prediction; register a shared prospective mask")
        support.append((xy.T, measured))
        units.append(measured.median().detach())
        for j in range(5):
            if j != i:
                z = initial_points[i,j][0][xy[:,1],xy[:,0],2]
                gauges[i,j] = (measured / z).median().detach()
    y, x = np.mgrid[8:384:16, 8:512:16]
    rays = np.stack([x,y,np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsic()).T
    relatives = {(i,j):(case["w2c"][j] @ case["c2w"][i]).float() for i,j in PAIRS}
    return dict(states=states, queries=queries, confidence=confidence, graph_audit=graph_audit,
                gauges=gauges, units=units, support=support, relatives=relatives,
                rays=torch.as_tensor(rays, dtype=torch.float32, device=initial_depth.device))


def training_loss(points, prepared, arm, step):
    if arm == "P1":
        terms = []
        for i,j,k in schedule(20):
            outputs = [({"pts3d":points[p][0][None]}, {"pts3d_in_other_view":points[p][1][None]})
                       for p in ((i,j),(i,k))]
            value, _ = legacy_objective(outputs, [prepared["gauges"][i,j], prepared["gauges"][i,k]],
                                        *prepared["support"][i], prepared["units"][i],
                                        [prepared["relatives"][i,j], prepared["relatives"][i,k]],
                                        prepared["rays"], 0., 0.)
            terms.append(value)
        return torch.stack(terms).mean(), None
    if arm not in ("P2", "P3", "P4", "P5"):
        raise ValueError("P0 does not train; expected P1 through P5")
    depths, _ = fuse(points, prepared["confidence"])
    fold = step % 2
    return profiled_objective(depths, prepared["states"][fold], prepared["queries"][fold],
                              graph_weight=.1 if arm in ("P4", "P5") else 0.,
                              detach_solve=arm in ("P2", "P5"))


def output_adjoints(points, prepared, arm, step, varying_pairs=PAIRS):
    leaves = dict(points)
    variables = []
    for pair in varying_pairs:
        leaves[pair] = tuple(value.detach().requires_grad_(True) for value in points[pair])
        variables.extend(leaves[pair])
    loss, scales = training_loss(leaves, prepared, arm, step)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite profiled training objective")
    gradients = torch.autograd.grad(loss, variables, allow_unused=True)
    adjoints = {pair:gradients[2*i:2*i+2] for i,pair in enumerate(varying_pairs)}
    for values in adjoints.values():
        for value in values:
            if value is not None and not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite output adjoint")
    return loss.detach(), None if scales is None else scales.detach(), adjoints


def replay_backward(model, features, adjoints, precision="bfloat16"):
    """Accumulate into existing parameter gradients; never take an optimizer step."""
    for pair, gradients in adjoints.items():
        outputs, _ = pair_forward(model, features, pair, precision)
        active = [(output, gradient) for output,gradient in zip(outputs,gradients) if gradient is not None]
        if active:
            torch.autograd.backward([out for out,_ in active], [grad for _,grad in active])


def source_hashes(extra=()):
    names = ("scripts/profiled_network.py", "scripts/profiled_calibration.py", "scripts/profiled_baseline_graph.py",
             "scripts/mast3r_network_capture.py", "scripts/mast3r_network_priors.py",
             "scripts/run_mast3r_neural_transfer.py", "third_party/TCO/peft/lora.py", *extra)
    return {name:sha(ROOT/name, text=True) for name in names}


def input_hashes():
    names = ("manifests/lora_profiled_design_v1.json", "results/lora_profiled_v1/inputs.json",
             "results/lora_profiled_v1/splits.npz", "results/lora_profiled_v1/baseline_graph.json",
             "manifests/lora_profiled_bridge_checks_v1.json")
    return {name:sha(ROOT/name) for name in names}
