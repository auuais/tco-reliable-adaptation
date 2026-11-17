"""Fit one source-specific D adapter and native KDP alignment in a fresh process."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R
import mast3r.cloud_opt.sparse_ga as alignment

from mast3r_crossover_cache import validate_pairs
from prepare_partner_sources import FRAMES, ROOT, PUBLIC, RAW, SIDES
from run_mast3r_neural_transfer import (WEIGHTS, SCENES, intrinsic, train, install, digest, capture,
    views_for, training_features, CalibratedPriors, optimizer_hook, write_arrays, read, dump, sha)


def run(scene, side, weights):
    label = f"{scene}_{side}"
    path, metadata = RAW/label, PUBLIC/label
    packet = PUBLIC/f"{side}.npz"
    packet_hash = read(PUBLIC/"inputs.json")["packets"][side]
    if sha(packet) != packet_hash or sha(weights/"model.safetensors") != WEIGHTS:
        raise ValueError("Frozen inputs changed")
    if path.exists() or metadata.exists():
        raise FileExistsError(f"Keep the previous attempt at {label}")
    path.mkdir(parents=True)
    metadata.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AsymmetricMASt3R.from_pretrained(str(weights), local_files_only=True).eval().cuda().requires_grad_(False)
    original = list(model.state_dict().items())
    before = digest(original)
    torch.manual_seed(42)
    np.random.seed(42)
    install(model, ROOT/"third_party/TCO/peft/lora.py")
    _, names, views = views_for(scene+"_base")
    with np.load(packet) as data:
        samples = [data[f"{scene}_{f}"] for f in FRAMES]
    with np.load(ROOT/"results/mast3r_neural_transfer/pose_packets.npz") as poses:
        cameras = poses[scene+"_clean"]
    record, support = train(model, training_features(model, views), samples, cameras, "D", 40)
    write_arrays(path/"adapter.npz", {n:p.detach().cpu().numpy() for n,p in model.named_parameters() if p.requires_grad})
    write_arrays(path/"support.npz", support)
    dump(metadata/"training.json", record)
    def factory(args):
        return CalibratedPriors(alignment, args, samples, cameras, intrinsic(), .01)
    with optimizer_hook(alignment, factory) as callbacks:
        capture(model, scene+"_base", path/"cache", path, metadata, "alignment")
    trace = callbacks[0].trace
    if len(trace) != 300 or not all(r.get("finite_gradient") and r.get("received_gradient") == 1. for r in trace):
        raise ValueError("Incomplete calibrated alignment")
    if digest(original) != before or validate_pairs(path/"cache", names) != 20:
        raise ValueError("Original weights or matching inventory changed")
    dump(path/"alignment_trace.json", trace)
    files = {f.relative_to(path).as_posix():sha(f) for f in sorted(path.rglob("*")) if f.is_file()}
    dump(metadata/"trained.json", dict(scene=scene, side=side, packet_sha256=packet_hash, checkpoint=WEIGHTS,
        base_state=before, tf32=True, files=files, training_sha256=sha(metadata/"training.json"),
        alignment_sha256=sha(metadata/"alignment.json"), view_ids=[hashlib.md5(n.encode()).hexdigest() for n in names],
        metric_scale=trace[0]["metric_scale"], trace_rows=len(trace)))
    print("Completed source-only training and native alignment", label, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("side", choices=SIDES)
    parser.add_argument("--weights", type=Path, default=Path("/weights"))
    arguments = parser.parse_args()
    run(arguments.scene, arguments.side, arguments.weights)
