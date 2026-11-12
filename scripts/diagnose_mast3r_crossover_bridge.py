"""Compare recomputed canonical inputs with archived inputs for the first bridge."""
import argparse
from pathlib import Path
import numpy as np
import torch

from evaluation_stability import dump
from plan_mast3r_crossover import ROOT, MANIFEST, read, require, sha256


def direct(name, deterministic):
    import mast3r.cloud_opt.sparse_ga as alignment
    from mast3r_alignment_hook import optimizer_hook
    from mast3r_calibrated_alignment import CalibratedPriors
    from mast3r_network_capture import capture
    from run_mast3r_crossover import compare_endpoint
    plan = read(MANIFEST)
    job = next(r for r in plan["runs"] if r["id"] == "heads_centered_clean_G0B0")
    cache = ROOT / "reports/mast3r_neural_transfer/cache/heads_zero"
    before = {p.relative_to(cache).as_posix(): sha256(p) for p in cache.rglob("*") if p.is_file()}
    folder = ROOT / "reports/mast3r_geometry_matching_v1/diagnosis" / name
    public = ROOT / "results/mast3r_geometry_matching_v1/diagnosis" / name
    require(not folder.exists() and not public.exists(), "Diagnostic attempt already exists")
    folder.mkdir(parents=True)
    public.mkdir(parents=True)
    with np.load(ROOT / "results/neighbor_sparse_depth/measurements.npz") as data:
        samples = [data[f'heads_{frame}'] for frame in plan["frames"]]
    with np.load(ROOT / "results/mast3r_neural_transfer/pose_packets.npz") as data:
        poses = data["heads_clean"]
    torch.use_deterministic_algorithms(deterministic)
    def factory(arguments):
        return CalibratedPriors(alignment, arguments, samples, poses, plan["known_K"], .01)
    with optimizer_hook(alignment, factory) as callbacks:
        capture(None, "heads_base", cache, folder, public, "native")
    dump(public / "trace.json", callbacks[0].trace)
    bridge = compare_endpoint(folder / "native.npz", job["endpoint"], plan["bridge_tolerance"])
    dump(public / "bridge.json", bridge)
    after = {p.relative_to(cache).as_posix(): sha256(p) for p in cache.rglob("*") if p.is_file()}
    require(before == after, "Original cache was modified")
    dump(public / "inputs.json", dict(cache=cache.relative_to(ROOT).as_posix(), files=before,
                                     deterministic=deterministic))
    print(bridge)


def main():
    source = ROOT / "reports/mast3r_neural_transfer/cache/heads_zero/canon_views"
    target = ROOT / "reports/mast3r_geometry_matching_v1/attempts/tf32-disabled01/cases/heads_centered_clean_G0B0/cache/canon_views"
    rows = []
    for path in sorted(target.glob("*.pth")):
        old = torch.load(source / path.name, map_location="cpu", weights_only=True)
        new = torch.load(path, map_location="cpu", weights_only=True)
        a, b = list(old[0]) + [old[1]], list(new[0]) + [new[1]]
        rows.append(dict(file=path.name, fields={name: dict(exact=bool(torch.equal(x, y)),
                          max_abs=float((x-y).abs().max()), dtype=str(x.dtype), shape=list(x.shape))
                          for name, x, y in zip(("canon", "ratio", "confidence", "focal"), a, b)}))
    require(len(rows) == 5, "Missing canonical views")
    old = read(ROOT / "results/mast3r_neural_transfer/cases/heads_zero_centered/KDP_clean_trace.json")
    new = read(ROOT / "results/mast3r_geometry_matching_v1/cases/heads_centered_clean_G0B0/trace.json")
    dump(ROOT / "results/mast3r_geometry_matching_v1/bridge_diagnosis.json",
         dict(canonical_views=rows, old_first_fine=old[0], new_first_fine=new[0],
              backend=dict(matmul_precision=torch.get_float32_matmul_precision(),
                           allow_tf32=torch.backends.cuda.matmul.allow_tf32)))
    print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    direct(args.direct, args.deterministic) if args.direct else main()
