"""Run the frozen 48-case MASt3R geometry/matching crossover without model inference."""
import argparse
from contextlib import contextmanager
import itertools
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch
import mast3r.cloud_opt.sparse_ga as alignment

from evaluation_stability import dump
from mast3r_alignment_hook import optimizer_hook
from mast3r_calibrated_alignment import CalibratedPriors
from mast3r_crossover_cache import materialize, validate_pairs
from mast3r_network_capture import capture, views_for
from plan_mast3r_crossover import ROOT, PARENT, MANIFEST, cache_inventory, expected_plan, read, require, sha256, text_sha256, validate_public


PUBLIC = ROOT / "results/mast3r_geometry_matching_v1"
RAW = ROOT / "reports/mast3r_geometry_matching_v1"
SOURCES = ("scripts/run_mast3r_crossover.py", "scripts/mast3r_crossover_cache.py",
           "scripts/mast3r_network_capture.py", "scripts/mast3r_alignment_hook.py",
           "scripts/mast3r_calibrated_alignment.py", "scripts/mast3r_joint_prior_loss.py")


def compare_endpoint(path, endpoint, tolerance):
    result = {}
    with np.load(path) as actual, np.load(PARENT / "native_outputs" / (endpoint["tag"] + ".npz")) as old:
        for field in ("depth", "intrinsics", "c2w"):
            a, b = actual[field], old[field]
            require(a.shape == b.shape and a.dtype == b.dtype, f"Changed endpoint schema {field}")
            require(np.isfinite(a).all(), f"Nonfinite endpoint {field}")
            result[field] = dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.max(np.abs(a - b))),
                                 close=bool(np.allclose(a, b, **tolerance)), shape=list(a.shape), dtype=str(a.dtype))
        a, b = actual["c2w"].astype(float), old["c2w"].astype(float)
        relative_a, relative_b = np.linalg.solve(a[0], a), np.linalg.solve(b[0], b)
        result["relative_c2w"] = dict(close=bool(np.allclose(relative_a, relative_b, **tolerance)),
                                      max_abs=float(np.max(np.abs(relative_a - relative_b))))
        result["nonpositive_counts_match"] = bool(np.array_equal((actual["depth"] <= 0).sum((1, 2)), (old["depth"] <= 0).sum((1, 2))))
    result["pass"] = all(result[f]["close"] for f in ("depth", "intrinsics", "relative_c2w")) and result["nonpositive_counts_match"]
    return result


@contextmanager
def checked_forward(names, evidence):
    original = alignment.forward_mast3r
    expected = set(itertools.permutations(names, 2))

    def forward(*args, **kwargs):
        paths, cache = original(*args, **kwargs)
        require(set(paths) == expected, "Alignment did not consume all ordered pairs")
        evidence["consumed_pairs"] = [list(pair) for pair in paths]
        return paths, cache

    alignment.forward_mast3r = forward
    try:
        yield
    finally:
        alignment.forward_mast3r = original


def run(selected):
    # CroCo's model import enables TF32 in the parent neural runner.
    torch.backends.cuda.matmul.allow_tf32 = True
    plan = read(MANIFEST)
    require(plan == expected_plan(), "Frozen crossover plan changed")
    inventory = {key: {r["file"]: r["sha256"] for r in rows} for key, rows in validate_public(plan).items()}
    fingerprints = {name: text_sha256(ROOT / name) for name in SOURCES}
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    session = dict(manifest_sha256=sha256(MANIFEST), sources=fingerprints,
                   python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(), execution_commit=commit,
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if (PUBLIC / "session.json").exists():
        existing = read(PUBLIC / "session.json")
        require(all(existing[key] == session[key] for key in session if key != "execution_commit"),
                "Crossover runtime or sources changed")
    else:
        dump(PUBLIC / "session.json", session)
    jobs = [r for r in plan["runs"] if not selected or r["id"] in selected]
    require(not selected or len(jobs) == len(selected), "Unknown or duplicate requested run")
    for job in jobs:
        tag = job["id"]
        folder = PUBLIC / "cases" / tag
        raw = RAW / "cases" / tag
        completion = folder / "complete.json"
        if completion.exists():
            done = read(completion)
            require(done["job"] == job and done["sources"] == fingerprints and done["manifest_sha256"] == sha256(MANIFEST), "Completion provenance differs")
            for relative, digest in done["files"].items():
                require(sha256(ROOT / relative) == digest, f"Changed completed file {relative}")
            print(tag, "already verified", flush=True)
            continue
        require(not folder.exists() and not raw.exists(), f"Incomplete attempt needs engineering review: {tag}")
        if "endpoint" not in job:
            for control in (r for r in plan["runs"] if "endpoint" in r):
                proof = read(PUBLIC / "cases" / control["id"] / "bridge.json")
                require(proof["pass"], "An endpoint bridge is unresolved")
        folder.mkdir(parents=True)
        raw.mkdir(parents=True)
        start = time.perf_counter()
        try:
            _, names, _ = views_for(job["scene"] + "_base")
            cache = raw / "cache"
            operands = materialize(RAW.parent / "mast3r_neural_transfer/cache" / job["coordinates_from"],
                                   RAW.parent / "mast3r_neural_transfer/cache" / job["confidences_and_matches_from"],
                                   cache, names, inventory)
            dump(folder / "operands.json", dict(geometry=job["coordinates_from"], bundle=job["confidences_and_matches_from"], files=operands))
            packet = "neighbor_sparse_depth" if job["grid"] == "centered" else "shifted_sparse_grid"
            with np.load(ROOT / "results" / packet / "measurements.npz") as data:
                samples = [data[f'{job["scene"]}_{frame}'] for frame in plan["frames"]]
            with np.load(PARENT / "pose_packets.npz") as data:
                poses = data[job["pose_packet"]]
            evidence = {}

            def factory(arguments):
                root, edges = arguments["mst"]
                evidence["tree"] = dict(root=int(root), edges=[[int(a), int(b)] for a, b in edges])
                return CalibratedPriors(alignment, arguments, samples, poses, plan["known_K"], .01)

            with checked_forward(names, evidence), optimizer_hook(alignment, factory) as callbacks:
                capture(None, job["scene"] + "_base", cache, raw, folder, "native")
            require(len(callbacks) == 1 and len(callbacks[0].trace) == 300, "Incomplete fine-prior trace")
            require(all(r.get("finite_gradient") and r.get("received_gradient") == 1. for r in callbacks[0].trace), "Missing prior gradients")
            require(validate_pairs(cache, names) == 20, "Cache changed pair inventory")
            dump(folder / "trace.json", callbacks[0].trace)
            dump(folder / "alignment.json", dict(**evidence, total_seconds=time.perf_counter() - start,
                                                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20))
            if "endpoint" in job:
                bridge = compare_endpoint(raw / "native.npz", job["endpoint"], plan["bridge_tolerance"])
                dump(folder / "bridge.json", bridge)
                require(bridge["pass"], f"Endpoint bridge failed: {tag}")
            files = list(folder.glob("*.json")) + [raw / "native.npz"]
            dump(completion, dict(job=job, execution_commit=commit, sources=fingerprints,
                                  manifest_sha256=sha256(MANIFEST),
                                  files={p.relative_to(ROOT).as_posix(): sha256(p) for p in files}))
            print(tag, "completed", flush=True)
        except Exception as error:
            dump(folder / "failure.json", dict(job=job, error_type=type(error).__name__, message=str(error),
                                               traceback=traceback.format_exc(), sources=fingerprints, execution_commit=commit))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="One manifest run ID. Omit for the registered batch.")
    run(parser.parse_args().run)
