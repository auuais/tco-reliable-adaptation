"""Rescore sealed point maps with the same released reference and evaluator as v6."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluation_stability import dump, sha


def read(path):
    return json.loads(Path(path).read_text())


def store_map(raw, points, scene, source_run, entries):
    import hashlib
    tensor_id = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    key = scene.replace("/", "_") + "_" + tensor_id
    if key not in entries:
        folder = raw / key
        folder.mkdir()
        path = folder / "points.npy"
        np.save(path, points, allow_pickle=False)
        entries[key] = dict(scene=scene, source_run=source_run, sha256=sha(path), key=key)
    return key


def evaluate_one(entry, raw, evidence):
    import open3d as o3d
    import torch
    from repeatability_capture import array_digest
    from run_repeat_calibration import evaluate

    record = read(evidence / "runs" / entry["source_run"] / "run.json")
    folder = raw / entry["key"]
    if sha(folder / "points.npy") != entry["sha256"] or record["arguments"]["scene"] != entry["scene"]:
        raise ValueError("Prediction or scene differs from seal")
    with np.load(evidence / record["evaluation_inputs_relative"], allow_pickle=False) as source:
        arrays = dict(source)
    if array_digest(arrays) != record["evaluation_tensor_sha256"]:
        raise ValueError("Reference tensors changed")
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    o3d.utility.random.seed(42)
    data = dict(images=torch.from_numpy(arrays["images"]), pointclouds=arrays["reference_points"],
                valid_mask=arrays["valid_mask"])
    metrics = evaluate(np.load(folder / "points.npy", allow_pickle=False), data, folder)
    if not all(np.isfinite(metrics[k]) for k in ("accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean")):
        raise ValueError("Nonfinite metric")
    dump(folder / "evaluation.json", dict(key=entry["key"], sha256=entry["sha256"],
         reference_sha256=record["evaluation_tensor_sha256"], metrics=metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--entry")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    entries = read(args.seal)["entries"]
    if args.entry:
        evaluate_one(entries[args.entry], args.raw, args.evidence)
        return
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               NUMEXPR_NUM_THREADS="2", PYTHONHASHSEED="42")

    def run(key):
        folder = args.raw / key
        if (folder / "evaluation.json").is_file():
            old = read(folder / "evaluation.json")
            if old["sha256"] != entries[key]["sha256"] or sha(folder / "points.npy") != old["sha256"]:
                raise ValueError("Completed output differs from sealed map")
            return
        attempt = len(list(folder.glob("attempt_*.log"))) + 1
        with (folder / f"attempt_{attempt}.log").open("x") as log:
            subprocess.run([sys.executable, __file__, "--seal", str(args.seal), "--raw", str(args.raw),
                            "--evidence", str(args.evidence), "--entry", key],
                           stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600, check=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, _ in enumerate(pool.map(run, entries), 1):
            print(f"Evaluated {number}/{len(entries)} maps", flush=True)


if __name__ == "__main__":
    main()
