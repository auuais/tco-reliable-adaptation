"""Frozen CPU reevaluation of immutable v4/v5 prediction evidence, in fresh processes."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = "evaluation_stability_v6"
MANIFEST = ROOT / "manifests" / (STUDY + ".json")
CASES = [
    ("v5", "prior5_heads_seq-01_visual_seed123"),
    ("v5", "prior5_fire_seq-03_visual_seed42"),
    ("v5", "prior5_fire_seq-03_both_seed42"),
    ("v4", "base4_fire_seq-03_visual_seed42"),
]
METRICS = ["accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean"]
SOURCES = ["scripts/evaluation_stability.py", "scripts/run_prior_ablation.py", "scripts/run_experiment.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Evidence hash mismatch: {path.name}")


def function_ast(path):
    tree = ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate"))


def freeze(roots):
    assert not MANIFEST.exists(), "Never replace a frozen manifest"
    assert function_ast(ROOT / SOURCES[1]) == function_ast(ROOT / SOURCES[2]), "Evaluator changed"
    cases = []
    for origin, run_id in CASES:
        directory = roots[origin] / "runs" / run_id
        record = json.loads((directory / "run.json").read_text())
        inputs = record["evaluation_inputs_relative"].replace("\\", "/")
        cases.append(dict(origin=origin, run_id=run_id, seed=record["arguments"]["seed"],
                          prediction_sha256=sha(directory / "predictions.npy"),
                          inputs_relative=inputs, inputs_file_sha256=sha(roots[origin] / inputs),
                          evaluation_tensor_sha256=record["evaluation_tensor_sha256"],
                          original_metrics=json.loads((directory / "metrics.json").read_text()),
                          original_research_commit=record["research_commit"]))
    dump(MANIFEST, dict(study_id=STUDY, cases=cases, threads=[1, 4], repeats=3,
                       scope="Purposively selected known failure/contrast cases; evaluator diagnostic only",
                       cpu_only=True, fresh_process_per_evaluation=True, original_metrics_immutable=True,
                       stability_thresholds=dict(distance_relative_range=1e-6, normal_absolute_range=1e-6),
                       thresholds_are_diagnostics_not_execution_failure=True,
                       sources={name: sha(ROOT / name) for name in SOURCES}))


def worker(case, root, threads, output):
    # Thread environment is set by parent BEFORE importing numerical libraries.
    import random
    import numpy as np
    import torch
    import open3d as o3d
    from run_prior_ablation import evaluate
    prediction_path = root / "runs" / case["run_id"] / "predictions.npy"
    inputs_path = root / case["inputs_relative"]
    verify(prediction_path, case["prediction_sha256"])
    verify(inputs_path, case["inputs_file_sha256"])
    arrays = dict(np.load(inputs_path, allow_pickle=False))
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(key.encode()); digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode()); digest.update(np.ascontiguousarray(value).tobytes())
    assert digest.hexdigest() == case["evaluation_tensor_sha256"]
    seed = case["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); o3d.utility.random.seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metrics = evaluate(np.load(prediction_path, allow_pickle=False),
                       dict(images=torch.from_numpy(arrays["images"]),
                            pointclouds=arrays["reference_points"], valid_mask=arrays["valid_mask"]), output)
    verify(prediction_path, case["prediction_sha256"])
    verify(inputs_path, case["inputs_file_sha256"])
    from threadpoolctl import threadpool_info
    dump(output / "execution.json", dict(status="complete", seconds=time.perf_counter()-started,
         source_run_id=case["run_id"], threads=threads, seed=seed, python=sys.version,
         numpy=np.__version__, torch=torch.__version__, open3d=o3d.__version__,
         platform=platform.platform(), cpu=platform.processor(), threadpools=threadpool_info(),
         prediction_sha256=case["prediction_sha256"], inputs_file_sha256=case["inputs_file_sha256"],
         research_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         manifest_sha256=sha(MANIFEST), metrics=metrics))


def aggregate(manifest, directory):
    all_rows, groups = [], []
    for case in manifest["cases"]:
        rows = []
        for threads in manifest["threads"]:
            for repeat in range(manifest["repeats"]):
                key = f"{case['run_id']}_t{threads}_r{repeat}"
                row = json.loads((directory / key / "execution.json").read_text())
                assert row["status"] == "complete" and row["manifest_sha256"] == sha(MANIFEST)
                rows.append(dict(run_id=key, **row))
        comparisons = {}
        for name in METRICS:
            values = [r["metrics"][name] for r in rows]
            lo, hi = min(values), max(values)
            original = case["original_metrics"][name]
            distance = name != "normal_consistency_mean"
            spread = (hi-lo)/max(abs(original), 1e-30) if distance else hi-lo
            threshold = manifest["stability_thresholds"]["distance_relative_range" if distance else "normal_absolute_range"]
            comparisons[name] = dict(min=lo, max=hi, original=original, range_relative_if_distance=spread,
                exceeds_threshold=spread > threshold, max_absolute_difference_from_original=max(abs(v-original) for v in values))
        groups.append(dict(source_run_id=case["run_id"], metrics=comparisons))
        all_rows.extend(rows)
    dump(ROOT / "results" / STUDY / "summary.json", dict(study_id=STUDY, manifest_sha256=sha(MANIFEST),
         completed_evaluations=len(all_rows), all_rows=all_rows, groups=groups,
         conclusion="Evaluator stability is assessed only for these four saved predictions and this local CPU/runtime."))
    print(json.dumps(groups, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", type=Path)
    parser.add_argument("--v5-root", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    roots = dict(v4=args.v4_root, v5=args.v5_root)
    if args.freeze:
        freeze(roots); return
    manifest = json.loads(MANIFEST.read_text())
    for name, digest in manifest["sources"].items():
        verify(ROOT / name, digest)
    assert function_ast(ROOT / SOURCES[1]) == function_ast(ROOT / SOURCES[2])
    if args.dry_run:
        print("Frozen CPU evaluations:", len(manifest["cases"])*len(manifest["threads"])*manifest["repeats"]); return
    directory = ROOT / "reports" / STUDY
    if args.aggregate_only:
        aggregate(manifest, directory); return
    if args.worker is not None:
        case = manifest["cases"][args.worker]
        worker(case, roots[case["origin"]], args.threads, args.output); return
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit first"
    directory.mkdir(parents=True, exist_ok=False)
    for index, case in enumerate(manifest["cases"]):
        for threads in manifest["threads"]:
            for repeat in range(manifest["repeats"]):
                key = f"{case['run_id']}_t{threads}_r{repeat}"
                env = dict(os.environ, **{name: str(threads) for name in
                    ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
                env["PYTHONHASHSEED"] = str(case["seed"])
                cmd = [sys.executable, "scripts/evaluation_stability.py", "--worker", str(index),
                       "--threads", str(threads), "--output", str(directory / key),
                       "--v4-root", str(args.v4_root), "--v5-root", str(args.v5_root)]
                print("Evaluating", key, flush=True)
                with (directory / (key+".log")).open("x") as log:
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    aggregate(manifest, directory)


if __name__ == "__main__":
    main()
