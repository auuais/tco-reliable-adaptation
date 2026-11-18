"""Pinned CPU-only synthetic bridge for the new M3 leaf and legacy evaluator.

Accepts only a fresh output path; never loads study prediction/reference arrays.
Run in two fresh CPU-only containers with the accepted thread environment.
The original numerical protocol and strict exact bridge are not relaxed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

import check_profiled_m3_dense_runtime as original
import profiled_m3_dense as dense


ROOT = Path(__file__).resolve().parents[1]


def check_case(prediction, reference, mask):
    # The old synthetic probe calls the exact extracted legacy evaluate body
    # with recording-only delegates, including its original metric helpers.
    legacy = original.probe(prediction, reference, mask)
    np.random.seed(42)
    torch.manual_seed(42)
    o3d.utility.random.seed(42)
    actual = dense.evaluate(prediction.copy(), reference.copy(), mask.copy())
    mapping = dict(aligned_points="aligned_predicted", reference_points="reference_points",
                   prediction_normals="prediction_normals", reference_normals="reference_normals",
                   icp_correspondences="icp_correspondences")
    checks = dict(metrics_exact=actual["metrics"] == legacy["metrics"],
                  icp_fitness_exact=actual["diagnostics"]["icp_fitness"] == legacy["icp_fitness"],
                  icp_inlier_rmse_exact=actual["diagnostics"]["icp_inlier_rmse_m"] == legacy["icp_inlier_rmse"],
                  icp_correspondence_count_exact=actual["diagnostics"]["icp_correspondence_count"] == legacy["icp_correspondence_count"])
    for old_name, new_name in mapping.items():
        checks[old_name + "_exact"] = original.tensor_hash(actual["arrays"][new_name]) == legacy["tensor_sha256"][old_name]
        checks[old_name + "_shape"] = list(actual["arrays"][new_name].shape) == legacy["shapes"][old_name]
    if not all(checks.values()):
        raise ValueError("Leaf/legacy numerical bridge failed: " + json.dumps(checks))
    return dict(pass_all=True, checks=checks, metrics=actual["metrics"], diagnostics=actual["diagnostics"],
                tensor_sha256={name: original.tensor_hash(value) for name, value in actual["arrays"].items()},
                shapes={name: list(value.shape) for name, value in actual["arrays"].items()},
                legacy=legacy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing bridge evidence; select a fresh output")
    runtime = original.inspect_runtime()
    if any(row.get("num_threads") != 1 for row in runtime["threadpools"]):
        raise ValueError("Every observed BLAS/OpenMP pool must retain the accepted one-thread setting")
    pred, ref, mask = original.synthetic_cloud()
    outlier = pred.copy()
    outlier[1, 4, 5] += np.array([1., 0., 0.], dtype=outlier.dtype)
    holes = ref.copy()
    holes[~mask] = np.nan
    cases = dict(float32_reference=(pred, ref, mask), float64_reference=(pred, ref.astype(np.float64), mask),
                 large_outlier=(outlier, ref, mask), undefined_reference_holes=(pred, holes, mask))
    reports = {name: check_case(*values) for name, values in cases.items()}
    # The helper unit tests separately prove asymmetric NN directions and the
    # absence of a final .1m cutoff. This bridge checks identical real kernels.
    sources = ("scripts/profiled_m3_dense.py", "scripts/test_profiled_m3_dense.py",
               "scripts/check_profiled_m3_dense.py", "scripts/check_profiled_m3_dense_runtime.py",
               "scripts/run_repeat_calibration.py", "third_party/TCO/evaluation/mv_recon/utils.py")
    report = dict(schema="profiled_m3_dense_leaf_bridge_v1", synthetic_only=True, pass_all=True,
                  runtime=runtime, source_sha256_lf={name: original.sha(ROOT / name, text=True) for name in sources},
                  synthetic_input_sha256={name: {key: original.tensor_hash(value) for key, value in zip(
                      ("prediction", "reference", "mask"), values)} for name, values in cases.items()},
                  cases=reports, study_arrays_opened=False, scientific_accuracy_scoring_performed=False,
                  release_authorized=False, limitation="Synthetic kernel equivalence on this pinned runtime; not study accuracy or cross-machine equivalence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(pass_all=True, synthetic_only=True, cases=len(cases), output=str(args.output))))


if __name__ == "__main__":
    main()
