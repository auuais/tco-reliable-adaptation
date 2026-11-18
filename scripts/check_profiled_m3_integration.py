"""Pinned CPU-only synthetic endpoint -> surface -> diagnostic integration.

Uses production raster sizes and generated arrays only. A compact fixed mask
keeps the ICP example economical; this is not a full-mask timing benchmark.
There is no study archive input, registration or score-release capability.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

import check_profiled_m3_dense_runtime as runtime
import profiled_m3_dense as dense
import profiled_m3_diagnostics as diagnostics
import profiled_m3_points as points
from test_profiled_m3_points import fixtures


ROOT = Path(__file__).resolve().parents[1]


def check():
    native, final = fixtures()
    converted = points.convert_endpoints(native, final)
    mask = np.zeros(points.DENSE_SHAPE, dtype=bool)
    mask[:, 110:140, 210:245] = True
    mask[:, 123:126, 225:228] = False
    reference = converted["native"]["points"].copy()
    camera = np.linalg.inv(converted["native"]["c2w"])
    validity = points.fixed_mask_validity(converted, mask)
    report = {}
    for endpoint in ("native", "final"):
        value = converted[endpoint]
        np.random.seed(42)
        torch.manual_seed(42)
        o3d.utility.random.seed(42)
        kernel = dense.evaluate(value["points"], reference, mask)
        primary = diagnostics.diagnose(value["depth_m"], value["points"], reference,
            mask, kernel["arrays"]["aligned_predicted"], reference_label="registered_rgb",
            alignment=kernel["metrics"], reference_w2c=camera)
        alternate = diagnostics.diagnose(value["depth_m"], value["points"], reference,
            mask, kernel["arrays"]["aligned_predicted"], reference_label="released",
            alignment=kernel["metrics"])
        assert primary["mask_tensor_sha256"] == alternate["mask_tensor_sha256"] == validity["mask_tensor_sha256"]
        assert primary["mask_count"] == alternate["mask_count"] == kernel["metrics"]["valid_points"] == int(mask.sum())
        assert all(value > 0 for value in primary["mask_counts_by_view"])
        assert not alternate["metric_axial_depth"]["supported"]
        assert primary["paired_pixel_distance"] == alternate["paired_pixel_distance"]
        # Independent corresponding-pixel axial calculation; never fitted by
        # the surface Sim3/ICP transform and never another g0 multiplication.
        axial = np.einsum("vhwc,vc->vhw", reference, camera[:,2,:3]) + camera[:,2,3,None,None]
        expected = float(np.mean(np.abs(value["depth_m"][mask] - axial[mask])))
        observed = primary["metric_axial_depth"]["pooled_pixels"]["all"]["mean_absolute_m"]
        assert abs(expected-observed) <= 1e-14
        report[endpoint] = dict(metrics=kernel["metrics"],kernel_diagnostics=kernel["diagnostics"],
            primary_diagnostics=primary,alternate_diagnostics=alternate,
            geometry_sha256={name:runtime.tensor_hash(array) for name,array in value.items()},
            evaluated_array_sha256={name:runtime.tensor_hash(array) for name,array in kernel["arrays"].items()},
            direct_axial_mae_m=expected)
    # A scale error must remain visible in raw metric depth even after the
    # evaluator fits a similarity. Native rounding is present but much smaller.
    assert report["final"]["direct_axial_mae_m"] > .05
    assert report["native"]["direct_axial_mae_m"] < 1e-5
    bad_native,bad_final = deepcopy(native),deepcopy(final)
    bad_native["depth"][0,128,224] = -100.
    bad_final["depth"][0,128,224] = float(final["g0"])*float(final["scales"][0])*(-100.)
    bad = points.convert_endpoints(bad_native,bad_final)
    bad_validity = points.fixed_mask_validity(bad,mask)
    assert bad_validity["main_nonpositive_depth"] > 0
    assert bad_validity["mask_tensor_sha256"] == validity["mask_tensor_sha256"]
    assert bad_validity["mask_count"] == bad_validity["evaluated_mask_count"] == int(mask.sum())
    return dict(shape=list(points.DENSE_SHAPE),mask_sha256=validity["mask_tensor_sha256"],
        fixed_mask_points=int(mask.sum()),validity=validity,endpoints=report,
        negative_depth_case=bad_validity,case="synthetic_nonuniform_relative_scales_with_shared_metric_centers",
        compact_mask_for_engineering_only=True,full_mask_performance_claim=False,
        source_endpoint_sha256={kind:{key:runtime.tensor_hash(value) for key,value in packet.items()}
                              for kind,packet in (("native",native),("final",final))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing integration evidence; choose a fresh output")
    environment=runtime.inspect_runtime()
    if any(row.get("num_threads")!=1 for row in environment["threadpools"]):
        raise ValueError("Every observed OpenMP/BLAS pool must use the accepted one-thread setting")
    result=check()
    sources=("scripts/check_profiled_m3_integration.py","scripts/check_profiled_m3_dense_runtime.py",
             "scripts/profiled_m3_dense.py","scripts/profiled_m3_diagnostics.py","scripts/profiled_m3_points.py",
             "scripts/test_profiled_m3_points.py","scripts/prepare_profiled_m3_references.py")
    report=dict(schema="profiled_m3_synthetic_dense_integration_v1",pass_all=True,synthetic_only=True,
        source_sha256_lf={name:runtime.sha(ROOT/name,text=True) for name in sources},runtime=environment,
        result=result,study_arrays_opened=False,scientific_accuracy_scoring_performed=False,release_authorized=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(pass_all=True,synthetic_only=True,mask_points=result["fixed_mask_points"],output=str(args.output))))


if __name__=="__main__":
    main()
