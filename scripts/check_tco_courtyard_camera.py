"""Qualify native-depth TCO camera checkpointing from completed case artifacts."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from check_tco_full40_recompute import compare_arrays
from run_profiled_d0 import read, sha, write_new


NATIVE_VIEWS = [[5]]
FIXED_OUTPUT_TOLERANCE = 1e-5
FIXED_FACTOR_TOLERANCE = 1e-4
REFINEMENTS = (
    "5views_4iterations_last", "5views_4iterations_all",
    "10views_4iterations_last", "10views_4iterations_all",
    "5views_2iterations_all",
)


def _receipt(path):
    path = Path(path)
    if path.is_dir():
        path /= "complete.json"
    value = read(path)
    if value["rental_compute_usd"] != 0:
        raise ValueError(f"Case {value['id']} exceeded the rental budget")
    return path, value


def _verify_fixture(path):
    path, row = _receipt(path)
    fixture = row["fixture"]
    if (fixture["device"] != "cpu" or fixture["torch"].split("+")[0] != "2.7.1"
            or set(fixture["cases"]) != set(REFINEMENTS)):
        raise ValueError("Expected the pinned CPU camera fixture with all cotangent cases")
    for name in REFINEMENTS:
        case = fixture["cases"][name]
        expected_id = f"{case['views']}views_{case['iterations']}iterations_{case['adjoints']}"
        if (name != expected_id or len(case["outputs"]) != case["iterations"]
                or case["replay"] != ["camera.recompute"]):
            raise ValueError(f"Invalid camera fixture case {name}")
        for difference in (*case["outputs"], case["last_tap_gradient"]):
            if any(not math.isfinite(difference[key]) or difference[key] < 0
                   for key in ("max_abs", "relative_l2")):
                raise ValueError(f"Invalid parity differences in camera fixture case {name}")
    return path, fixture


def _check_pair(native, candidate, *, reduced, leak=False):
    expected_raster = [5, 3, 224, 294] if reduced else [5, 3, 392, 518]
    expected_updates = 2 if reduced else 0
    expected_factors = 384 if reduced else 0
    for row in (native, candidate):
        if (row["native_depth_batches"] != NATIVE_VIEWS or row["raster_shape"] != expected_raster
                or row["head_taps"] != [4, 11, 17, 23] or row["updates"] != expected_updates
                or row["lora_tensors"] != expected_factors
                or len(row["losses"]) != expected_updates or not all(map(math.isfinite, row["losses"]))):
            raise ValueError(f"Case {row['id']} did not use the registered full decoder and native depth batches")
        audit = row["audit"]
        if audit["frozen_start_sha256"] != audit["frozen_end_sha256"]:
            raise ValueError(f"Frozen weights changed in {row['id']}")
    first_flag = True if leak else False
    if native["camera_checkpoint"] is not first_flag or candidate["camera_checkpoint"] is not True:
        raise ValueError("Unexpected camera checkpoint assignment for the qualification pair")

    paired_fields = ("dataset", "scene", "frames", "input_sha256", "seed", "steps", "runtime")
    if any(native[field] != candidate[field] for field in paired_fields):
        raise ValueError("Mismatched inputs or runtime across a camera qualification pair")
    equal_reference = native["reference_sha256"] == candidate["reference_sha256"]
    if equal_reference != (not leak):
        raise ValueError("Reference digest failed to match the registered qualification intervention")
    fields = ("frozen_start_sha256", "frozen_end_sha256", "initial_prediction", "taps")
    if reduced:
        fields += ("adapter_initial", "render_orders", "update_predictions", "pose_encoding_hashes")
    for name in fields:
        if native["audit"][name] != candidate["audit"][name]:
            raise ValueError(f"Camera qualification mismatch in {name}")


def _check_fixed_inventory(path, row):
    audit = row["audit"]
    factors = set(audit["adapter_initial"])
    if (len(factors) != 384 or any(not name.endswith((".lora_A", ".lora_B")) for name in factors)
            or factors != set(audit["adapter_final"])):
        raise ValueError(f"Incorrect adapter inventory in {row['id']}")
    if len(audit["render_orders"]) != 2 or any(sorted(order) != list(range(5))
                                               for order in audit["render_orders"]):
        raise ValueError(f"Incorrect render order in {row['id']}")
    if len(audit["pose_encoding_hashes"]) != 2:
        raise ValueError(f"Missing camera encodings from the two updates in {row['id']}")
    norms = audit["per_factor_norms"]
    digest = audit["gradients"]
    if set(norms) != {"0", "1"} or set(digest) != set(norms):
        raise ValueError(f"Missing per-factor gradients in {row['id']}")
    for step in norms:
        if (set(norms[step]) != factors or set(digest[step]) != factors
                or any(not math.isfinite(value) or value < 0 for value in norms[step].values())):
            raise ValueError(f"Invalid gradient inventory in {row['id']} update {step}")
    if not all(any(value > 0 for name, value in norms["1"].items() if name.endswith(suffix))
               for suffix in (".lora_A", ".lora_B")):
        raise ValueError(f"Both LoRA factors must carry gradients after the first update in {row['id']}")
    expected = {f"grad/{step}/{name}" for name in factors for step in (0, 1)}
    expected.update(f"parameter/{step}/{name}" for name in factors for step in (1, 2))
    with np.load(path.with_name("lora_audit.npz"), allow_pickle=False) as arrays:
        if set(arrays.files) != expected:
            raise ValueError(f"Fixed-adjoint arrays do not cover all gradients and updates in {row['id']}")


def qualify(fixture, native_identity, camera_identity, native_fixed, camera_fixed, leak, *, authority=None):
    """Return compact parity, fixture, and evaluation-leak evidence."""
    fixture_path, fixture_data = _verify_fixture(fixture)
    native_id_path, native_id = _receipt(native_identity)
    camera_id_path, camera_id = _receipt(camera_identity)
    native_path, native = _receipt(native_fixed)
    camera_path, camera = _receipt(camera_fixed)
    leak_path, leak_case = _receipt(leak)
    _check_pair(native_id, camera_id, reduced=False)
    _check_pair(native, camera, reduced=True)
    _check_pair(camera, leak_case, reduced=True, leak=True)
    for path, row in ((native_path, native), (camera_path, camera), (leak_path, leak_case)):
        if row["runtime"]["torch"] != fixture_data["torch"]:
            raise ValueError(f"Case {row['id']} used a different Torch build from the CPU fixture")
        _check_fixed_inventory(path, row)
    if (camera["output_sha256"] != leak_case["output_sha256"]
            or camera["audit"]["adapter_final"] != leak_case["audit"]["adapter_final"]):
        raise ValueError("Evaluation-only geometry changed the adapted camera candidate")

    array_paths = {
        "native_fixed_factors": (camera_path, native_path, "lora_audit.npz", FIXED_FACTOR_TOLERANCE),
        "native_fixed_predictions": (camera_path, native_path, "prediction.npz", FIXED_OUTPUT_TOLERANCE),
        "native_release_predictions": (camera_id_path, native_id_path, "prediction.npz", FIXED_OUTPUT_TOLERANCE),
        "leak_predictions": (leak_path, camera_path, "prediction.npz", 0),
    }
    measurements = {}
    for name, (active, baseline, filename, tolerance) in array_paths.items():
        difference = compare_arrays(active.with_name(filename), baseline.with_name(filename))
        if difference["max_relative_l2"] > tolerance:
            raise ValueError(f"Camera checkpoint parity failed for {name}: {difference}")
        measurements[name] = difference
    paths = dict(fixture=fixture_path, native_identity=native_id_path, camera_identity=camera_id_path,
                 native_fixed=native_path, camera_fixed=camera_path, leak=leak_path)
    return dict(qualified=True, fixture=fixture_data, native_depth_batches=NATIVE_VIEWS,
                render_orders=camera["audit"]["render_orders"],
                frozen_sha256=camera["audit"]["frozen_end_sha256"],
                leak_output_sha256=leak_case["output_sha256"],
                forward_hashes_identical=True,
                raw_gradient_hashes_identical=native["audit"]["gradients"] == camera["audit"]["gradients"],
                receipts={key: sha(path) for key, path in paths.items()},
                authority_sha256=sha(authority, text=True) if authority is not None else None,
                rental_compute_usd=0, **measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    for option in ("native-identity", "camera-identity", "native-fixed", "camera-fixed", "leak"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = qualify(args.fixture, args.native_identity, args.camera_identity,
                     args.native_fixed, args.camera_fixed, args.leak, authority=args.authority)
    if args.out:
        write_new(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
