"""Prospective M3 input adapter; no GPU execution or scoring on import.

The M2 modules remain immutable. The endpoint body below is deliberately copied
without arithmetic changes; only its original-prior-row provider is replaced.
CPU input equivalence is necessary but does not replace the registered M3
centered/clean endpoint acceptance check before new scientific GPU execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from check_profiled_endpoints import dump_new, load_fresh_graph
from profiled_artifacts import resolve_logical
from profiled_final_calibration import refine_native

ROOT = Path(__file__).resolve().parents[1]
FRAMES = (25, 225, 425, 625, 825)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def verify_input_files(identities):
    require(isinstance(identities, dict) and identities, "Missing sealed input identities")
    for name, expected in sorted(identities.items()):
        require(isinstance(expected, str) and len(expected) == 64, "Invalid input SHA256")
        path = resolve_logical(ROOT, name, must_exist=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Prepared input changed: {name}")


def input_digest(case):
    """Bind the actual small fitting tensors, including camera and fold values."""
    digest = hashlib.sha256()
    for name in ("scene", "side", "grid", "pose_condition"):
        digest.update(json.dumps([name, case[name]], separators=(",", ":")).encode())
    arrays = [(name, case[name]) for name in ("K", "w2c", "c2w")]
    for i in range(5):
        arrays.extend([(f"xy_{i}", case["samples"][i]["xy"]),
                       (f"depth_{i}", case["samples"][i]["depth"]),
                       (f"fold_{i}", case["folds"][i]),
                       (f"original_{i}", case["original_samples"][i])])
    for name, value in arrays:
        array = _array(value)
        kind = "<i8" if array.dtype.kind in "iu" else "<f8"
        normalized = np.ascontiguousarray(array, dtype=kind)
        digest.update(json.dumps([name, list(normalized.shape), kind]).encode())
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def _validate_prior_rows(case):
    from prepare_profiled_inputs import native_selection

    require(len(case["samples"]) == len(case["folds"]) == len(case["original_samples"]) == 5,
            "Exactly five registered views are required")
    selected = []
    for view, frame in enumerate(FRAMES):
        values = np.array(case["original_samples"][view], dtype=np.float64, copy=True)
        xy = _array(case["samples"][view]["xy"])
        depth = _array(case["samples"][view]["depth"])
        folds = _array(case["folds"][view])
        require(values.ndim == 2 and values.shape[1] == 7 and np.isfinite(values).all(),
                "Original source rows must be finite Nx7 observations")
        require(xy.dtype.kind in "iu" and xy.shape == (len(values), 2), "Invalid native pixels")
        require(depth.shape == folds.shape == (len(values),), "Observation/fold count mismatch")
        require(np.isfinite(depth).all() and (depth > 0).all(), "Invalid measured axial depth")
        require(folds.dtype.kind in "iu" and np.isin(folds, [0, 1]).all(), "Invalid fold labels")
        require(min(int((folds == fold).sum()) for fold in (0, 1)) >= 8,
                "Each view needs eight observations in both folds")
        source = frame + (-4 if case["side"] == "earlier" else 4)
        require((values[:, 3] == source).all(), "A source row belongs to the wrong temporal direction")
        rows, actual_xy, outside, duplicates = native_selection(values)
        require(not outside and not duplicates and len(rows) == len(values),
                "Original rows changed the fixed native fitting coverage")
        order = np.argsort(actual_xy[:, 1] * 512 + actual_xy[:, 0])
        expected_order = np.argsort(xy[:, 1] * 512 + xy[:, 0])
        np.testing.assert_array_equal(actual_xy[order], xy[expected_order])
        np.testing.assert_array_equal(values[rows, 2][order], depth[expected_order])
        selected.append(values)
    return selected


def from_prepared_case(payload, device="cpu"):
    """Convert an already verified, selected CPU packet into the M2 case schema."""
    require(payload["scene"] in ("heads", "chess", "fire"), "Unknown scene")
    require(payload["side"] in ("earlier", "later"), "Unknown temporal source")
    require(payload["grid"] in ("centered", "shifted"), "Unknown sensor grid")
    require(payload["pose_condition"] in ("clean", "noisy"), "Unknown pose condition")
    verify_input_files(payload["input_sha256"])
    case = {key: payload[key] for key in ("scene", "side", "grid", "pose_condition",
                                        "packet_sha256", "splits_sha256")}
    case["input_sha256"] = dict(payload["input_sha256"])
    case["input_semantic_sha256"] = dict(payload.get("input_semantic_sha256", {}))
    for name, shape in (("K", (3, 3)), ("w2c", (5, 4, 4)), ("c2w", (5, 4, 4))):
        array = np.asarray(payload[name], dtype=np.float64)
        require(array.shape == shape and np.isfinite(array).all(), f"Invalid {name}")
        case[name] = torch.tensor(array.copy(), dtype=torch.float64, device=device)
    require(np.allclose(_array(case["w2c"]) @ _array(case["c2w"]), np.eye(4), rtol=0, atol=1e-10),
            "Supplied world/camera transforms disagree")
    intrinsic = _array(case["K"])
    require(intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0 and np.linalg.det(intrinsic) > 0
            and np.array_equal(intrinsic[2], [0., 0., 1.]), "Invalid normalized camera intrinsic")
    for name in ("w2c", "c2w"):
        cameras = _array(case[name])
        require(np.allclose(cameras[:, 3, :], [0., 0., 0., 1.], rtol=0, atol=1e-12)
                and (np.linalg.det(cameras[:, :3, :3]) > 0).all(),
                "Camera transforms must be affine, invertible and right handed")
    require(all(np.asarray(item["xy"]).dtype.kind in "iu" for item in payload["samples"]),
            "Native pixel coordinates must already have an integer dtype")
    require(all(np.asarray(item).dtype.kind in "iu" for item in payload["folds"]),
            "Fold labels must already have an integer dtype")
    case["samples"] = [dict(xy=torch.tensor(np.asarray(item["xy"]).copy(), dtype=torch.long, device=device),
                            depth=torch.tensor(np.asarray(item["depth"]).copy(), dtype=torch.float64, device=device))
                       for item in payload["samples"]]
    case["folds"] = [torch.tensor(np.asarray(item).copy(), dtype=torch.long, device=device)
                     for item in payload["folds"]]
    case["original_samples"] = [np.array(item, dtype=np.float64, copy=True) for item in payload["original_samples"]]
    _validate_prior_rows(case)
    case["prepared_input_digest"] = input_digest(case)
    return case


def load_case(plan_path, attempt, scene, side, grid, pose_condition, device="cuda"):
    from prepare_profiled_m3_inputs import load_prepared_case

    payload = load_prepared_case(plan_path, attempt, scene, side, grid, pose_condition)
    return from_prepared_case(payload, device=device)


def native_prior_samples(case):
    """Preserve the selected original rows and detect input mutation before KDP."""
    verify_input_files(case["input_sha256"])
    require(input_digest(case) == case["prepared_input_digest"], "Fitting tensors changed after preparation")
    return _validate_prior_rows(case)


def capture_endpoint(model, case, cache, raw, public):
    """Capture native KDP plus final refinement for one adapter state."""
    import mast3r.cloud_opt.sparse_ga as alignment
    from mast3r_alignment_hook import optimizer_hook
    from mast3r_calibrated_alignment import CalibratedPriors
    from mast3r_network_capture import capture, views_for
    from prepare_dense_focal_pilot import write_arrays
    from prepare_profiled_inputs import sha

    if cache.exists() or raw.exists() or public.exists():
        raise FileExistsError("Each endpoint needs unused cache, raw and public directories")
    original_samples = native_prior_samples(case)
    poses = case["w2c"].detach().cpu().numpy()
    intrinsic = case["K"].detach().cpu().numpy()
    if intrinsic.shape != (3, 3):
        raise ValueError("Registered KDP bridge expects the shared supplied 3x3 intrinsic")

    def factory(arguments):
        return CalibratedPriors(alignment, arguments, original_samples, poses, intrinsic, .01)

    with optimizer_hook(alignment, factory) as callbacks:
        metadata = capture(model, case["scene"] + "_base", cache, raw, public, "native")
    if metadata["prior_cache_files"] != 0 or metadata["coarse_iterations"] != 300 or metadata["fine_iterations"] != 300:
        raise ValueError("Endpoint did not use the registered fresh 300/300 native alignment")
    if len(callbacks) != 1 or len(callbacks[0].trace) != 300:
        raise ValueError("Native KDP prior hook did not execute all 300 fine iterations")
    if not all(event.get("finite_gradient") and event.get("received_gradient") == 1. for event in callbacks[0].trace):
        raise ValueError("Native KDP priors did not receive finite unit loss gradients")
    counts = [len(sample["xy"]) for sample in case["samples"]]
    if callbacks[0].counts != counts:
        raise ValueError("Native KDP fitting sample counts differ from the sealed selection")
    dump_new(public / "native_trace.json", callbacks[0].trace)
    _, names, _ = views_for(case["scene"] + "_base")
    matches, confidence, graph = load_fresh_graph(cache, names)
    dump_new(public / "native_graph.json", graph)
    with np.load(raw / "native.npz", allow_pickle=False) as archive:
        native = {name: archive[name].copy() for name in archive.files}
    if set(native) != {"depth", "intrinsics", "c2w"}:
        raise ValueError("Unexpected native output fields")
    # The tiny final solve is deliberately CPU float64 in this endpoint path.
    final_depth, metric_cameras, scales, diagnostics = refine_native(
        torch.from_numpy(native["depth"]), torch.from_numpy(native["intrinsics"]),
        torch.from_numpy(native["c2w"]), case["samples"], matches, confidence)
    final = dict(depth=final_depth.numpy(), intrinsics=native["intrinsics"].copy(),
                 c2w=metric_cameras.numpy(), scales=scales.numpy(),
                 g0=np.asarray(diagnostics["g0"], dtype=np.float64))
    write_arrays(raw / "final.npz", final)
    dump_new(public / "final.json", diagnostics)
    evidence = dict(capture=metadata, final=diagnostics, sample_counts=counts,
                    original_prior_rows_preserved=True, final_compute="CPU float64",
                    native_file=(raw / "native.npz").relative_to(ROOT).as_posix(),
                    native_sha256=sha(raw / "native.npz"),
                    final_file=(raw / "final.npz").relative_to(ROOT).as_posix(),
                    final_sha256=sha(raw / "final.npz"),
                    graph_file=(public / "native_graph.json").relative_to(ROOT).as_posix(),
                    graph_sha256=sha(public / "native_graph.json"))
    return native, final, matches, confidence, evidence
