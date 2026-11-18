"""Sensor-only bridge for the single D1 inference/calibration diagnosis.

No file/model access occurs here. Prepare both folds once from the pristine B0
fusion, then assemble every state's graph-zero matrices with that same frozen
preparation. The unused graph and P1-only preparation are deliberately omitted.
Production invocation still requires the separately committed D1 registration.
"""
import math

import torch

from profiled_calibration import (
    _at, _metric, _observations, assemble_system, prepare_calibration,
    query_residual,
)


VIEWS = 5
MINIMUM_SAMPLES = 8
METADATA_FIELDS = ("support_fold", "gains", "sensor_normalizers", "ridge",
                   "support_counts", "query_counts")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _depth(value, name):
    _require(isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == VIEWS
             and value.shape[1] > 0 and value.shape[2] > 0
             and value.dtype in (torch.float32, torch.float64),
             name + " must be a nonempty float32/float64 [5,H,W] tensor")
    _require(bool(torch.isfinite(value).all()), name + " must be finite; no pixel filtering is permitted")
    return value


def _camera_inputs(case, device):
    intrinsic = torch.as_tensor(case["K"], device=device, dtype=torch.float64)
    camera = torch.as_tensor(case["c2w"], device=device, dtype=torch.float64)
    _require(tuple(intrinsic.shape) in ((3, 3), (VIEWS, 3, 3)) and
             tuple(camera.shape) == (VIEWS, 4, 4), "Expected supplied K and five camera-to-world matrices")
    _require(bool(torch.isfinite(intrinsic).all()) and bool(torch.isfinite(camera).all()),
             "Supplied calibration must be finite")
    expanded = intrinsic[None].expand(VIEWS, -1, -1) if intrinsic.ndim == 2 else intrinsic
    _require(bool((expanded[:, 0, 0] > 0).all()) and bool((expanded[:, 1, 1] > 0).all()) and
             bool((torch.linalg.det(expanded) > 0).all()) and
             torch.allclose(expanded[:, 2], expanded.new_tensor([0., 0., 1.]).expand(VIEWS, -1),
                            rtol=0., atol=1e-12), "Invalid supplied pinhole intrinsics")
    _require(torch.allclose(camera[:, 3], camera.new_tensor([0., 0., 0., 1.]).expand(VIEWS, -1),
                            rtol=0., atol=1e-10) and
             bool((torch.linalg.det(camera[:, :3, :3]) > 0).all()),
             "Supplied cameras must be affine, invertible and right handed")
    # Preserve approximately rigid supplied rotations; do not orthogonalize.
    return intrinsic, camera


def _case_packets(initial_depth, case):
    views, height, width = initial_depth.shape
    _require(isinstance(case.get("samples"), (tuple, list)) and len(case["samples"]) == VIEWS
             and isinstance(case.get("folds"), (tuple, list)) and len(case["folds"]) == VIEWS,
             "Exactly five observation packets and five fold assignments are required")
    packets = _observations(case["samples"], views, height, width, initial_depth.device,
                            2 * MINIMUM_SAMPLES)
    labels = []
    for index, (sample, value) in enumerate(zip(packets, case["folds"])):
        raw = torch.as_tensor(value, device=initial_depth.device)
        _require(raw.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                 and raw.ndim == 1 and len(raw) == len(sample["xy"]),
                 "Fold assignments must be integer vectors matching observations")
        _require(bool(((raw == 0) | (raw == 1)).all()), "Only the two original fold labels 0 and 1 are permitted")
        _require(all(int((raw == fold).sum()) >= MINIMUM_SAMPLES for fold in (0, 1)),
                 "Each view needs eight observations in both original folds")
        _require(bool((_at(initial_depth[index], sample["xy"]) > 0).all()),
                 "Initial depth must be positive at every permitted support/query observation")
        labels.append(raw.detach().to(dtype=torch.long).clone())
    return packets, labels


def _partition(packets, labels, fold):
    """Exact selection rule of profiled_network.partition, without broad imports."""
    support, query = [], []
    for sample, assignment in zip(packets, labels):
        for destination, mask in ((support, assignment == fold), (query, assignment != fold)):
            destination.append({name: value[mask].detach().clone() for name, value in sample.items()})
    return support, query


def _metadata(state, query, fold):
    return dict(support_fold=fold, gains=state["gains"].detach().cpu().tolist(),
                sensor_normalizers=state["sensor_normalizers"].detach().cpu().tolist(),
                ridge=float(state["ridge"]), support_counts=[len(p["xy"]) for p in state["support"]],
                query_counts=[len(p["xy"]) for p in query])


@torch.no_grad()
def prepare_sensor_folds(initial_depth, fused_confidence, case):
    """Return ``states``, ``queries`` and historical-teacher-compatible metadata.

All fixed tensors are cloned/detached by the original preparation and the
selection wrapper. Device follows initial_depth, preserving original CUDA
float64 preparation semantics when used by the later registered controller.
Tests use CPU only. This function performs no solve or network operation.
"""
    initial = _depth(initial_depth, "Initial fused depth")
    confidence = torch.as_tensor(fused_confidence, device=initial.device)
    _require(confidence.shape == initial.shape and confidence.dtype in (torch.float32, torch.float64)
             and bool(torch.isfinite(confidence).all()) and bool((confidence > 0).all()),
             "Expected finite positive fused confidence with the five-view depth shape")
    intrinsic, camera = _camera_inputs(case, initial.device)
    packets, labels = _case_packets(initial, case)
    states, queries, metadata = [], [], []
    for fold in (0, 1):
        support, query = _partition(packets, labels, fold)
        state = prepare_calibration(initial, intrinsic, camera, support, [], confidence,
                                    minimum_samples=MINIMUM_SAMPLES)
        _require(state["edges"] == [] and state["graph_filtering"] == [], "Sensor-only preparation acquired graph data")
        states.append(state)
        queries.append(query)
        metadata.append(_metadata(state, query, fold))
    return dict(states=states, queries=queries, metadata=metadata)


def _teacher_fields(rows, name):
    _require(isinstance(rows, (list, tuple)) and len(rows) == 2, name + " must contain exactly two folds")
    for fold, row in enumerate(rows):
        _require(isinstance(row, dict) and set(METADATA_FIELDS) <= set(row), name + " lacks required teacher fields")
        _require(type(row["support_fold"]) is int and row["support_fold"] == fold,
                 name + " fold order must be exactly 0,1")
        for key in ("support_counts", "query_counts"):
            value = row[key]
            _require(isinstance(value, list) and len(value) == VIEWS and
                     all(type(v) is int and v >= MINIMUM_SAMPLES for v in value), name + " has invalid " + key)
        for key in ("gains", "sensor_normalizers"):
            value = row[key]
            _require(isinstance(value, list) and len(value) == VIEWS and
                     all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in value),
                     name + " has invalid " + key)
        value = row["ridge"]
        _require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                 name + " ridge must be finite and positive")
    _require(rows[0]["support_counts"] == rows[1]["query_counts"] and
             rows[1]["support_counts"] == rows[0]["query_counts"], name + " complementary fold counts differ")


def verify_teacher_metadata(metadata, teacher_states, *, rtol, atol):
    """Compare sensor fields only with prospectively registered numeric limits.

Historical teacher rows also contain graph_filtering; D1 neither reconstructs
nor claims equality for that unused metadata. Counts/fold IDs compare exactly.
"""
    _require(type(rtol) in (float, int) and type(atol) in (float, int) and
             math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0,
             "Teacher comparison tolerances must be finite nonnegative numbers")
    _teacher_fields(metadata, "Reconstructed teacher")
    _teacher_fields(teacher_states, "Historical teacher")
    comparisons = []
    for current, expected in zip(metadata, teacher_states):
        for key in ("support_counts", "query_counts"):
            _require(current[key] == expected[key], "Teacher " + key + " differs")
        fields = {}
        for key in ("gains", "sensor_normalizers", "ridge"):
            left = current[key] if key != "ridge" else [current[key]]
            right = expected[key] if key != "ridge" else [expected[key]]
            differences = [abs(float(a)-float(b)) for a, b in zip(left, right)]
            limits = [atol+rtol*abs(float(b)) for b in right]
            _require(all(error <= limit for error, limit in zip(differences, limits)),
                     "Teacher " + key + " exceeds the registered tolerance")
            fields[key] = dict(max_absolute_difference=max(differences), limits=limits)
        comparisons.append(dict(support_fold=current["support_fold"], fields=fields,
                                exact_support_query_counts=True))
    return dict(pass_all=True, rtol=rtol, atol=atol, folds=comparisons,
                graph_metadata_reconstructed=False, graph_metadata_compared=False)


def _validate_state(state, depths):
    _require(isinstance(state, dict) and tuple(state.get("shape", ())) == tuple(depths.shape),
             "Frozen preparation shape differs from current depth")
    _require(state.get("minimum_samples") == MINIMUM_SAMPLES, "Frozen minimum sample rule differs")
    for key in ("gains", "sensor_normalizers"):
        value = state.get(key)
        _require(isinstance(value, torch.Tensor) and value.shape == (VIEWS,) and value.dtype == torch.float64
                 and value.device == depths.device and not value.requires_grad and
                 bool(torch.isfinite(value).all()) and bool((value > 0).all()),
                 "Invalid frozen " + key)
    ridge = state.get("ridge")
    _require(isinstance(ridge, torch.Tensor) and ridge.ndim == 0 and ridge.dtype == torch.float64
             and ridge.device == depths.device and not ridge.requires_grad and
             bool(torch.isfinite(ridge)) and bool(ridge > 0), "Invalid frozen positive ridge")
    _require(isinstance(state.get("support"), list) and len(state["support"]) == VIEWS,
             "Frozen support must contain exactly five views")
    packets = _observations(state["support"], VIEWS, depths.shape[1], depths.shape[2],
                            depths.device, MINIMUM_SAMPLES)
    for packet, stored in zip(packets, state["support"]):
        for key in ("weight", "factor"):
            value = stored.get(key)
            _require(isinstance(value, torch.Tensor) and value.shape == (len(packet["xy"]),)
                     and value.dtype == torch.float64 and value.device == depths.device and not value.requires_grad
                     and bool(torch.isfinite(value).all()) and bool((value > 0).all()),
                     "Invalid frozen support " + key)
    return packets


@torch.no_grad()
def assemble_predictive_matrices(depths, state, query):
    """Return float64 A,b,Q,d,ridge without altering the fixed preparation.

Only required support/query predicted depths must be positive. All raster
values must be finite; no coordinate is dropped. Negative finite values away
from fitting/query coordinates are left untouched. Compute signed directional
matrices later as differences between these state matrices, not by preparing
another state from a depth direction.
"""
    depths = _depth(depths, "Current fused depth")
    support = _validate_state(state, depths)
    packets = _observations(query, VIEWS, depths.shape[1], depths.shape[2],
                            depths.device, MINIMUM_SAMPLES)
    metric = _metric(depths, state)
    identity = torch.eye(VIEWS, device=depths.device, dtype=torch.float64)
    blocks, targets = [], []
    for index, (fixed_support, packet) in enumerate(zip(support, packets)):
        support_ids = fixed_support["xy"][:, 1]*depths.shape[2]+fixed_support["xy"][:, 0]
        query_ids = packet["xy"][:, 1]*depths.shape[2]+packet["xy"][:, 0]
        _require(not bool(torch.isin(query_ids, support_ids).any()), "Support/query native pixel overlap")
        _require(bool((_at(depths[index], fixed_support["xy"]) > 0).all()) and
                 bool((_at(depths[index], packet["xy"]) > 0).all()),
                 "Current depth must be positive at every required support/query observation")
        factor = 1. / (state["sensor_normalizers"][index]*math.sqrt(VIEWS*len(packet["xy"])))
        blocks.append((_at(metric[index], packet["xy"])*factor)[:, None]*identity[index])
        targets.append(packet["depth"]*factor)
    a, b = assemble_system(depths, state, graph_weight=0.)
    q, d = torch.cat(blocks), torch.cat(targets)
    _require(bool(torch.isfinite(q).all()) and bool(torch.isfinite(d).all()), "Nonfinite predictive query system")
    # A single fixed scale vector makes the production bridge exercise the
    # original residual path without solving or selecting an adapter/state.
    probe = torch.tensor([.8, 1.1, .9, 1.2, 1.], device=depths.device, dtype=torch.float64)
    reference = query_residual(depths, state, query, probe)
    actual = q@probe-d
    _require(torch.allclose(actual, reference, rtol=1e-12, atol=1e-12),
             "Explicit query matrix disagrees with the original query residual")
    return dict(A=a, b=b, Q=q, d=d, ridge=state["ridge"].detach().clone())
