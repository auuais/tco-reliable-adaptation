"""Pure float64 mathematics for the registered, unlabeled D0 diagnosis.

No artifact loader, model, sensor depth or reference is accessed here.  Output
differences are fixed-input secants, never a claimed network Jacobian.  Invalid
coordinates are not removed and equal-pair summaries never become pooled means.
"""
import math

import numpy as np


QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
SYNTHETIC_RTOL = 1e-12
SYNTHETIC_ATOL = 1e-12
GRAM_ROUNDOFF_MULTIPLIER = 64.0
NATIVE_SHAPE = (384, 512)

DEPTH_SCALARS = (
    "base_rms", "candidate_rms", "delta_mean", "delta_abs_mean", "delta_rms",
    "delta_l2", "relative_delta_l2", "scalar_coefficient", "scalar_rms",
    "scalar_l2", "shape_rms", "shape_l2", "shape_fraction_l2",
    "scalar_energy_fraction", "shape_energy_fraction",
    "decomposition_reconstruction_max_abs", "base_shape_inner_product",
)
DEPTH_QUANTILES = (
    "signed_delta_quantiles", "absolute_delta_quantiles", "absolute_shape_quantiles",
)
INVALID_COUNTS = (
    "base_nonfinite_count", "candidate_nonfinite_count",
    "base_nonpositive_finite_count", "candidate_nonpositive_finite_count",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real_array(value, name, ndim):
    array = np.asarray(value)
    _require(array.ndim == ndim and all(array.shape), name + " has wrong or empty shape")
    _require(array.dtype.kind in "fiu", name + " must contain real numeric values")
    return np.asarray(array, dtype=np.float64)


def _number(value, name, nonnegative=False):
    _require(not isinstance(value, (bool, np.bool_)), name + " cannot be boolean")
    _require(isinstance(value, (int, float, np.integer, np.floating)), name + " must be numeric")
    result = float(value)
    _require(math.isfinite(result), name + " must be finite")
    _require(not nonnegative or result >= 0, name + " must be nonnegative")
    return result


def _finite(value):
    result = float(value)
    _require(math.isfinite(result), "Float64 arithmetic overflow or invalid result")
    return result


def effective_weight_stats(A, B, base_norm, rank=4, alpha=1):
    """Measure (alpha/rank) B A without materializing the full weight update.

    A is [rank,in_features], B is [out_features,rank].  ``base_norm`` must be the
    Frobenius norm of this exact original frozen matrix, not the whole model.
    The matrices here are LoRA factors, not the calibration design matrix.
    """
    _require(isinstance(rank, (int, np.integer)) and not isinstance(rank, (bool, np.bool_))
             and rank > 0, "rank must be a positive integer")
    rank = int(rank)
    alpha = _number(alpha, "alpha")
    base_norm = _number(base_norm, "base_norm", nonnegative=True)
    A = _real_array(A, "A", 2)
    B = _real_array(B, "B", 2)
    _require(A.shape[0] == rank and B.shape[1] == rank,
             "Factor orientation/rank mismatch: expected A[r,in], B[out,r]")
    _require(np.isfinite(A).all() and np.isfinite(B).all(), "LoRA factors must be finite")
    with np.errstate(over="raise", invalid="raise"):
        terms = (B.T @ B) * (A @ A.T).T
    _require(np.isfinite(terms).all(), "Gram calculation overflow")
    norm_squared = _finite(math.fsum(float(x) for x in terms.ravel()))
    absolute_sum = _finite(math.fsum(abs(float(x)) for x in terms.ravel()))
    tolerance = GRAM_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * absolute_sum
    _require(norm_squared >= -tolerance, "Gram norm is negative beyond float64 roundoff")
    clamped = norm_squared < 0
    norm_squared = max(0.0, norm_squared)
    scale = _finite(alpha / rank)
    update = _finite(abs(scale) * math.sqrt(norm_squared))
    relative = None if base_norm == 0 else _finite(update / base_norm)
    return dict(
        schema="profiled_d0_weight_stats_v1", factor_A_shape=list(A.shape),
        factor_B_shape=list(B.shape), rank=rank, alpha=alpha, scaling=scale,
        factors_finite=True, unscaled_product_frobenius_squared=norm_squared,
        gram_roundoff_clamped=clamped, gram_roundoff_bound=float(tolerance),
        update_frobenius=update, base_frobenius=base_norm,
        relative_frobenius=relative, zero_base_denominator=base_norm == 0,
        denominator_scope="corresponding_original_frozen_matrix",
    )


def aggregate_weight_stats(records):
    """Root-sum-square norms over adapted matrices only, not mean ratios."""
    records = list(records)
    _require(len(records) > 0, "No adapted matrices to summarize")
    updates = [_number(r["update_frobenius"], "update norm", True) for r in records]
    bases = [_number(r["base_frobenius"], "base norm", True) for r in records]
    update = _finite(math.sqrt(math.fsum(x*x for x in updates)))
    base = _finite(math.sqrt(math.fsum(x*x for x in bases)))
    return dict(
        schema="profiled_d0_weight_aggregate_v1", module_count=len(records),
        update_frobenius=update, base_frobenius=base,
        relative_frobenius=None if base == 0 else _finite(update / base),
        zero_base_denominator=base == 0,
        denominator_scope="adapted_original_frozen_matrices_only",
        aggregation="sqrt(sum(layer_frobenius_squared)); ratio_of_aggregate_norms",
    )


def grid_coordinates():
    """Registered row-major native (x,y) grid: start 8, stride 16, 768 points."""
    y, x = np.meshgrid(np.arange(8, 384, 16, dtype=np.int64),
                       np.arange(8, 512, 16, dtype=np.int64), indexing="ij")
    return np.column_stack((x.ravel(), y.ravel()))


def _coordinates(xy, shape):
    xy = np.asarray(xy)
    _require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy) > 0,
             "Coordinates must be a nonempty [N,2] array")
    _require(xy.dtype.kind in "iu", "Coordinates must be integers in (x,y) order")
    _require(((xy >= 0) & (xy < [shape[1], shape[0]])).all(), "Coordinate outside depth map")
    _require(len(set(map(tuple, xy.tolist()))) == len(xy), "Duplicate diagnostic coordinates")
    return xy.astype(np.int64, copy=False)


def compare_depths(base, candidate, xy):
    """Compare two raw self-camera depth maps on exactly the supplied pixels.

    Finite nonpositive depths remain in arithmetic, with invalid geometry flagged.
    If any selected value is nonfinite, every numeric comparison statistic is
    undefined (None); the count is unchanged and no finite-only statistic is made.
    Scalar/shape components are orthogonal secant components in this coordinate
    vector, not fitted predictions, calibrated endpoints or network derivatives.
    """
    base = _real_array(base, "base", 2)
    candidate = _real_array(candidate, "candidate", 2)
    _require(base.shape == candidate.shape, "Depth-map shapes differ")
    xy = _coordinates(xy, base.shape)
    d0 = base[xy[:, 1], xy[:, 0]]
    d1 = candidate[xy[:, 1], xy[:, 0]]
    f0, f1 = np.isfinite(d0), np.isfinite(d1)
    all_finite = bool(f0.all() and f1.all())
    out = dict(
        schema="profiled_d0_depth_comparison_v1", count=int(len(xy)),
        map_shape=list(base.shape), selected_coordinates_filtered=False,
        base_nonfinite_count=int((~f0).sum()), candidate_nonfinite_count=int((~f1).sum()),
        base_nonpositive_finite_count=int((f0 & (d0 <= 0)).sum()),
        candidate_nonpositive_finite_count=int((f1 & (d1 <= 0)).sum()),
        numeric_statistics_defined=all_finite,
        geometrically_valid=bool(all_finite and (d0 > 0).all() and (d1 > 0).all()),
        quantile_probabilities=list(QUANTILES),
        quantile_method="linear", zero_base_denominator=None, zero_change=None,
        interpretation="Unlabeled raw-output secant; not a network Jacobian or accuracy score",
    )
    out.update({key: None for key in DEPTH_SCALARS + DEPTH_QUANTILES})
    if not all_finite:
        return out
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        delta = d1 - d0
        base_sq = _finite(np.dot(d0, d0))
        candidate_sq = _finite(np.dot(d1, d1))
        delta_sq = _finite(np.dot(delta, delta))
        norm = _finite(math.sqrt(delta_sq))
        out.update(
            base_rms=_finite(math.sqrt(base_sq / len(d0))),
            candidate_rms=_finite(math.sqrt(candidate_sq / len(d0))),
            delta_mean=_finite(np.mean(delta)), delta_abs_mean=_finite(np.mean(np.abs(delta))),
            delta_rms=_finite(math.sqrt(delta_sq / len(delta))), delta_l2=norm,
            signed_delta_quantiles=np.quantile(delta, QUANTILES, method="linear").tolist(),
            absolute_delta_quantiles=np.quantile(np.abs(delta), QUANTILES, method="linear").tolist(),
            zero_base_denominator=base_sq == 0, zero_change=delta_sq == 0,
        )
        if base_sq == 0:
            return out
        coefficient = _finite(np.dot(d0, delta) / base_sq)
        scalar = coefficient * d0
        shape = delta - scalar
        scalar_sq = _finite(np.dot(scalar, scalar))
        shape_sq = _finite(np.dot(shape, shape))
        out.update(
            relative_delta_l2=_finite(norm / math.sqrt(base_sq)),
            scalar_coefficient=coefficient,
            scalar_rms=_finite(math.sqrt(scalar_sq / len(delta))),
            scalar_l2=_finite(math.sqrt(scalar_sq)),
            shape_rms=_finite(math.sqrt(shape_sq / len(delta))),
            shape_l2=_finite(math.sqrt(shape_sq)),
            shape_fraction_l2=None if delta_sq == 0 else _finite(math.sqrt(shape_sq / delta_sq)),
            scalar_energy_fraction=None if delta_sq == 0 else _finite(scalar_sq / delta_sq),
            shape_energy_fraction=None if delta_sq == 0 else _finite(shape_sq / delta_sq),
            decomposition_reconstruction_max_abs=_finite(np.max(np.abs(delta - (scalar + shape)))),
            base_shape_inner_product=_finite(np.dot(d0, shape)),
            absolute_shape_quantiles=np.quantile(np.abs(shape), QUANTILES, method="linear").tolist(),
        )
    return out


def aggregate_equal_pairs(records):
    """Mean per-pair statistics with no point pooling or undefined-pair removal.

    A statistic is undefined at aggregate level if any declared pair's statistic
    is undefined. Quantile means are summaries of per-pair quantiles, not pooled
    quantiles. Preserve the full pair records in the caller's evidence report.
    """
    records = list(records)
    _require(len(records) > 0, "No directed pairs to summarize")
    for record in records:
        _require(record.get("schema") == "profiled_d0_depth_comparison_v1", "Wrong comparison schema")
        _require(record.get("selected_coordinates_filtered") is False, "Coordinate filtering is forbidden")
        _require(type(record.get("count")) is int and record["count"] > 0, "Invalid pair point count")
        _require(record.get("quantile_probabilities") == list(QUANTILES), "Quantile specification differs")
    means = {}
    undefined = {}
    for key in DEPTH_SCALARS:
        values = [r[key] for r in records]
        undefined[key] = sum(value is None for value in values)
        means[key] = None if undefined[key] else _finite(math.fsum(_number(v, key) for v in values) / len(values))
    quantiles = {}
    for key in DEPTH_QUANTILES:
        values = [r[key] for r in records]
        undefined[key] = sum(value is None for value in values)
        if undefined[key]:
            quantiles[key] = None
        else:
            _require(all(len(value) == len(QUANTILES) for value in values), "Wrong quantile vector length")
            quantiles[key] = [_finite(math.fsum(_number(value[i], key) for value in values) / len(values))
                              for i in range(len(QUANTILES))]
    totals = {}
    for key in INVALID_COUNTS:
        _require(all(type(r[key]) is int and 0 <= r[key] <= r["count"] for r in records),
                 "Invalid diagnostic count")
        totals[key] = sum(r[key] for r in records)
    return dict(
        schema="profiled_d0_equal_pair_aggregate_v1", pair_count=len(records),
        total_selected_points=sum(r["count"] for r in records),
        aggregation="equal arithmetic mean across all declared directed pairs; no point pooling",
        undefined_policy="any undefined constituent makes that aggregate statistic undefined",
        pairs_with_defined_numeric_statistics=sum(r["numeric_statistics_defined"] is True for r in records),
        geometrically_valid_pairs=sum(r["geometrically_valid"] is True for r in records),
        invalid_counts=totals, means=means, mean_of_pair_quantiles=quantiles,
        quantile_probabilities=list(QUANTILES), undefined_pairs_per_statistic=undefined,
    )
