"""All-observation native MASt3R refinement in one explicit metric gauge.

This endpoint performs no adapter update and has no support/query loss. Callers
must supply only the allowed adaptation observations, preserve the native output
separately, and score every candidate using the same independently fixed mask.
"""
import torch

from profiled_calibration import (_observations, assemble_system,
                                  prepare_calibration, solve_scales,
                                  system_diagnostics)


@torch.no_grad()
def refine_native(native_depth, intrinsics, native_c2w, samples, matches,
                  fused_confidences, *, minimum_samples=8):
    """Return ``(refined_depth, metric_c2w, scales, diagnostics)``.

Inputs follow ``prepare_calibration``: five axial depth rasters, native optimized
    K/camera-to-world transforms, per-view ``xy``/metric ``depth`` packets, and
    fresh native graph/confidence. Outputs use float64 on the depth's device.
    K and rotations are unchanged; centers become g0 * (C_i - C_0). Five fitted
    relative scales affect depth only. No supplied/reference poses are accepted.

The pooled midpoint median sets g0. Nonpositive predictions at any allowed fit
    sample fail explicitly; no candidate-specific observation mask is applied.
    Negative dense pixels outside the samples remain in the output and are
    counted. Downstream world points are C_metric + R_native K_native^-1 p z_final,
    with neither g0 nor the relative scale applied again.
    """
    if not isinstance(native_depth, torch.Tensor) or native_depth.ndim != 3 or native_depth.shape[0] != 5:
        raise ValueError("Expected five native axial-depth rasters [5,H,W]")
    if minimum_samples < 1:
        raise ValueError("Minimum sample count must be positive")
    device = native_depth.device
    depth = native_depth.detach().to(dtype=torch.float64).clone()
    if not torch.isfinite(depth).all():
        raise ValueError("Native dense depth contains nonfinite values")
    views, height, width = depth.shape
    camera = torch.as_tensor(native_c2w, device=device, dtype=torch.float64).detach().clone()
    if camera.shape != (views, 4, 4) or not torch.isfinite(camera).all():
        raise ValueError("Expected five finite native camera-to-world transforms")
    packets = _observations(samples, views, height, width, device, minimum_samples)
    predicted = [depth[i, p["xy"][:, 1], p["xy"][:, 0]] for i, p in enumerate(packets)]
    invalid_counts = [int((z <= 0).sum()) for z in predicted]
    if any(invalid_counts):
        raise ValueError(
            "Nonpositive native predictions at allowed fitting samples: "
            f"counts_by_view={invalid_counts}; no candidate-specific exclusion is permitted"
        )
    ratios = torch.cat([p["depth"] / z for p, z in zip(packets, predicted)])
    if not torch.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("All allowed measured/native depth ratios must be finite and positive")
    g0 = torch.quantile(ratios, .5)
    metric_c2w = camera.clone()
    metric_c2w[:, :3, 3] = g0 * (camera[:, :3, 3] - camera[0, :3, 3])
    # Keep depth in native units. The explicit common gain avoids both default
    # per-view normalization and accidentally applying the metric conversion twice.
    state = prepare_calibration(
        depth, intrinsics, metric_c2w, packets, matches, fused_confidences,
        metric_scales=torch.ones(views, dtype=torch.float64, device=device) * g0,
        minimum_samples=minimum_samples,
    )
    matrix, target = assemble_system(depth, state, graph_weight=.1)
    scales = solve_scales(matrix, target, state["ridge"])
    refined_depth = scales[:, None, None] * g0 * depth
    before = matrix @ torch.ones_like(scales) - target
    after = matrix @ scales - target
    penalty = .5 * state["ridge"] * (scales - 1.).square().sum()
    stationarity = matrix.T @ after + state["ridge"] * (scales - 1.)
    diagnostics = system_diagnostics(matrix, state, scales)
    diagnostics.update(
        endpoint="native KDP followed by all-allowed-observation five-scale refinement",
        g0=float(g0), metric_scales=state["gains"].cpu().tolist(),
        gauge_estimator="pooled positive measured/native ratios; midpoint median",
        gauge_ratio_count=len(ratios), sample_counts=[len(p["xy"]) for p in packets],
        sample_exclusions=0, nonpositive_fit_predictions=invalid_counts,
        nonpositive_dense_pixels=[int((z <= 0).sum()) for z in depth],
        graph_weight=.1, native_origin=camera[0, :3, 3].cpu().tolist(),
        sensor_normalizers=state["sensor_normalizers"].cpu().tolist(),
        weighted_fit_before=float(.5 * before.square().sum()),
        weighted_fit_after=float(.5 * after.square().sum()),
        ridge_penalty_after=float(penalty),
        penalized_fit_after=float(.5 * after.square().sum() + penalty),
        normal_equation_max_residual=float(stationarity.abs().max()),
        fit_is_accuracy_metric=False,
    )
    return refined_depth, metric_c2w, scales, diagnostics
