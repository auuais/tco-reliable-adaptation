"""Query-space calibration compensation and matched isotropic damping."""
import math
import torch

from profiled_calibration import (_at, _metric, _observations, assemble_system,
                                  query_residual, solve_scales)


@torch.no_grad()
def prepare_reference(a0, q0, b, ridge):
    """Set per-camera damping from the exact local response at the pristine state."""
    for matrix in (a0, q0):
        if matrix.ndim != 2 or matrix.shape[1] != 5 or matrix.dtype != torch.float64:
            raise ValueError("Expected a five-column float64 graph-zero matrix")
        if not bool((torch.count_nonzero(matrix, dim=1) == 1).all()):
            raise ValueError("Compensation reference requires one camera per row")
    n_support = torch.count_nonzero(a0, dim=0)
    if not bool((n_support > 0).all()) or not bool((torch.count_nonzero(q0, dim=0) > 0).all()):
        raise ValueError("Each camera needs support and query observations")
    s0 = solve_scales(a0, b, ridge)
    ridge = torch.as_tensor(ridge, dtype=a0.dtype, device=a0.device)
    residual = a0 @ s0 - b
    active = a0 != 0
    row_response = a0 + active * (residual[:, None] / s0)
    h = a0.square().sum(dim=0) + ridge
    trace = q0.square().sum(dim=0) * row_response.square().sum(dim=0) / h.square()
    if not bool(torch.isfinite(trace).all()):
        raise ValueError("Nonfinite calibration compensation curvature")
    return dict(a0=a0.detach().clone(), s0=s0.detach().clone(),
                damping=(trace / n_support).detach().clone(), response_trace=trace.detach().clone())


def penalty_value(a, q, scales, reference, penalty):
    if penalty == "compensation":
        scale_change = scales - reference["s0"]
        return .5 * (q @ scale_change).square().sum()
    if penalty == "matched_support":
        support_change = (a - reference["a0"]) * reference["s0"]
        return .5 * (support_change.square() @ reference["damping"]).sum()
    if penalty == "none":
        return scales.new_zeros(())
    raise ValueError("Unknown compensation penalty")


def objective(a, q, b, d, ridge, reference, *, penalty, detach_solve=False):
    """Matrix-level objective for arithmetic and finite-difference checks."""
    scales = solve_scales(a, b, ridge, detach=detach_solve)
    base = .5 * (q @ scales - d).square().sum()
    regularizer = penalty_value(a, q, scales, reference, penalty)
    return base + regularizer, scales, dict(base=base, regularizer=regularizer)


def query_matrix(depths, state, query):
    """Build differentiable Q from the same frozen gauges and query coordinates."""
    metric = _metric(depths, state)
    views, height, width = state["shape"]
    packets = _observations(query, views, height, width, depths.device, state["minimum_samples"])
    identity = torch.eye(views, dtype=metric.dtype, device=metric.device)
    blocks = []
    for i, packet in enumerate(packets):
        factor = 1. / (state["sensor_normalizers"][i] * math.sqrt(views * len(packet["xy"])))
        blocks.append((_at(metric[i], packet["xy"]) * factor)[:, None] * identity[i])
    return torch.cat(blocks)


@torch.no_grad()
def prepare_fold(pristine_depth, state, query):
    a0, b = assemble_system(pristine_depth, state, graph_weight=0.)
    q0 = query_matrix(pristine_depth, state, query)
    return prepare_reference(a0, q0, b, state["ridge"])


def compensation_loss(depths, state, query, reference, *, penalty, detach_solve=False):
    """Production-shaped loss for the later registered two-pass runner."""
    a, b = assemble_system(depths, state, graph_weight=0.)
    scales = solve_scales(a, b, state["ridge"], detach=detach_solve)
    # Keep the original operation ordering for the base P3 objective.
    base = .5 * query_residual(depths, state, query, scales).square().sum()
    q = query_matrix(depths, state, query) if penalty == "compensation" else None
    regularizer = penalty_value(a, q, scales, reference, penalty)
    return base + regularizer, scales, dict(base=base, regularizer=regularizer)
