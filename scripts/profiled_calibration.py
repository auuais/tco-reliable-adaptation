"""Differentiable five-view calibration with fixed sensor and matching weights.

Camera-to-world centers must already be metric. Depths may be in model units.
Only the current depth argument participates in autograd; preparation clones
and detaches every tensor used as a fixed observation or normalization.
"""
import math

import torch


def canonical_depth(pointmaps, confidences):
    """Return pinned MASt3R canonical axial depth and fused point confidence."""
    if pointmaps.ndim != 4 or pointmaps.shape[-1] != 3 or pointmaps.shape[:-1] != confidences.shape:
        raise ValueError("Expected partner pointmaps [P,H,W,3] and confidences [P,H,W]")
    if not len(pointmaps) or not torch.isfinite(pointmaps).all():
        raise ValueError("Empty or nonfinite partner geometry")
    weights = confidences.detach().to(device=pointmaps.device, dtype=pointmaps.dtype) - .999
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Canonical confidence minus 0.999 must be finite and positive")
    denominator = weights.sum(dim=0)
    canonical = (weights[..., None] * pointmaps).sum(dim=0) / denominator[..., None]
    confidence = weights.square().sum(dim=0) / denominator
    return canonical[..., 2], confidence.detach()


def _fixed(value, device, dtype=torch.float64):
    return torch.as_tensor(value, device=device, dtype=dtype).detach().clone()


def _pixels(value, height, width, device):
    raw = _fixed(value, device)
    if raw.ndim != 2 or raw.shape[1] != 2 or not torch.isfinite(raw).all():
        raise ValueError("Pixels must be finite [N,2] native x,y coordinates")
    xy = raw.to(dtype=torch.long)
    if not torch.equal(raw, xy.to(dtype=raw.dtype)):
        raise ValueError("Fractional native pixel coordinates are unsupported")
    if len(xy) and ((xy < 0).any() or (xy[:, 0] >= width).any() or (xy[:, 1] >= height).any()):
        raise ValueError("Native pixel outside depth raster")
    return xy


def _observations(items, views, height, width, device, minimum):
    if len(items) != views:
        raise ValueError("One observation packet per view is required")
    result = []
    for item in items:
        xy = _pixels(item["xy"], height, width, device)
        depth = _fixed(item["depth"], device)
        if depth.shape != (len(xy),) or len(xy) < minimum:
            raise ValueError("Insufficient samples or incompatible observation depth shape")
        if not torch.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError("Measured axial depth must be finite and positive")
        if len(torch.unique(xy, dim=0)) != len(xy):
            raise ValueError("Duplicate native observation pixels require prior grouping")
        result.append(dict(xy=xy, depth=depth))
    return result


def _at(depth, xy):
    return depth[xy[:, 1], xy[:, 0]]


def _huber_weight(residual):
    return 1. / torch.clamp(residual.abs(), min=1.)


def _world_rays(xy, intrinsic, rotation):
    homogeneous = torch.cat([xy.to(dtype=torch.float64), torch.ones_like(xy[:, :1], dtype=torch.float64)], dim=1)
    rays = homogeneous @ torch.linalg.inv(intrinsic).T
    if not torch.allclose(rays[:, 2], torch.ones_like(rays[:, 2]), rtol=0., atol=1e-10):
        raise ValueError("Intrinsics must produce axial-depth rays with z equal to one")
    return rays @ rotation.T


def prepare_calibration(initial_depths, intrinsics, c2w, support, matches, fused_confidences,
                        *, metric_scales=None, minimum_samples=8):
    """Freeze support-only normalization, priors, camera rays, and robust weights.

Support packets contain integer ``xy`` and measured metric ``depth`` arrays.
Each graph edge contains ``i``, ``j``, ``xy_i`` and ``xy_j``, with i < j.
An optional metric_scales vector supplies the final endpoint's common gauge.
"""
    if initial_depths.ndim != 3 or initial_depths.shape[0] != 5:
        raise ValueError("This registered layer requires five depth views [5,H,W]")
    if minimum_samples < 1:
        raise ValueError("Minimum sample count must be positive")
    device = initial_depths.device
    initial = _fixed(initial_depths, device)
    if not torch.isfinite(initial).all():
        raise ValueError("Initial depth contains nonfinite values")
    views, height, width = initial.shape
    intrinsic = _fixed(intrinsics, device)
    if intrinsic.shape == (3, 3):
        intrinsic = intrinsic[None].repeat(views, 1, 1)
    camera = _fixed(c2w, device)
    confidence = _fixed(fused_confidences, device)
    if intrinsic.shape != (views, 3, 3) or camera.shape != (views, 4, 4) or confidence.shape != initial.shape:
        raise ValueError("Camera or fused confidence shape differs from five-view geometry")
    if not torch.isfinite(intrinsic).all() or not torch.isfinite(camera).all():
        raise ValueError("Nonfinite fixed camera matrices")
    expected_bottom = camera.new_tensor([0., 0., 0., 1.]).expand(views, -1)
    if not torch.allclose(camera[:, 3], expected_bottom, rtol=0., atol=1e-10):
        raise ValueError("Expected homogeneous camera-to-world transforms")
    packets = _observations(support, views, height, width, device, minimum_samples)
    medians = torch.stack([torch.quantile(p["depth"], .5) for p in packets])
    normalizers = .02 * medians
    gains = []
    for i, packet in enumerate(packets):
        z = _at(initial[i], packet["xy"])
        if (z <= 0).any():
            raise ValueError("Initial support prediction must be positive before metric conversion")
        gains.append(torch.quantile(packet["depth"] / z, .5))
    gains = torch.stack(gains)
    if metric_scales is not None:
        gains = _fixed(metric_scales, device)
        if gains.shape != (views,) or not torch.isfinite(gains).all() or (gains <= 0).any():
            raise ValueError("Metric conversion must contain five finite positive scales")
    metric = initial * gains[:, None, None]
    for i, packet in enumerate(packets):
        residual = (_at(metric[i], packet["xy"]) - packet["depth"]) / normalizers[i]
        packet["weight"] = _huber_weight(residual).detach().clone()
        packet["factor"] = torch.sqrt(packet["weight"] / (views*len(packet["xy"]))) / normalizers[i]

    edges, filtering = [], []
    seen = set()
    for edge in matches:
        i, j = int(edge["i"]), int(edge["j"])
        if not 0 <= i < j < views or (i, j) in seen:
            raise ValueError("Graph edges must be unique unordered view pairs i < j")
        seen.add((i, j))
        first = _pixels(edge["xy_i"], height, width, device)
        second = _pixels(edge["xy_j"], height, width, device)
        if len(first) != len(second):
            raise ValueError("Matching endpoint counts differ")
        if not len(first):
            continue
        paired = torch.cat([first, second], dim=1)
        if len(torch.unique(paired, dim=0)) != len(paired):
            raise ValueError("Duplicate pixel correspondences require prior grouping")
        ci, cj = _at(confidence[i], first), _at(confidence[j], second)
        valid_confidence = torch.isfinite(ci) & torch.isfinite(cj) & (ci > 0) & (cj > 0)
        valid_depth = (_at(metric[i], first) > 0) & (_at(metric[j], second) > 0)
        valid = valid_confidence & valid_depth
        rejected = int((~valid_confidence).sum())
        filtering.append(dict(i=i, j=j, initial_count=len(first), retained_count=int(valid.sum()),
                              invalid_confidence=rejected, nonpositive_depth=int((~valid_depth).sum())))
        first, second, ci, cj = first[valid], second[valid], ci[valid], cj[valid]
        if not len(first):
            continue
        fused = torch.sqrt(ci) * torch.sqrt(cj)
        fused = fused / fused.mean()
        ray_i = _world_rays(first, intrinsic[i], camera[i, :3, :3])
        ray_j = _world_rays(second, intrinsic[j], camera[j, :3, :3])
        target = camera[j, :3, 3] - camera[i, :3, 3]
        normalizer = .02 * torch.sqrt(medians[i]*medians[j])
        residual = (_at(metric[i], first)[:, None] * ray_i
                    - _at(metric[j], second)[:, None] * ray_j - target)
        robust = _huber_weight(torch.linalg.vector_norm(residual / normalizer, dim=1))
        edges.append(dict(i=i, j=j, xy_i=first, xy_j=second, ray_i=ray_i, ray_j=ray_j,
                          target=target, confidence=fused, weight=robust,
                          normalizer=normalizer, rejected_confidence=rejected))
    for edge in edges:
        count = len(edge["xy_i"])
        edge["factor"] = torch.sqrt(edge["confidence"]*edge["weight"] / (len(edges)*count)) / edge["normalizer"]
    state = dict(shape=tuple(initial.shape), gains=gains.detach().clone(), sensor_normalizers=normalizers.detach().clone(),
                 support=packets, edges=edges, intrinsics=intrinsic, c2w=camera,
                 graph_filtering=filtering, minimum_samples=minimum_samples, ridge=None)
    sensor_matrix, _ = assemble_system(initial, state, graph_weight=0.)
    state["ridge"] = (.001 * sensor_matrix.square().sum() / views).detach().clone()
    if not torch.isfinite(state["ridge"]) or state["ridge"] <= 0:
        raise ValueError("Initial sensor-only ridge must be finite and positive")
    return state


def _metric(depths, state):
    if tuple(depths.shape) != state["shape"] or not torch.isfinite(depths).all():
        raise ValueError("Current depth shape or finite check failed")
    if depths.device != state["gains"].device:
        raise ValueError("Current depths and frozen state must share a device")
    return depths.to(dtype=torch.float64) * state["gains"][:, None, None]


def assemble_system(depths, state, graph_weight=.1):
    """Return differentiable A and fixed b in float64, in sensor-then-edge order."""
    if not math.isfinite(graph_weight) or graph_weight < 0:
        raise ValueError("Graph weight must be finite and nonnegative")
    metric = _metric(depths, state)
    views = len(metric)
    identity = torch.eye(views, dtype=torch.float64, device=depths.device)
    matrices, targets = [], []
    for i, packet in enumerate(state["support"]):
        factor = packet["factor"]
        matrices.append((_at(metric[i], packet["xy"])*factor)[:, None] * identity[i])
        targets.append(packet["depth"]*factor)
    if graph_weight > 0:
        for edge in state["edges"]:
            i, j = edge["i"], edge["j"]
            factor = math.sqrt(graph_weight)*edge["factor"]
            first = _at(metric[i], edge["xy_i"])[:, None] * edge["ray_i"]
            second = _at(metric[j], edge["xy_j"])[:, None] * edge["ray_j"]
            block = first[..., None]*identity[i] - second[..., None]*identity[j]
            matrices.append((block*factor[:, None, None]).reshape(-1, views))
            targets.append((edge["target"][None] * factor[:, None]).reshape(-1))
    a, b = torch.cat(matrices), torch.cat(targets)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite calibration system")
    return a, b


def solve_scales(a, b, ridge, detach=False):
    """Solve the fixed-weight ridge objective, optionally detach only its solution."""
    if a.ndim != 2 or a.shape[1] != 5 or b.shape != (len(a),):
        raise ValueError("Expected a [N,5] calibration matrix and [N] targets")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite calibration matrix or target")
    a, b = a.to(torch.float64), b.to(torch.float64)
    penalty = _fixed(ridge, a.device)
    if penalty.ndim != 0 or not torch.isfinite(penalty) or penalty <= 0:
        raise ValueError("Ridge strength must be finite and positive")
    identity = torch.eye(5, dtype=torch.float64, device=a.device)
    hessian = a.T @ a + penalty*identity
    rhs = a.T @ b + penalty*torch.ones(5, dtype=torch.float64, device=a.device)
    cholesky = torch.linalg.cholesky(hessian)
    scales = torch.cholesky_solve(rhs[:, None], cholesky).squeeze(1)
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError(f"Invalid unconstrained calibration scales: {scales.detach().cpu().tolist()}")
    return scales.detach() if detach else scales


def query_residual(depths, state, query, scales):
    """Whiten query residuals using only support-derived normalizers."""
    metric = _metric(depths, state)
    views, height, width = state["shape"]
    if scales.shape != (views,) or not torch.isfinite(scales).all():
        raise ValueError("Expected five finite fitted scales")
    packets = _observations(query, views, height, width, depths.device, state["minimum_samples"])
    residuals = []
    for i, packet in enumerate(packets):
        support_ids = state["support"][i]["xy"][:, 1]*width + state["support"][i]["xy"][:, 0]
        query_ids = packet["xy"][:, 1]*width + packet["xy"][:, 0]
        if torch.isin(query_ids, support_ids).any():
            raise ValueError("Support/query native pixel overlap")
        factor = 1. / (state["sensor_normalizers"][i]*math.sqrt(views*len(packet["xy"])))
        residuals.append((scales[i]*_at(metric[i], packet["xy"])-packet["depth"])*factor)
    return torch.cat(residuals)


def profiled_objective(depths, state, query, graph_weight=.1, detach_solve=False):
    a, b = assemble_system(depths, state, graph_weight)
    scales = solve_scales(a, b, state["ridge"], detach=detach_solve)
    residual = query_residual(depths, state, query, scales)
    return .5*residual.square().sum(), scales


def system_diagnostics(a, state, scales):
    """Describe data observability separately from ridge solvability."""
    singular = torch.linalg.svdvals(a.detach().double()).cpu()
    tolerance = torch.finfo(torch.float64).eps * max(a.shape) * float(singular[0])
    components = []
    remaining = set(range(state["shape"][0]))
    pairs = [(edge["i"], edge["j"]) for edge in state["edges"]]
    while remaining:
        component = {min(remaining)}
        while True:
            expanded = component | {j for i, j in pairs if i in component} | {i for i, j in pairs if j in component}
            if expanded == component:
                break
            component = expanded
        components.append(sorted(component))
        remaining -= component
    counts = [len(p["xy"]) for p in state["support"]]
    return dict(singular_values=singular.tolist(), data_rank=int((singular>tolerance).sum()),
                singular_ratio=float(singular[-1]/singular[0]) if singular[0] > 0 else 0.,
                support_counts=counts, every_view_supported=all(n>=state["minimum_samples"] for n in counts),
                graph_components=components, graph_connected=len(components)==1,
                ridge=float(state["ridge"]), positive_scales=bool((scales>0).all()),
                scales=scales.detach().cpu().tolist(),
                graph_filtering=state["graph_filtering"],
                rejected_graph_confidences=sum(e["invalid_confidence"] for e in state["graph_filtering"]))
