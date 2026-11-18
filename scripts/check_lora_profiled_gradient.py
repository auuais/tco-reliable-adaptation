"""Validate five-scale variable projection and its query hypergradient on CPU.

The network-output derivative is a prescribed scalar direction. This checks the
alignment mathematics, not MASt3R/VGGT differentiation or reconstruction quality.
"""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/lora_profiled_design/math_checks.json"


def scene_fixture():
    """Five calibrated cameras observe ten shared, noncoplanar landmarks."""
    centers = np.array([[-.4, -.1, 0.], [-.2, .15, 0.], [0., 0., 0.],
                        [.25, -.12, 0.], [.5, .08, 0.]])
    points = np.array([[-.5, -.3, 2.8], [-.2, .2, 3.1], [.1, -.4, 3.5],
                       [.4, .3, 4.1], [.6, -.1, 3.3], [-.6, .4, 4.4],
                       [.3, .1, 3.8], [-.3, -.2, 4.2], [.5, .4, 3.0],
                       [0., -.3, 4.6]])
    local = points[None] - centers[:, None]
    truth = local[..., 2]
    rays = local / truth[..., None]
    view = np.arange(5)[:, None]
    pixel = np.arange(10)[None]
    bias = np.array([.045, -.025, .035, -.04, .015])[:, None]
    spatial = .025 * np.sin(.8 * pixel + .3 * view)
    depth = truth * (1. + bias + spatial)
    direction = truth * (.08 + .05 * np.cos(.7 * pixel + .4 * view))
    return centers, rays, truth, depth, direction


def residual_blocks(centers, rays, truth, depth, direction):
    """Build A(theta) s - b from sparse axial depth and matched 3D points."""
    support, query = np.arange(6), np.arange(6, 10)
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))
    dimension = len(centers)
    a, da, b = [], [], []
    sensor_factor = 1. / (.06 * np.sqrt(dimension * len(support)))
    for i in range(dimension):
        for p in support:
            row, derivative = np.zeros(dimension), np.zeros(dimension)
            row[i], derivative[i] = depth[i, p], direction[i, p]
            a.append(sensor_factor * row)
            da.append(sensor_factor * derivative)
            b.append(sensor_factor * truth[i, p])
    match_factor = np.sqrt(.2 / (len(edges) * len(support))) / .06
    for i, j in edges:
        for p in support:
            for axis in range(3):
                row, derivative = np.zeros(dimension), np.zeros(dimension)
                row[i] = rays[i, p, axis] * depth[i, p]
                row[j] = -rays[j, p, axis] * depth[j, p]
                derivative[i] = rays[i, p, axis] * direction[i, p]
                derivative[j] = -rays[j, p, axis] * direction[j, p]
                a.append(match_factor * row)
                da.append(match_factor * derivative)
                b.append(match_factor * (centers[j, axis] - centers[i, axis]))
    q, dq, target = [], [], []
    query_factor = 1. / (.06 * np.sqrt(dimension * len(query)))
    for i in range(dimension):
        for p in query:
            row, derivative = np.zeros(dimension), np.zeros(dimension)
            row[i], derivative[i] = depth[i, p], direction[i, p]
            q.append(query_factor * row)
            dq.append(query_factor * derivative)
            target.append(query_factor * truth[i, p])
    return tuple(np.asarray(x) for x in (a, da, b, q, dq, target)), edges


def solve_scales(a, b, ridge):
    h = a.T @ a + ridge * np.eye(a.shape[1])
    rhs = a.T @ b + ridge * np.ones(a.shape[1])
    lower = np.linalg.cholesky(h)
    scales = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
    return scales, h, rhs


def loss_and_gradient(theta, blocks, ridge):
    a0, da, b, q0, dq, target = blocks
    a, q = a0 + theta * da, q0 + theta * dq
    scales, h, rhs = solve_scales(a, b, ridge)
    ds = np.linalg.solve(h, da.T @ (b - a @ scales) - a.T @ da @ scales)
    residual = q @ scales - target
    direct = float(residual @ (dq @ scales))
    implicit = float(residual @ (dq @ scales + q @ ds))
    return dict(loss=float(.5 * (residual @ residual)), gradient=implicit,
                detached_gradient=direct, scales=scales, scale_derivative=ds,
                normal_residual=float(np.max(np.abs(h @ scales-rhs))))


def input_diagnostics(a, edges, sensors_per_view):
    """Regularization cannot supply missing data identifiability or connectivity."""
    singular = np.linalg.svd(a, compute_uv=False)
    full_rank = singular[-1] > max(1e-12, 1e-6 * singular[0])
    connected = {0}
    while True:
        new = connected | {j for i, j in edges if i in connected} | {i for i, j in edges if j in connected}
        if new == connected:
            break
        connected = new
    return dict(data_full_rank=bool(full_rank), graph_connected=len(connected) == a.shape[1],
                every_view_anchored=all(n > 0 for n in sensors_per_view),
                singular_values=singular.tolist())


def checks():
    fixture = scene_fixture()
    blocks, edges = residual_blocks(*fixture)
    a0, da, b, q0, dq, target = blocks
    ridge, theta, epsilon = .02, .17, 1e-6
    result = loss_and_gradient(theta, blocks, ridge)
    left = loss_and_gradient(theta-epsilon, blocks, ridge)
    right = loss_and_gradient(theta+epsilon, blocks, ridge)
    finite = (right["loss"]-left["loss"]) / (2*epsilon)
    finite_scales = (right["scales"]-left["scales"]) / (2*epsilon)
    np.testing.assert_allclose(result["gradient"], finite, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(result["scale_derivative"], finite_scales, rtol=2e-6, atol=2e-8)
    assert result["normal_residual"] < 1e-10
    assert np.all(result["scales"] > 0)
    assert abs(result["gradient"]-result["detached_gradient"]) > 1e-3

    # Check the ridge Schur complement on a smaller sensor-only block. P is
    # symmetric positive semidefinite, and becomes a projector only at zero ridge.
    a = (a0+theta*da)[:30]
    residual = np.linspace(-.3, .4, len(a))
    h = a.T @ a + ridge*np.eye(5)
    delta = -np.linalg.solve(h, a.T @ residual)
    p = np.eye(len(a)) - a @ np.linalg.solve(h, a.T)
    actual = float(np.sum((residual+a@delta)**2)+ridge*(delta@delta))
    reduced = float(residual @ p @ residual)
    np.testing.assert_allclose(actual, reduced, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(p, p.T, atol=1e-12)
    minimum_eigenvalue = float(np.linalg.eigvalsh(p).min())
    assert minimum_eigenvalue >= -1e-12
    idempotence_error = float(np.linalg.norm(p@p-p))
    assert idempotence_error > 1e-7

    # Independent per-view reparameterizations of depth can be exactly absorbed
    # by unregularized scales, on identical support and query geometry.
    original, _, _ = solve_scales(a0, b, 0.)
    factors = np.array([.8, 1.3, .95, 1.2, .85])
    reparameterized, _, _ = solve_scales(a0*factors, b, 0.)
    np.testing.assert_allclose(reparameterized*factors, original, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose((q0*factors)@reparameterized, q0@original, rtol=1e-11, atol=1e-11)

    valid = input_diagnostics(a0, edges, [6]*5)
    assert valid["data_full_rank"] and valid["graph_connected"] and valid["every_view_anchored"]
    disconnected = input_diagnostics(a0, edges[:2]+((3,4),), [6]*5)
    assert disconnected["data_full_rank"] and not disconnected["graph_connected"]
    missing = a0.copy()
    missing[:,4] = 0.
    poor = input_diagnostics(missing, edges, [6,6,6,6,0])
    assert not poor["data_full_rank"] and not poor["every_view_anchored"]
    # A ridge solve is still possible for missing data. Detection is separate.
    unsupported, _, _ = solve_scales(missing, b, ridge)
    np.testing.assert_allclose(unsupported[4], 1., atol=1e-12)

    return dict(scope="Synthetic analytic check of five metric scales and a prescribed scalar network-output derivative. No model execution or reconstruction accuracy claim.",
        views=5, landmarks=10, support_depths=30, query_depths=20, matches=30,
        ridge=ridge, theta=theta, finite_difference_epsilon=epsilon,
        normal_equation_max_residual=result["normal_residual"], scales=result["scales"].tolist(),
        implicit_gradient=result["gradient"], finite_difference_gradient=float(finite),
        detached_gradient=result["detached_gradient"], gradient_absolute_error=abs(result["gradient"]-finite),
        scale_derivative_max_error=float(np.max(abs(result["scale_derivative"]-finite_scales))),
        reduced_cost=actual, schur_cost=reduced, schur_minimum_eigenvalue=minimum_eigenvalue,
        ridge_projector_idempotence_error=idempotence_error, gauge_invariance_pass=True,
        healthy_inputs=valid, disconnected_inputs=disconnected, missing_view_inputs=poor,
        unsupported_view_ridge_scale=float(unsupported[4]), pass_all=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = checks()
    if args.verify:
        old = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for key in ("implicit_gradient", "finite_difference_gradient", "detached_gradient", "reduced_cost", "schur_cost"):
            np.testing.assert_allclose(report[key], old[key], rtol=2e-6, atol=2e-8)
        assert old["pass_all"]
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:report[k] for k in ("pass_all", "gradient_absolute_error", "implicit_gradient", "detached_gradient")}))
