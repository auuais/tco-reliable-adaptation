"""Estimate a shared metric depth-to-RGB extrinsic using only image matches and pose priors."""
import numpy as np

ROTATION_UNIT = np.deg2rad(1.5)
TRANSLATION_UNIT = .025


def homogeneous(poses):
    e = np.asarray(poses, dtype=np.float64)
    if e.ndim != 3 or e.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("Expected a stack of world-to-camera matrices")
    m = np.repeat(np.eye(4)[None], len(e), axis=0)
    m[:, :3] = e[:, :3]
    return m


def increment(x):
    from scipy.spatial.transform import Rotation
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(x[:3] * ROTATION_UNIT).as_matrix()
    delta[:3, 3] = x[3:] * TRANSLATION_UNIT
    return delta


def edge(poses, k, i, j, xy, target):
    return (poses[j] @ np.linalg.inv(poses[i]), np.linalg.inv(k[i]), np.linalg.inv(k[j]),
            np.column_stack((xy, np.ones(len(xy)))), np.column_stack((target, np.ones(len(target)))))


def signed_distances(transform, edges):
    inverse = np.linalg.inv(transform)
    distances = []
    for relative, ki, kj, source, target in edges:
        motion = transform @ relative @ inverse
        x, y, z = motion[:3, 3]
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        f = kj.T @ skew @ motion[:3, :3] @ ki
        left, right = source @ f.T, target @ f
        den = np.sqrt(np.sum(left[:, :2]**2 + right[:, :2]**2, axis=1))
        if np.any(den < 1e-12):
            raise ValueError("Degenerate epipolar geometry")
        distances.append(np.sum(left * target, axis=1) / den)
    return distances


def fit(physical, edges):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    def measurement(x):
        # One vote per pair. Soft L1 with a 2 px transition applied BEFORE pair weighting.
        distances = signed_distances(increment(x) @ physical, edges)
        return np.concatenate([r / np.sqrt(len(r)) * np.sqrt(2 / (1 + np.hypot(1., r/2)))
                               for r in distances])

    def residual(x):
        return np.concatenate((measurement(x), x))

    solution = least_squares(residual, np.zeros(6), bounds=(-3., 3.), max_nfev=200)
    if not solution.success or not np.isfinite(solution.fun).all():
        raise RuntimeError(solution.message)
    delta = increment(solution.x)
    transform = delta @ physical
    singular = np.linalg.svd(solution.jac[:-6], compute_uv=False)
    return transform, dict(delta_rotation_deg=float(Rotation.from_matrix(delta[:3,:3]).magnitude()*180/np.pi),
                           delta_translation_m=float(np.linalg.norm(delta[:3,3])),
                           delta_standard_units=solution.x.tolist(), bound_active=solution.active_mask.tolist(),
                           robust_measurement_jacobian_singular_values=singular.tolist(),
                           training_edges=len(edges), evaluations=solution.nfev,
                           objective_initial=float(np.sum(residual(np.zeros(6))**2)),
                           objective_final=float(np.sum(solution.fun**2)))
