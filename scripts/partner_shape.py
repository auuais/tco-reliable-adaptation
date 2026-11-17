"""Within-cell partner-disagreement modes and the registered conditional fit."""
import numpy as np
from scipy.optimize import minimize


def native_rows(rows):
    """Keep the nearest measurement per native pixel after center-preserving resize."""
    pixels = np.rint((rows[:, :2] + .5) * [512/518, 384/392] - .5).astype(int)
    nearest = {}
    for i in np.argsort(rows[:, 2], kind="stable"):
        x, y = pixels[i]
        if 0 <= x < 512 and 0 <= y < 384:
            nearest.setdefault((x, y), i)
    indices = list(nearest.values())
    return pixels[indices], rows[indices, 2]


def tiles(a):
    h, w = a.shape[-2:]
    return a.reshape(*a.shape[:-2], h//8, 8, w//8, 8).swapaxes(-3, -2).reshape(*a.shape[:-2], -1, 64)


def images(blocks, h, w):
    return blocks.reshape(*blocks.shape[:-2], h//8, w//8, 8, 8).swapaxes(-3, -2).reshape(*blocks.shape[:-2], h, w)


def remove_relief(blocks, relief):
    blocks = blocks - blocks[..., 36:37]
    energy = np.sum(relief**2, axis=-1, keepdims=True)
    dot = np.sum(blocks * relief, axis=-1, keepdims=True)
    scale = np.divide(dot, energy, out=np.zeros_like(dot), where=energy > 1e-20)
    return blocks - scale * relief


def make_modes(single, fused, seed):
    """Use the four-image Gram matrix to avoid a full image-sized covariance."""
    if single.shape != (4, *fused.shape) or not np.isfinite(single).all() or not np.isfinite(fused).all():
        raise ValueError("Expected four finite partner ratio images")
    h, w = fused.shape
    relief = tiles(fused.astype(float))
    relief = relief - relief[:, 36:37]
    centered = single.astype(float) - single.mean(axis=0, dtype=float)
    x = remove_relief(tiles(centered), relief).reshape(4, -1)
    eig, vectors = np.linalg.eigh(x @ x.T)
    basis = (vectors[:, -1:-3:-1].T @ x) / 2.
    # A canonical sign makes coefficients easy to compare across implementations.
    for mode in basis:
        if mode[np.argmax(np.abs(mode))] < 0:
            mode *= -1
    real = basis.reshape(2, -1, 64)
    permutation = np.random.default_rng(seed).permutation(real.shape[1])
    shuffled = remove_relief(real[:, permutation], relief).reshape(2, -1)
    for j in range(2):
        if j and np.dot(shuffled[0], shuffled[0]) > 1e-20:
            shuffled[j] -= shuffled[0] * (np.dot(shuffled[j], shuffled[0]) / np.dot(shuffled[0], shuffled[0]))
        wanted, observed = np.linalg.norm(basis[j]), np.linalg.norm(shuffled[j])
        if wanted > 1e-8 and observed < 1e-12:
            raise ValueError("Could not construct magnitude-matched shuffled control")
        shuffled[j] *= wanted / observed if observed > 0 else 0.
    total = np.maximum(eig, 0).sum()
    return (images(real, h, w), images(shuffled.reshape(2, -1, 64), h, w),
            np.maximum(eig[-1:-3:-1], 0)/total if total > 0 else np.zeros(2))


def fit(base, design, measurement):
    unit = .02 * np.median(measurement)
    if unit <= 0 or not np.isfinite(unit):
        raise ValueError("Invalid sparse measurement unit")
    x = design / unit
    y = (measurement-base) / unit
    singular = np.linalg.svd(x, compute_uv=False)
    if len(y) < 8 or singular[0] < 1e-8 or singular[-1] < .001 * singular[0]:
        return np.zeros(2), singular, True

    def objective(a):
        error = x @ a - y
        norm = np.hypot(1., error)
        value = np.mean(norm-1.) + .01 * (a @ a)
        grad = x.T @ (error/norm) / len(error) + .02*a
        return value, grad

    result = minimize(objective, np.zeros(2), jac=True, bounds=[(-1., 1.)]*2, method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 200})
    def optimal(a):
        gradient = objective(a)[1]
        projected = a - np.clip(a-gradient, -1., 1.)
        return np.isfinite(a).all() and np.max(np.abs(projected)) <= 2e-7

    if not optimal(result.x):
        result = minimize(objective, result.x, jac=True, bounds=[(-1., 1.)]*2, method="SLSQP",
                          options={"ftol": 1e-14, "maxiter": 200})
    if not optimal(result.x):
        raise ValueError(f"Shape fit failed first-order optimality: {result.message}")
    return result.x, singular, False
