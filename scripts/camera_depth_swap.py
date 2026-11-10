"""Express camera rigs and depth maps in a common first-camera coordinate frame."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Rig:
    rotation: np.ndarray       # first-camera coordinates to each camera
    center: np.ndarray         # centers in first-camera coordinates, in depth units
    intrinsic: np.ndarray


def relative_rig(extrinsic, intrinsic, scale=1.0):
    e = np.asarray(extrinsic, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    if e.ndim != 3 or e.shape[1:] not in ((3, 4), (4, 4)) or k.shape != (len(e), 3, 3):
        raise ValueError("Each view needs one extrinsic and one intrinsic")
    if not (np.isfinite(e).all() and np.isfinite(k).all() and np.isfinite(scale) and scale > 0):
        raise ValueError("Invalid camera or depth scale")
    raw = e[:, :3, :3]
    if np.any(np.linalg.det(raw) <= 0) or np.max(np.abs(raw @ raw.transpose(0, 2, 1) - np.eye(3))) > .01:
        raise ValueError("Invalid rotation convention")
    centers = -np.linalg.solve(raw, e[:, :3, 3, None])[..., 0]
    u, _, vt = np.linalg.svd(raw)
    rotations = u @ vt
    return Rig(rotations @ rotations[0].T, (centers - centers[0]) @ rotations[0].T / scale, k)


def median_baseline(rig):
    i, j = np.triu_indices(len(rig.center), 1)
    length = float(np.median(np.linalg.norm(rig.center[i] - rig.center[j], axis=1)))
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Translation scale is unobservable for this trajectory")
    return length


def unproject(rig, depth):
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 3 or len(depth) != len(rig.center) or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Expected one finite positive depth map per camera")
    _, height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    result = np.empty((*depth.shape, 3), dtype=np.float32)
    for frame, (r, c, k) in enumerate(zip(rig.rotation, rig.center, rig.intrinsic)):
        rays = pixels @ np.linalg.inv(k).T
        rays /= rays[..., 2:3]
        result[frame] = (rays * depth[frame, ..., None]) @ r + c
    if not np.isfinite(result).all():
        raise ValueError("Invalid world points")
    return result
