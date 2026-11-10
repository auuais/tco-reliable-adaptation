"""Use split RGB tracks to choose rotation, translation and intrinsics sources."""
from itertools import product

import numpy as np

from camera_depth_swap import Rig
from fixed_match_validation import sample_points


def bearings(rig, frame, pixels):
    hom = np.column_stack((pixels, np.ones(len(pixels))))
    local = hom @ np.linalg.inv(rig.intrinsic[frame]).T
    return (local / local[:, 2:3]) @ rig.rotation[frame]


def scaled_depth_observations(rig, depths, pairs):
    """Triangulate axial depth against each target bearing. Keep cheiral solutions."""
    edges = []
    for pair in pairs:
        for source, target in (("i", "j"), ("j", "i")):
            i, j = pair[source], pair[target]
            p, q = pair["xy_" + source], pair["xy_" + target]
            ray = bearings(rig, i, p)
            other = bearings(rig, j, q)
            unit = other / np.linalg.norm(other, axis=1, keepdims=True)
            transverse = ray - unit * np.sum(ray * unit, axis=1, keepdims=True)
            square = np.sum(transverse ** 2, axis=1)
            z = (transverse @ (rig.center[j] - rig.center[i])) / np.maximum(square, 1e-15)
            in_target = (rig.center[i] + z[:, None] * ray - rig.center[j]) @ rig.rotation[j, 2]
            valid = (square > np.sum(ray ** 2, axis=1) * np.sin(np.deg2rad(1.)) ** 2) & (z > 0) & (in_target > 0)
            predicted_z = sample_points(depths[i, ..., None], p)[:, 0]
            ratio = np.full(len(z), np.nan)
            ratio[valid] = z[valid] / predicted_z[valid]
            # Preproject the two terms so subsequent scale comparisons need only one division.
            a = ((predicted_z[:, None] * ray) @ rig.rotation[j].T) @ rig.intrinsic[j].T
            b = ((rig.center[i] - rig.center[j]) @ rig.rotation[j].T) @ rig.intrinsic[j].T
            edges.append((i, ratio, a, b, q, pair["fold"]))
    return edges


def fit_scales(edges, views, train_fold, shared=False):
    votes = [[] for _ in range(views)]
    for frame, ratio, _, _, _, fold in edges:
        use = np.isfinite(ratio)
        if train_fold is not None:
            use &= fold == train_fold
        if np.count_nonzero(use) >= 5:
            votes[frame].append(float(np.median(ratio[use])))
    # Each view-pair casts one vote. Missing or poorly conditioned views keep unit scale.
    scales = np.array([np.median(v) if v else 1. for v in votes])
    if shared:
        global_scale = np.median([s for s, v in zip(scales, votes) if v]) if any(votes) else 1.
        scales.fill(global_scale)
    return np.clip(scales, .5, 2.)


def validation_error(edges, scales, val_fold):
    medians = []
    for frame, _, a, b, target, fold in edges:
        use = fold == val_fold
        if np.count_nonzero(use) < 10:
            continue
        camera = a[use] * scales[frame] + b
        valid = camera[:, 2] > 1e-8
        error = np.full(len(camera), 1e4)
        error[valid] = np.linalg.norm(camera[valid, :2] / camera[valid, 2:3] - target[use][valid], axis=1)
        medians.append(float(np.median(np.minimum(error, 1e4))))
    if not medians:
        raise ValueError("No eligible validation edge")
    return float(np.mean(medians))


def select_geometry(initial, prior, depths, pairs):
    views = len(depths)
    solutions, cross_validation = {}, {}
    # The unmodified prediction competes directly with the camera combinations.
    for rotation, center, intrinsic in product(range(2), repeat=3):
        source = (initial, prior)
        rig = Rig(source[rotation].rotation, source[center].center, source[intrinsic].intrinsic)
        tag = f"{rotation}{center}{intrinsic}"
        edges = scaled_depth_observations(rig, depths, pairs)
        strategies = ["per_view"]
        if tag in ("000", "111"):
            strategies += ["unchanged", "shared"]
        for strategy in strategies:
            def scales(fold):
                if strategy == "unchanged":
                    return np.ones(views)
                return fit_scales(edges, views, fold, shared=strategy == "shared")
            name = tag + "_" + strategy
            errors = [validation_error(edges, scales(1 - fold), fold) for fold in (0, 1)]
            cross_validation[name] = errors
            solutions[name] = (rig, scales(None))
    winner = min(solutions, key=lambda name: (np.mean(cross_validation[name]),
                                               name != "000_unchanged", name))
    return winner, solutions, cross_validation
