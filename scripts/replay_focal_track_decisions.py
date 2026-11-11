"""Check fold isolation and recover epipolar decisions from the published raw matches."""
import argparse
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np

from focal_zoom_gradient import LEVELS, noisy_poses
from focal_zoom_observations import PUBLIC, SCALE, UPSTREAM
from score_saved_geometry import read
from visual_focal_baseline import keys


def load_folds(z, meta):
    blob = z.read(keys(meta["scene"], meta["batch"]))
    if hashlib.sha256(blob).hexdigest() != meta["matches_sha256"]:
        raise ValueError("Match archive changed")
    nodes = [set(), set()]
    pairs = [[], []]
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        for n, (i, j) in enumerate(meta["pair_order"]):
            a, b, fold = (data[f"{n}_{k}"] for k in ("xy_i", "xy_j", "fold"))
            if a.shape != b.shape or a.shape != (fold.size, 2) or not np.isin(fold, [0, 1]).all():
                raise ValueError("Invalid match dimensions or fold")
            for f in (0, 1):
                mask = fold == f
                x, y = a[mask], b[mask]
                for view, points in ((i, x), (j, y)):
                    nodes[f].update((view, int(u), int(v)) for u, v in np.rint(2 * points))
                if len(x) >= 6:
                    pairs[f].append((i, j, x, y))
    if not all(nodes) or nodes[0] & nodes[1]:
        raise ValueError("Connected image features cross folds, or a fold is empty")
    return pairs


def independent_gain(pairs, poses, observed, proposed, view):
    pair_medians = []
    for i, j, a, b in pairs:
        if view not in (i, j):
            continue
        other, first, second = (j, a, b) if i == view else (i, b, a)
        take = np.linspace(0, len(first) - 1, min(64, len(first)), dtype=int)
        first = np.column_stack((first[take], np.ones(len(take))))
        second = np.column_stack((second[take], np.ones(len(take))))
        motion = poses[other] @ np.linalg.inv(poses[view])
        # Apply the cross product columnwise to form E = [t]_x R.
        essential = np.cross(motion[:3, 3], motion[:3, :3].T).T
        values = []
        for target in (observed[view], proposed):
            fundamental = np.linalg.inv(observed[other]).T @ essential @ np.linalg.inv(target)
            left, right = second @ fundamental, first @ fundamental.T
            numerator = np.abs(np.sum(right * second, axis=1))
            denominator = np.sqrt((left[:, :2] ** 2 + right[:, :2] ** 2).sum(axis=1))
            if np.any(denominator < 1e-14):
                raise ValueError("Degenerate epipolar line")
            values.append(float(np.median(numerator / denominator)))
        pair_medians.append(values)
    if len(pair_medians) < 2:
        return None
    before, after = np.median(pair_medians, axis=0)
    return float(before - after)


def replay(archives):
    primary = read(PUBLIC / "summary.json")
    obs = read(PUBLIC / "observations.json")
    if len(obs) != 33:
        raise ValueError("Incomplete set of redetected images")
    with zipfile.ZipFile(PUBLIC / "redetected_tracks.zip") as z:
        folds = {(o["camera"]["scene"], o["camera"]["batch"]): load_folds(z, o["tracks"]) for o in obs}
    if len(folds) != 33:
        raise ValueError("Duplicate image batch")
    print("Verified disjoint matched feature nodes in both track folds for every one of the 33 batches.")
    if archives is None:
        return
    sys.path.insert(0, str(UPSTREAM))
    from audit_masked_camera_gradients import inputs
    clean = {s: (poses, k) for s, _, poses, k, _ in inputs(archives)}
    levels = {(name, seed): (angle, shift) for name, seed, angle, shift in LEVELS}
    maximum = 0.
    count = 0
    for row in primary["rows"]:
        scene, tag = row["scene"], row["batch"]
        recorded = row["fold_gains"]
        if row["case"] in ("hidden_zoom", "uniform_clean"):
            if recorded != [None, None] or row["edits"]["both_folds"]:
                raise ValueError("An invisible focal change triggered the outlier method")
            continue
        e, k = clean[scene]
        poses = noisy_poses(e, scene, row["seed"], *levels[row["noise"], row["seed"]])
        view = row["view"]
        observed = k.copy()
        observed[:, [0, 1], [0, 1]] *= SCALE
        proposed = observed[view].copy()
        observed[view, [0, 1], [0, 1]] *= SCALE ** row["direction"]
        gains = [independent_gain(side, poses, observed, proposed, view) for side in folds[scene, tag]]
        for actual, expected in zip(gains, recorded):
            if (actual is None) != (expected is None):
                raise ValueError("Different support for focal decision")
            if actual is not None:
                maximum = max(maximum, abs(actual - expected))
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
        if all(g is not None and g >= .25 for g in gains) != row["edits"]["both_folds"]:
            raise ValueError("Different two-fold focal decision")
        count += 1
    if count != 300 or len(primary["rows"]) != 465:
        raise ValueError("Incomplete pose perturbation replay")
    print(f"Recomputed both support decisions in {count} visible cases. Maximum gain discrepancy {maximum:.3g} px.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path)
    replay(parser.parse_args().archives)
