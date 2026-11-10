"""Offline RGB-track diagnostic using public reference geometry after prediction sealing."""
import argparse
from pathlib import Path

import numpy as np

from evaluation_stability import dump, sha
from fixed_match_validation import sample_points
from score_saved_geometry import read

ROOT = Path(__file__).resolve().parents[1]


def measure(arrays, source, target, xy, matched):
    valid = sample_points(arrays["valid_mask"][source, ..., None].astype(float), xy)[:, 0] >= 1 - 1e-8
    xyz = sample_points(arrays["reference_points"][source], xy[valid])
    e = arrays["extrinsics"][target]
    # Exact inverse of the released loader's transposed camera-to-world transform.
    camera = np.linalg.solve(e[:, :3].T, (xyz + e[:, :3].T @ e[:, 3]).T).T
    uvz = camera @ arrays["intrinsics"][target].T
    positive = uvz[:, 2] > 1e-8
    error = np.full(len(uvz), np.inf)
    error[positive] = np.linalg.norm(uvz[positive, :2] / uvz[positive, 2:3] - matched[valid][positive], axis=1)
    return error


def statistics(errors):
    finite = errors[np.isfinite(errors)]
    return dict(valid_source_depth=len(errors), positive_target_depth=len(finite),
                within_5px=int(np.count_nonzero(errors <= 5)), within_10px=int(np.count_nonzero(errors <= 10)),
                positive_depth_median_px=float(np.median(finite)) if len(finite) else None)


def main():
    from repeatability_capture import array_digest
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    args = parser.parse_args()
    pairs, scenes = [], {}
    for scene in read(ROOT / "results/learned_support_v8/support.json")["scenes"]:
        name = scene["scene"]
        rid = "repeat6_" + name.replace("/", "_") + "_both_released_seed42_r0"
        run = read(args.evidence / "runs" / rid / "run.json")
        arrays = dict(np.load(args.evidence / run["evaluation_inputs_relative"], allow_pickle=False))
        if array_digest(arrays) != run["evaluation_tensor_sha256"]:
            raise ValueError("Reference changed")
        scene_errors = []
        for pair in scene["pairs"]:
            path = args.support / pair["file"]
            if sha(path) != pair["sha256"]:
                raise ValueError("RGB support changed")
            data = dict(np.load(path, allow_pickle=False))
            for a, b in (("i", "j"), ("j", "i")):
                xy, matched = data["xy_" + a], data["xy_" + b]
                same = measure(arrays, pair[a], pair[a], xy, xy)
                # Bilinear interpolation of 3D coordinates across depths can move a subpixel observation.
                errors = measure(arrays, pair[a], pair[b], xy, matched)
                pairs.append(dict(scene=name, source=pair[a], target=pair[b], input_matches=len(xy),
                                  self_projection=statistics(same), cross_projection=statistics(errors)))
                scene_errors.append(errors)
        scenes[name] = statistics(np.concatenate(scene_errors))
    out = ROOT / "results/track_camera_selection_v13/track_diagnostic.json"
    if out.exists():
        raise FileExistsError(out)
    dump(out, dict(seal_sha256=sha(out.parent / "seal.json"), support_sha256=sha(ROOT / "results/learned_support_v8/support.json"),
                   script_sha256=sha(Path(__file__)), reference_geometry_used_for_offline_diagnosis=True,
                   scenes=scenes, directed_pairs=pairs))


if __name__ == "__main__":
    main()
