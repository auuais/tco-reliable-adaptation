"""Audit shared, gained and lost matches after the crossover predictions are sealed."""
import itertools
from statistics import mean

import numpy as np
import torch

from evaluation_stability import dump
from mast3r_crossover_cache import directed_paths
from plan_mast3r_crossover import ROOT, PARENT, MANIFEST, read, require, sha256, text_sha256
from prepare_dense_focal_pilot import write_arrays


PUBLIC = ROOT / "results/mast3r_geometry_matching_v1"
RAW = ROOT / "reports/mast3r_neural_transfer/cache"
REASONS = ("visible", "invalid_source_pixel", "invalid_matched_target_pixel", "invalid_source_depth",
           "behind_target_camera", "projection_outside_target", "invalid_projected_target_depth", "depth_disagreement")


def project(points, pose, intrinsic):
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, camera[:, 2]


def match_errors(matches, points, masks, poses, intrinsic, first, second, native_hw=(384, 512)):
    """Classify visibility and native-pixel reprojection error using fixed references."""
    height, width = native_hw
    ref_height, ref_width = masks.shape[1:]
    xy1, xy2 = matches[:, :2], matches[:, 2:]
    factor = np.array([ref_width / width, ref_height / height])

    def indices(xy):
        safe = np.nan_to_num(xy, nan=-1., posinf=-1., neginf=-1.)
        pixel = np.rint((safe + .5) * factor - .5).astype(int)
        inside = np.isfinite(xy).all(1) & ((xy >= 0) & (xy < [width, height])).all(1)
        pixel = np.clip(pixel, [0, 0], [ref_width - 1, ref_height - 1])
        return pixel[:, 0], pixel[:, 1], inside

    source_x, source_y, source_inside = indices(xy1)
    target_x, target_y, target_inside = indices(xy2)
    source_points = points[first, source_y, source_x]
    _, source_depth = project(source_points, poses[first], intrinsic)
    rays = np.column_stack([xy1, np.ones(len(xy1))]) @ np.linalg.inv(intrinsic).T
    camera_to_world = np.linalg.inv(poses[first])
    lifted = (rays * source_depth[:, None]) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    projected, projected_depth = project(lifted, poses[second], intrinsic)
    probe_x, probe_y, projected_inside = indices(projected)
    _, target_depth = project(points[second, probe_y, probe_x], poses[second], intrinsic)
    reason = np.zeros(len(matches), dtype=np.uint8)
    conditions = (source_inside & masks[first, source_y, source_x],
                  target_inside & masks[second, target_y, target_x],
                  np.isfinite(source_depth) & (source_depth > 0),
                  np.isfinite(projected_depth) & (projected_depth > 0), projected_inside,
                  masks[second, probe_y, probe_x] & np.isfinite(target_depth) & (target_depth > 0),
                  np.abs(projected_depth - target_depth) <= np.maximum(.02, .02 * target_depth))
    for code, condition in enumerate(conditions, 1):
        reason[(reason == 0) & ~condition] = code
    errors = np.linalg.norm(projected - xy2, axis=1)
    errors[reason != 0] = 0.
    return errors, reason


def describe(matches, errors, reason):
    count = len(matches)
    visible = reason == 0
    valid = (reason != 1) & (reason != 2)
    cells = np.floor(matches[:, :2] * [8/512, 6/384]).astype(int)
    occupied = {tuple(p) for p in cells[visible]}
    return dict(count=count, valid_fraction=float(valid.mean()) if count else None,
                visible_count=int(visible.sum()), spatial_coverage=len(occupied) / 48,
                median_error_px=float(np.median(errors[visible])) if visible.any() else None,
                fraction_within_2px=float((errors[visible] <= 2).mean()) if visible.any() else None,
                excluded={name: int((reason == code).sum()) for code, name in enumerate(REASONS) if code})


def aggregate(records):
    result = {}
    for group in ("shared", "gained", "lost"):
        unordered = []
        for first, second in itertools.combinations(range(5), 2):
            pair = [r[group] for r in records if {r["first"], r["second"]} == {first, second}]
            unordered.append({field: mean(values) if values else None for field in
                              ("count", "valid_fraction", "visible_count", "spatial_coverage", "median_error_px", "fraction_within_2px")
                              for values in [[v[field] for v in pair if v[field] is not None]]})
        result[group] = {field: mean(values) if values else None for field in unordered[0]
                         for values in [[p[field] for p in unordered if p[field] is not None]]}
    return result


def run():
    seal = read(PUBLIC / "predictions.json")
    require(len(seal["rows"]) == 48 and seal["manifest_sha256"] == sha256(MANIFEST), "Predictions must be sealed first")
    plan = read(MANIFEST)
    inputs = read(ROOT / "manifests/dense_focal_inputs.json")
    report = dict(seal_sha256=sha256(PUBLIC / "predictions.json"), states={}, same_view_checks={},
                  source_sha256=text_sha256(ROOT / "scripts/audit_mast3r_crossover_matches.py"))
    (PUBLIC / "match_audit").mkdir(exist_ok=True)
    for scene in plan["scenes"]:
        key = scene + "_base"
        reference = ROOT / "reports/dense_focal_pilot/evaluation" / inputs["references"][key]["file"]
        require(sha256(reference) == inputs["references"][key]["sha256"], "Reference changed")
        with np.load(reference) as data:
            points, masks = data["registered_rgb"].astype(float), data["valid_mask"]
        with np.load(PARENT / "pose_packets.npz") as data:
            poses = data[scene + "_clean"]
        intrinsic = np.asarray(plan["known_K"])
        yy, xx = np.indices(masks.shape[1:])
        expected = (np.column_stack([xx.ravel(), yy.ravel()]) + .5) * [512/masks.shape[2], 384/masks.shape[1]] - .5
        self_errors = []
        for view in range(5):
            projected, depth = project(points[view].reshape(-1, 3), poses[view], intrinsic)
            valid = masks[view].ravel()
            require((depth[valid] > 0).all(), "Nonpositive valid reference depth")
            self_errors.append(float(np.max(np.linalg.norm(projected[valid] - expected[valid], axis=1))))
        require(max(self_errors) < .01, "Reference projection convention failed")
        report["same_view_checks"][scene] = dict(max_native_pixel_error=self_errors, reference_sha256=sha256(reference))
        rgb_hash = read(PARENT / "plan.json")["rgb_sha256"][scene]
        names = [f"{key}_{rgb_hash[:12]}_{index}" for index in range(5)]
        for grid in ("centered", "shifted"):
            state = f"{scene}_{grid}_D"
            records, arrays = [], {}
            for (first, second), (_, matching) in zip(itertools.permutations(range(5), 2), directed_paths(names)):
                old = torch.load(RAW / (scene + "_zero") / matching, map_location="cpu", weights_only=False)[1]
                new = torch.load(RAW / state / matching, map_location="cpu", weights_only=False)[1]
                a = {tuple(row) for row in np.column_stack([np.asarray(old[0]), np.asarray(old[1])]).astype(np.int32)}
                b = {tuple(row) for row in np.column_stack([np.asarray(new[0]), np.asarray(new[1])]).astype(np.int32)}
                record = dict(first=first, second=second, source_match_files={"pretrained": sha256(RAW / (scene + "_zero") / matching),
                                                                           "adapted": sha256(RAW / state / matching)})
                for group, subset in (("shared", a & b), ("gained", b-a), ("lost", a-b)):
                    matches = np.array(sorted(subset), dtype=np.int32).reshape(-1, 4)
                    errors, reason = match_errors(matches, points, masks, poses, intrinsic, first, second)
                    record[group] = describe(matches, errors, reason)
                    prefix = f"{first}_{second}_{group}"
                    arrays.update({prefix + "_matches": matches, prefix + "_errors": errors, prefix + "_reason": reason})
                records.append(record)
            path = PUBLIC / "match_audit" / (state + ".npz")
            write_arrays(path, arrays)
            report["states"][state] = dict(records=records, unordered_pair_then_scene=aggregate(records), archive_sha256=sha256(path))
    dump(PUBLIC / "match_audit.json", report)
    print("Audited six states, both directions of ten pairs, and reference projection checks.")


if __name__ == "__main__":
    run()
