"""Check the alternative public RGB SfM poses on the fifteen development images."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from audit_sensor_frame import SCENES, calibration_matrix, effective_camera, stats, track_errors
from calibration_audit import resized_k
from camera_depth_swap import Rig, median_baseline, relative_rig
from evaluation_stability import dump, sha
from fixed_match_validation import sample_points
from score_saved_geometry import read

ROOT = Path(__file__).resolve().parents[1]
REVISION = "22b7c48a0e55b49d0214a8dbb022426f518e2795"
PUBLIC = ROOT / "results/sensor_frame_v14"
FRAMES = (0, 200, 400, 600, 800)


def fetch(output):
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for scene in SCENES:
        name, sequence = scene.split("/")
        keys = [f"{sequence}/frame-{n:06d}.color.png" for n in FRAMES]
        url = f"https://raw.githubusercontent.com/tsattler/visloc_pseudo_gt_limitations/{REVISION}/pgt/sfm/7scenes/{name}_test.txt"
        with urlopen(url, timeout=30) as response:
            blob = response.read()
        subset = {}
        for line in blob.decode().splitlines():
            fields = line.split()
            if fields and fields[0] in keys:
                if fields[0] in subset or len(fields) != 9:
                    raise ValueError("Duplicate or invalid pose row")
                subset[fields[0]] = [float(x) for x in fields[1:]]
        records[scene] = dict(url=url, file_sha256=hashlib.sha256(blob).hexdigest(), keys=keys, poses=subset)
    dump(output / "fifteen_frames.json", records)


def rig_as_extrinsic(rig):
    result = np.repeat(np.eye(4)[None], len(rig.center), axis=0)
    result[:, :3, :3] = rig.rotation
    result[:, :3, 3] = -np.einsum("nij,nj->ni", rig.rotation, rig.center)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["fetch", "score"])
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--evidence", type=Path)
    p.add_argument("--support", type=Path)
    p.add_argument("--protocol", type=Path)
    args = p.parse_args()
    if args.action == "fetch":
        fetch(args.subset)
        return
    from repeatability_capture import array_digest
    from scipy.spatial.transform import Rotation

    records = read(args.subset / "fifteen_frames.json")
    support_meta = {s["scene"]: s for s in read(ROOT / "results/learned_support_v8/support.json")["scenes"]}
    transform = calibration_matrix(args.protocol)
    if not np.array_equal(transform, np.array(read(ROOT / "manifests/sensor_frame_v14.json")["calibration"])):
        raise ValueError("Registration calibration changed")
    rows = []
    for scene in SCENES:
        subset = records[scene]
        missing = [k for k in subset["keys"] if k not in subset["poses"]]
        if missing:
            rows.append(dict(scene=scene, missing=missing))
            continue
        fields = np.array([subset["poses"][key] for key in subset["keys"]])
        quaternion = fields[:, [1, 2, 3, 0]]
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-5):
            raise ValueError("Invalid public quaternion")
        sfm = np.repeat(np.eye(4)[None], len(fields), axis=0)
        sfm[:, :3, :3] = Rotation.from_quat(quaternion).as_matrix()
        sfm[:, :3, 3] = fields[:, 4:7]
        rid = "repeat6_" + scene.replace("/", "_") + "_both_released_seed42_r0"
        run = read(args.evidence / "runs" / rid / "run.json")
        arrays = dict(np.load(args.evidence / run["evaluation_inputs_relative"], allow_pickle=False))
        if array_digest(arrays) != run["evaluation_tensor_sha256"]:
            raise ValueError("Reference tensor mismatch")
        original = [effective_camera(e) for e in arrays["extrinsics"]]
        depths = [points @ e[2,:3] + e[2,3] for points, e in zip(arrays["reference_points"], original)]
        height, width = depths[0].shape
        # Public pose text supplies focal length. The evaluation convention centers the principal point.
        k = np.stack([resized_k(focal, width, height) for focal in fields[:, 7]])
        rig = relative_rig(sfm, k)
        metric_rig = relative_rig(np.stack([transform @ e for e in original]), k)
        scale = median_baseline(metric_rig) / median_baseline(rig)
        matched = Rig(rig.rotation, rig.center * scale, rig.intrinsic)
        extrinsic = rig_as_extrinsic(matched)
        sampson, errors = [], []
        for pair in support_meta[scene]["pairs"]:
            path = args.support / pair["file"]
            if sha(path) != pair["sha256"]:
                raise ValueError("Image correspondence file changed")
            data = dict(np.load(path, allow_pickle=False))
            for a, b in (("i","j"), ("j","i")):
                i, j = pair[a], pair[b]
                xy, q = data["xy_" + a], data["xy_" + b]
                valid = sample_points(arrays["valid_mask"][i,...,None].astype(float), xy)[:,0] >= 1-1e-8
                depth = sample_points(depths[i][...,None], xy)[:,0]
                sd, re = track_errors(xy, q, valid, depth, k[i], k[j], extrinsic[j] @ np.linalg.inv(extrinsic[i]))
                sampson.append(sd)
                errors.append(re)
        rows.append(dict(scene=scene, source_file_sha256=subset["file_sha256"], frames_used=len(fields),
                         focal_original_px=fields[:,7].tolist(), translation_scale=scale,
                         **stats(np.concatenate(sampson), np.concatenate(errors))))
    out = PUBLIC / "alternative_rgb_sfm.json"
    if out.exists():
        raise FileExistsError(out)
    dump(out, dict(revision=REVISION, subset_sha256=sha(args.subset / "fifteen_frames.json"),
                   source_sha256=sha(Path(__file__)), rows=rows,
                   note="Offline descriptive sensitivity on the five development images per scene. SfM uses RGB imagery from the same scene and potentially shared image structure. Median-baseline normalization uses the registered-pose convention. Epipolar error is invariant to that scale."))


if __name__ == "__main__":
    main()
