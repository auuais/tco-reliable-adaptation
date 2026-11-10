"""Test the registered RGB depth's camera-frame convention on fixed image correspondences."""
import argparse
import ast
import hashlib
from pathlib import Path
import zipfile

import numpy as np

from calibration_variants import calibration_prior
from evaluation_stability import dump, sha
from fixed_match_validation import sample_points
from score_saved_geometry import read

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/sensor_frame_v14"
MANIFEST = ROOT / "manifests/sensor_frame_v14.json"
SCENES = ("heads/seq-01", "chess/seq-03", "fire/seq-03")
K_VARIANTS = ("released", "axis_nominal", "axis_registration")
FRAMES = ("original_depth_pose", "compose_depth_to_rgb", "inverse_direction_control")


def calibration_matrix(path):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "d_to_rgb" for t in node.targets):
            data = np.array(ast.literal_eval(node.value.args[0]), dtype=np.float64)
            if data.shape != (4, 4) or not np.allclose(data[3], [0, 0, 0, 1]) or np.linalg.det(data[:3, :3]) <= 0:
                raise ValueError("Invalid depth-to-RGB calibration")
            return data
    raise ValueError("Registration transform not found")


def effective_camera(extrinsic):
    # Invert the same transposed camera-to-world matrix that generated v6 references.
    c2w = np.eye(4)
    r, t = extrinsic[:3, :3], extrinsic[:3, 3]
    c2w[:3, :3], c2w[:3, 3] = r.T, -r.T @ t
    return np.linalg.inv(c2w)


def track_errors(xy, target, valid, depth, k1, k2, relative):
    p = np.column_stack((xy, np.ones(len(xy))))
    q = np.column_stack((target, np.ones(len(target))))
    r, t = relative[:3, :3], relative[:3, 3]
    x, y, z = t
    cross_t = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    f = np.linalg.inv(k2).T @ cross_t @ r @ np.linalg.inv(k1)
    left, right = p @ f.T, q @ f
    denominator = np.linalg.norm(np.column_stack((left[:, :2], right[:, :2])), axis=1)
    if np.any(denominator <= 1e-15):
        raise ValueError("Degenerate epipolar geometry")
    sampson = np.abs(np.sum(q * left, axis=1)) / denominator
    cam = ((p[valid] @ np.linalg.inv(k1).T) * depth[valid, None]) @ r.T + t
    uvz = cam @ k2.T
    positive = uvz[:, 2] > 1e-8
    error = np.full(len(uvz), np.inf)
    error[positive] = np.linalg.norm(uvz[positive, :2] / uvz[positive, 2:3] - target[valid][positive], axis=1)
    return sampson, error


def stats(sampson, reprojection):
    finite = reprojection[np.isfinite(reprojection)]
    return dict(track_observations=len(sampson), sampson_median_px=float(np.median(sampson)),
                valid_depth=len(reprojection), positive_depth=len(finite), within_five_pixels=int(sum(reprojection<=5)),
                positive_depth_reprojection_median_px=float(np.median(finite)) if len(finite) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--archives", type=Path)
    p.add_argument("--prepared", type=Path)
    p.add_argument("--evidence", type=Path)
    p.add_argument("--support", type=Path)
    args = p.parse_args()
    calibration = calibration_matrix(args.protocol)
    plan = dict(scenes=list(SCENES), intrinsics=list(K_VARIANTS), camera_conventions=list(FRAMES),
                calibration=calibration.tolist(), calibration_source_sha256=sha(args.protocol),
                matcher_support_sha256=sha(ROOT / "results/learned_support_v8/support.json"),
                source_sha256=sha(Path(__file__)), frames=[0, 200, 400, 600, 800],
                analysis="Development diagnostic after v13 predictions were sealed. Verify original camera files, recover registered depth, apply the same rigid transform used by preprocessing to the world-to-camera matrix. Cross three K conventions and three frame conventions, including an inverse direction control. Report Sampson distance and registered-depth cross-view reprojection. Bilinear interpolated axial depth, all four source depth neighbors valid. All directed observations retained. No fitted calibration or replacement of v6 or v13 metrics.")
    if args.freeze:
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        dump(MANIFEST, plan)
        return
    if read(MANIFEST) != plan:
        raise ValueError("Audit differs from frozen design")
    from repeatability_capture import array_digest

    support = {s["scene"]: s for s in read(ROOT / "results/learned_support_v8/support.json")["scenes"]}
    rows, identities, pairs_detail = [], [], []
    for scene in SCENES:
        name, seq = scene.split("/")
        rid = "repeat6_" + scene.replace("/", "_") + "_both_released_seed42_r0"
        run = read(args.evidence / "runs" / rid / "run.json")
        arrays = dict(np.load(args.evidence / run["evaluation_inputs_relative"], allow_pickle=False))
        if array_digest(arrays) != run["evaluation_tensor_sha256"]:
            raise ValueError("Reference changed")
        cameras = [effective_camera(e) for e in arrays["extrinsics"]]
        depths = [points @ e[2, :3] + e[2, 3] for points, e in zip(arrays["reference_points"], cameras)]
        with zipfile.ZipFile(args.archives / (name + "-" + seq + ".zip")) as archive:
            for i, frame in enumerate(plan["frames"]):
                filename = f"frame-{frame:06d}.pose.txt"
                data = archive.read(seq + "/" + filename)
                if data != (args.prepared / scene / filename).read_bytes():
                    raise ValueError("Prepared pose differs from archive")
                original = np.loadtxt(data.decode().splitlines())
                expected = np.column_stack((original[:3,:3].T, -original[:3,:3].T @ original[:3,3]))
                if not np.allclose(expected, arrays["extrinsics"][i], atol=1e-7, rtol=0):
                    raise ValueError("Captured pose differs from the loader convention")
                identities.append(dict(scene=scene, frame=frame, pose_sha256=hashlib.sha256(data).hexdigest()))
        observations = []
        for pair in support[scene]["pairs"]:
            path = args.support / pair["file"]
            if sha(path) != pair["sha256"]:
                raise ValueError("Match hash mismatch")
            data = dict(np.load(path, allow_pickle=False))
            for a, b in (("i", "j"), ("j", "i")):
                i, j = pair[a], pair[b]
                xy, target = data["xy_"+a], data["xy_"+b]
                valid = sample_points(arrays["valid_mask"][i, ..., None].astype(float), xy)[:, 0] >= 1-1e-8
                depth = sample_points(depths[i][..., None], xy)[:, 0]
                observations.append((i, j, xy, target, valid, depth))
        for k_name in K_VARIANTS:
            k = calibration_prior(arrays["intrinsics"], depths[0].shape[1], depths[0].shape[0], k_name)
            for label, transform in zip(FRAMES, (np.eye(4), calibration, np.linalg.inv(calibration))):
                world_to_cam = [transform @ camera for camera in cameras]
                samples, errors = [], []
                for i, j, xy, target, valid, depth in observations:
                    relative = world_to_cam[j] @ np.linalg.inv(world_to_cam[i])
                    sd, re = track_errors(xy, target, valid, depth, k[i], k[j], relative)
                    samples.append(sd)
                    errors.append(re)
                    pairs_detail.append(dict(scene=scene, intrinsics=k_name, frame=label, i=i, j=j, **stats(sd,re)))
                rows.append(dict(scene=scene, intrinsics=k_name, frame=label,
                                 **stats(np.concatenate(samples), np.concatenate(errors))))
    if (PUBLIC / "summary.json").exists():
        raise FileExistsError(PUBLIC / "summary.json")
    dump(PUBLIC / "summary.json", dict(manifest_sha256=sha(MANIFEST), poses=identities, rows=rows,
                                      directed_pairs=pairs_detail, additional_cloud_usd=0))


if __name__ == "__main__":
    main()
