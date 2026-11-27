"""Prepare one ARKitScenes raw visit with synchronized RGB, priors and evaluation-only LiDAR."""
import argparse
from bisect import bisect_left
import csv
import hashlib
import io
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from prepare_tco_final_ln_data import (
    DEFAULT_PACKETS, OUTPUT, WIDTH, apply_pose, array_signature, dense_reference,
    fixture, save_npz, sha_file, verify_fixture_and_packet, write_json,
)


REVISION = "474aaa83d7b1c9080619d8485de2d7c885c01fc5"
METADATA_SHA = "5a817c0ee8150cea1de4fa9d6399880eaa646c2b62326c3314e674d201536c57"
PUBLISHER = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
SPLIT_URL = ("https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/"
             + REVISION + "/raw/raw_train_val_splits.csv")
DATA_ROOT = Path("E:/ResearchData/TCO/data/arkitscenes/raw")
TRAINING_PAIR = (369706, 40777073)
ASSETS = ("lowres_wide.traj", "lowres_wide_intrinsics.zip", "lowres_depth.zip", "lowres_wide.zip")


def fetch_membership(split, visit, video):
    with urllib.request.urlopen(SPLIT_URL, timeout=30) as response:
        table = response.read()
    if hashlib.sha256(table).hexdigest() != METADATA_SHA:
        raise ValueError("Public split metadata hash mismatch")
    rows = list(csv.DictReader(io.StringIO(table.decode("utf-8-sig"))))
    found = [row for row in rows if row["video_id"] == str(video)]
    if len(found) != 1 or found[0]["visit_id"] != str(visit) or found[0]["fold"] != split:
        raise ValueError("Public split or visit/video association mismatch")
    if split == "Training" and (visit, video) != TRAINING_PAIR:
        raise ValueError("This study registered one separate Training bridge visit")


def acquire(root, split, video):
    directory = root / split / str(video)
    directory.mkdir(exist_ok=True, parents=True)
    sources = {}
    for asset in ASSETS:
        url = f"{PUBLISHER}/{split}/{video}/{asset}"
        req = urllib.request.Request(url + "?ln-size=1", headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            m = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if not m or len(response.read(2)) != 1:
                raise ValueError("Public asset did not supply its exact size")
            size = int(m.group(1))
        path = directory / asset
        if not path.is_file():
            partial = path.with_suffix(path.suffix + ".partial")
            subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "600", "-o", str(partial),
                            url + "?ln-full=1"], check=True)
            if partial.stat().st_size != size:
                raise ValueError(f"Incomplete download of {asset}")
            partial.replace(path)
        if path.stat().st_size != size or (asset.endswith(".zip") and not zipfile.is_zipfile(path)):
            raise ValueError(f"Invalid local publisher asset {asset}")
        sources[asset] = dict(bytes=size, sha256=sha_file(path))
    return directory, sources


def parse_trajectory(text):
    trajectory = {}
    max_difference = 0.
    for line in text.splitlines():
        values = np.fromstring(line, sep=" ")
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Invalid trajectory sample")
        # The pinned publisher's TrajStringToMatrix returns inv([R|t]).
        # It indexes poses by round(seconds, 3). Reject ambiguous duplicate keys.
        key = int(round(1000 * round(float(values[0]), 3)))
        if key in trajectory:
            raise ValueError("Timestamp collision in published trajectory")
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_rotvec(values[1:4]).as_matrix()
        w2c[:3, 3] = values[4:]
        independent, _ = cv2.Rodrigues(values[1:4])
        max_difference = max(max_difference, float(np.max(np.abs(independent-w2c[:3, :3]))))
        trajectory[key] = w2c
    if not trajectory or max_difference > 1e-10:
        raise ValueError("Independent angle-axis rotation check failed")
    return trajectory, max_difference


def frame_members(archive, video, suffix):
    pattern = re.compile(r"(?:^|/)" + str(video) + r"_(\d+\.\d{3})\." + suffix + r"$")
    frames = {}
    for name in archive.namelist():
        hit = pattern.search(name)
        if hit:
            key = int(hit.group(1).replace(".", ""))
            if key in frames:
                raise ValueError("Duplicate image/calibration timestamp")
            frames[key] = name
    if not frames:
        raise ValueError("Missing registered public asset")
    return frames


def associations(rgb, depth, intrinsics, poses):
    pose_times = sorted(poses)
    valid = {}
    for t in sorted(rgb.keys() & depth.keys()):
        calibration = next((x for x in (t, t-1, t+1) if x in intrinsics), None)
        pos = bisect_left(pose_times, t)
        nearest = min(pose_times[max(0, pos-1):pos+1], key=lambda x: (abs(x-t), x))
        if calibration is not None and abs(nearest-t) < 5:
            valid[t] = (calibration, nearest)
    return valid


def select_window(valid, poses):
    times = sorted(valid)
    centres = {t: np.linalg.inv(poses[p])[:3, 3] for t, (_, p) in valid.items()}
    for start in range(len(times)):
        for stride in (1, 2, 4, 8):
            last = start + 4 * stride
            if last >= len(times):
                continue
            span = times[start:last+1]
            if max(np.diff(span)) > 350:
                continue
            targets = times[start:last+1:stride]
            if len({valid[t][1] for t in targets}) != 5:
                continue
            distance = float(np.linalg.norm(centres[targets[-1]] - centres[targets[0]]))
            if .05 <= distance <= .75:
                return targets, stride, distance
    raise ValueError("No five-view window from the registered time/pose rules")


def adjacency(raw_surfaces, masks, cameras, ks):
    fractions = []
    for i in range(4):
        points = raw_surfaces[i][masks[i]][::7]
        target = apply_pose(points, cameras[i+1])
        target = target[target[:, 2] > 0]
        uvz = target @ ks[i+1].T
        uv = np.rint(uvz[:, :2] / uvz[:, 2:]).astype(int)
        inside = ((uv >= 0) & (uv < masks[i+1].shape[::-1])).all(axis=1)
        target, uv = target[inside], uv[inside]
        known = masks[i+1][uv[:, 1], uv[:, 0]]
        uv = uv[known]
        other = apply_pose(raw_surfaces[i+1][uv[:, 1], uv[:, 0]], cameras[i+1])
        fraction = float(np.count_nonzero(np.abs(target[known, 2]-other[:, 2]) <= .05)/len(points))
        if fraction < .05:
            raise ValueError("Insufficient adjacent native LiDAR surface overlap")
        fractions.append(fraction)
    return fractions


def build(root, split, visit, video, out):
    fetch_membership(split, visit, video)
    directory, archives = acquire(root, split, video)
    trajectory, rotation_error = parse_trajectory((directory/"lowres_wide.traj").read_text())
    rgb_inputs, cameras, intrinsics, masks, surfaces, checks = [], [], [], [], [], []
    member_hashes = {}
    native_k, native_pts, native_masks = [], [], []
    with (zipfile.ZipFile(directory/"lowres_wide.zip") as photos,
          zipfile.ZipFile(directory/"lowres_depth.zip") as depths,
          zipfile.ZipFile(directory/"lowres_wide_intrinsics.zip") as calibrations):
        rgb = frame_members(photos, video, "png")
        depth = frame_members(depths, video, "png")
        calib = frame_members(calibrations, video, "pincam")
        valid = associations(rgb, depth, calib, trajectory)
        times, stride, distance = select_window(valid, trajectory)
        for t in times:
            intr_time, pose_time = valid[t]
            rgb_b = photos.read(rgb[t])
            dep_b = depths.read(depth[t])
            cam_b = calibrations.read(calib[intr_time])
            for name, data in [(rgb[t],rgb_b),(depth[t],dep_b),(calib[intr_time],cam_b)]:
                member_hashes[name] = hashlib.sha256(data).hexdigest()
            width, height, fx, fy, cx, cy = np.fromstring(cam_b.decode(), sep=" ")
            size = (int(width), int(height))
            raster = (WIDTH, int(round(WIDTH*height/width/14))*14)
            with Image.open(io.BytesIO(rgb_b)) as im:
                if im.size != size or im.mode != "RGB":
                    raise ValueError("Unsupported RGB orientation or image grid")
                rgb_array = np.array(im.resize(raster, Image.Resampling.BICUBIC)).transpose(2, 0, 1)
            with Image.open(io.BytesIO(dep_b)) as im:
                if im.size != size or im.mode not in ("I;16", "I", "I;16L", "I;16B"):
                    raise ValueError("RGB/depth registered dimensions differ or input format changed")
                mm = np.asarray(im)
            native = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            sx, sy = np.divide(raster, size)
            kc = np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]]) @ native
            w2c = trajectory[pose_time]
            c2w = np.linalg.inv(w2c)
            grid = np.indices(mm.shape)[::-1].transpose(1, 2, 0)
            observed = (mm >= 200) & (mm <= 10000)
            p = (grid[observed] - [cx, cy])/[fx, fy]
            sensor_xyz = np.column_stack((p, np.ones(len(p)))) * (mm[observed].astype(float)[:,None]/1000.)
            native_surface = np.zeros(mm.shape+(3,), np.float32)
            native_surface[observed] = apply_pose(sensor_xyz, c2w)
            surface, mask, err = dense_reference(mm, native, kc, raster, c2w)
            y, x = np.nonzero(mask)
            uvz = apply_pose(surface[y[::19],x[::19]].astype(np.float64), w2c) @ kc.T
            reprojection = float(np.max(np.abs(uvz[:, :2]/uvz[:, 2:] - np.column_stack((x[::19], y[::19])))))
            if len(x) < 1000 or reprojection > .5005 or err > .50001:
                raise ValueError("Incorrect pixel association or incomplete reference")
            native_k.append(native)
            native_masks.append(observed)
            native_pts.append(native_surface)
            rgb_inputs.append(rgb_array.astype(np.float32) / 255.)
            cameras.append(w2c[:3].astype(np.float32))
            intrinsics.append(kc.astype(np.float32))
            masks.append(mask)
            surfaces.append(surface)
            checks.append(dict(frame=t, valid_pixels=int(mask.sum()), native_valid_pixels=int(observed.sum()),
                               fraction=float(mask.mean()), intrinsic_time_ms=intr_time, pose_time_ms=pose_time,
                               timestamp_pose_delta_ms=pose_time-t, max_correspondence_px=err,
                               fp32_roundtrip_max_px=reprojection))
    overlap = adjacency(native_pts, native_masks, cameras, native_k)
    model = {k:np.stack(v) for k,v in [("images",rgb_inputs),("extrs",cameras),("intrs",intrinsics)]}
    truth = dict(pointclouds=np.stack(surfaces), valid_mask=np.stack(masks))
    scene = f"arkit/{visit}/{video}"
    slug = scene.replace("/", "__") + ".npz"
    row = dict(scene=scene, visit=visit, video=video, split=split, frames=times, frame_unit="milliseconds",
               width=WIDTH, raster=list(raster), native_grid=list(size), raw_input_frames=len(rgb),
               exact_rgb_depth_pairs=len(rgb.keys() & depth.keys()), pose_and_intrinsics_matched_frames=len(valid),
               stride=stride, baseline_m=distance, source_sha256=member_hashes, archive_files=archives,
               input_sha256=array_signature(dict(rgb=model["images"],poses=model["extrs"],intrinsics=model["intrs"])),
               reference_sha256=array_signature(dict(mask=truth["valid_mask"], surface=truth["pointclouds"])),
               model_file="model/"+slug,reference_file="reference/"+slug,
               model_npz_sha256=save_npz(out/"model"/slug,model),reference_npz_sha256=save_npz(out/"reference"/slug,truth),
               reference_pixel_checks=checks, adjacent_native_overlap_at_50mm=overlap,
               pose_rotation_max_difference_vs_cv2=rotation_error)
    verify_fixture_and_packet(out, row)
    return row


def build_record(data, split, visit, video, packets):
    proof = fixture()
    row = build(data, split, visit, video, packets)
    record = dict(scene=row, independent_geometry_fixture=proof, publisher_source_revision=REVISION,
                  split_csv_sha256=METADATA_SHA,
                  registration=("Match exact rounded-millisecond RGB/depth timestamp; intrinsics prefer same then -1/+1 ms. "
                                "Match closest rounded trajectory sample strictly under 5 ms (earlier tie). "
                                "Choose first complete five-view window by ascending start, strides 1/2/4/8 "
                                "in matched timestamps, five distinct poses, adjacent available samples at most "
                                "350 ms apart, endpoint baseline 0.05–0.75 m. Actual native camera orientation. "
                                "W2C rotation from axis-angle; its inverse sends positive-Z sensor XYZ into world. "
                                "Use synchronized native mobile LiDAR in [0.2,10] m; preserve original measured "
                                "surface points on 518-wide raster without interpolation into missing points. "
                                "Reference support is dense at the physical sensor's 256×192 resolution. "
                                "Each adjacent pair must overlap on at least 5% of sampled native points at 50 mm."))
    fn = "arkit_training_input.json" if split == "Training" else f"arkit_validation_{visit}.json"
    write_json(OUTPUT/fn, record)
    print(row["scene"],row["frames"],row["adjacent_native_overlap_at_50mm"])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog=(
        "Validation is acquired only through lock_tco_final_ln_replication.py prepare after its scientific gate."))
    p.add_argument("split", choices=("Training",))
    p.add_argument("visit", type=int)
    p.add_argument("video", type=int)
    p.add_argument("--data", type=Path, default=DATA_ROOT)
    p.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    args = p.parse_args()
    build_record(args.data, args.split, args.visit, args.video, args.packets)


if __name__ == "__main__":
    main()
