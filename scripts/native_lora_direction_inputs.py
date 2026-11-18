"""Prepare separate Stanford 12-Scenes image and temporal observation packets."""
import argparse
import hashlib
import io
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from download_stanford12_direction import ENVIRONMENTS
from prepare_profiled_inputs import grouped_folds, sha, verify_grouping
from run_profiled_d0 import write_new


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reports/native_lora_direction_v1"
SIDES = {"earlier": -1, "later": 1}
STATES = {"zero": 0, "minus_half": -.5, "plus_half": .5, "minus_full": -1,
          "plus_full": 1, "repeat_zero": 0, "repeat_full": 1}
FOLD_SEED = 20261118
WIDTH, HEIGHT = 512, 384


def calibration(raw):
    fields = dict(line.strip().split(" = ", 1) for line in raw.decode().splitlines())
    color_size = tuple(int(fields[f"m_color{axis}"]) for axis in ("Width", "Height"))
    depth_size = tuple(int(fields[f"m_depth{axis}"]) for axis in ("Width", "Height"))
    matrices = {}
    for camera in ("Color", "Depth"):
        for kind in ("Intrinsic", "Extrinsic"):
            name = camera + kind
            matrices[name] = np.fromstring(fields["m_calibration" + name], sep=" ").reshape(4, 4)
    for camera in ("Color", "Depth"):
        if not np.array_equal(matrices[camera + "Extrinsic"], np.eye(4)):
            raise ValueError("Expected publisher-aligned optical frames with identity extrinsics")
    if int(fields["m_depthShift"]) != 1000:
        raise ValueError("Publisher depth unit is not millimeters")
    sx, sy = np.array([WIDTH, HEIGHT]) / color_size
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return color_size, depth_size, resize @ matrices["ColorIntrinsic"][:3, :3], matrices["DepthIntrinsic"][:3, :3]


def valid_pose(raw):
    try:
        pose = np.asarray([float(part) for part in raw.decode().split()]).reshape(4, 4)
    except ValueError:
        # Released invalid tracking frames include a platform-specific INF spelling.
        return None
    if (np.isfinite(pose).all() and np.array_equal(pose[3], [0, 0, 0, 1])
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-4):
        return pose
    return None


def read_window(archive, room, names):
    base = room + "/data/frame-"
    entries = archive.read(room + "/split.txt").decode().splitlines()
    for line in entries:
        match = re.fullmatch(r"(sequence\d+) \[frames=(\d+)\]\s+\[start=(\d+)\s*;\s*end=(\d+)\]", line.strip())
        if not match:
            raise ValueError(f"Unknown publisher capture boundary: {line}")
        label, count, first, last = match.groups()
        first, last, count = int(first), int(last), int(count)
        if last - first + 1 != count:
            raise ValueError("Capture size contradicts its boundaries")
        poses = {}
        for index in range(first, last + 1):
            path = f"{base}{index:06d}"
            if all(path + "." + kind in names for kind in ("pose.txt", "color.jpg", "depth.png")):
                pose = valid_pose(archive.read(path + ".pose.txt"))
                if pose is not None:
                    poses[index] = pose
        for start in range(first + 1, last):
            for stride in (4, 8, 16):
                end = start + 4 * stride
                if end >= last or not all(index in poses for index in range(start - 1, end + 2)):
                    continue
                displacement = np.linalg.norm(poses[end][:3, 3] - poses[start][:3, 3])
                if .05 <= displacement <= .75:
                    return label, start + stride * np.arange(5), poses, float(displacement), stride
    raise ValueError(f"No eligible five-view window for {room}")


def grid_coordinates(width, height):
    x, y = np.meshgrid((np.arange(8) + .5) * width / 8 - .5,
                       (np.arange(8) + .5) * height / 8 - .5)
    return np.rint(np.column_stack((x.ravel(), y.ravel()))).astype(np.int64)


def project(depth_mm, source_frame, xy, source_intrinsic, target_intrinsic, source_pose, target_pose):
    depth = depth_mm[xy[:, 1], xy[:, 0]].astype(np.float64) / 1000
    valid = (depth >= .2) & (depth <= 10.)
    rays = np.column_stack((xy[valid], np.ones(valid.sum()))) @ np.linalg.inv(source_intrinsic).T
    transform = np.linalg.solve(target_pose, source_pose)
    target = (depth[valid, None] * rays) @ transform[:3, :3].T + transform[:3, 3]
    positive = (target[:, 2] >= .2) & (target[:, 2] <= 10.)
    target = target[positive]
    projected = target @ target_intrinsic.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(np.int64)
    inside = ((pixels >= 0) & (pixels < [WIDTH, HEIGHT])).all(axis=1)
    sampled_xy = xy[valid][positive][inside]
    source = np.column_stack((np.full(len(sampled_xy), source_frame), sampled_xy))
    pixels, axial = pixels[inside], target[inside, 2]
    # Nearest depth wins. Equal-depth collisions retain source row-major order.
    selected = {}
    for i in np.argsort(axial, kind="stable"):
        selected.setdefault(tuple(pixels[i]), int(i))
    ids = list(selected.values())
    return dict(xy=pixels[ids], rows=np.column_stack((pixels[ids], axial[ids])),
                source_ids=source[ids])


def frame_path(room, index, suffix):
    return f"{room}/data/frame-{index:06d}.{suffix}"


def packet(archive, room, targets, offset, poses, sizes, intrinsics, output, photos, room_id):
    _, depth_size = sizes
    color_k, depth_k = intrinsics
    views, files = [], {}
    grid = grid_coordinates(*depth_size)
    for index in targets:
        source = int(index) + offset
        path = frame_path(room, source, "depth.png")
        raw = archive.read(path)
        files[path] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != depth_size:
                raise ValueError(f"Depth dimensions differ from publisher metadata: {path}")
            depth = np.asarray(image)
        if depth.dtype.kind not in ("u", "i") or depth.ndim != 2 or depth.max() > 65535 or depth.min() < 0:
            raise ValueError(f"Expected 16-bit publisher depth: {path}")
        views.append(project(depth, source, grid, depth_k, color_k, poses[source], poses[int(index)]))
    labels, audit = grouped_folds(views, FOLD_SEED, room_id)
    verify_grouping(views, labels)
    arrays = {}
    for i, (view, (_, fold)) in enumerate(zip(views, labels)):
        for key in ("xy", "source_ids"):
            arrays[f"{key}_{i}"] = view[key]
        arrays[f"depth_{i}"] = view["rows"][:, 2]
        arrays[f"fold_{i}"] = fold
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    overlay = Image.new("RGB", (WIDTH * 5, HEIGHT))
    for i, (image, view) in enumerate(zip(photos, views)):
        panel = image.copy()
        drawing = ImageDraw.Draw(panel)
        for (x, y), z in zip(view["xy"], view["rows"][:, 2]):
            drawing.ellipse((x - 3, y - 3, x + 3, y + 3), fill=None,
                            outline=(255, int(np.clip(z * 45, 30, 255)), 0), width=1)
        overlay.paste(panel, (i * WIDTH, 0))
    overlay.save(output.with_suffix(".png"))
    return dict(sha256=sha(output), files=files, source_frames=(targets+offset).tolist(), folds=audit)


def prepare(archive_dir, destination):
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for environment in ENVIRONMENTS:
        archive_path = archive_dir / f"{environment}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            rooms = sorted({n.rsplit("/", 1)[0] for n in names if n.startswith(environment + "/")
                            and n.endswith("/split.txt")}, key=str.casefold)
            room = rooms[0]
            room_path = destination / environment
            room_path.mkdir()
            info = archive.read(room + "/info.txt")
            csize, dsize, ck, dk = calibration(info)
            capture, targets, poses, baseline, stride = read_window(archive, room, names)
            photos, sources = [], {}
            for frame in targets:
                rgb_path = frame_path(room, frame, "color.jpg")
                raw = archive.read(rgb_path)
                sources[rgb_path] = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != csize:
                        raise ValueError(f"RGB dimensions differ from metadata: {rgb_path}")
                    photos.append(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS))
            for frame in sorted({int(t) + delta for t in targets for delta in (-1, 0, 1)}):
                key = frame_path(room, frame, "pose.txt")
                sources[key] = hashlib.sha256(archive.read(key)).hexdigest()
            for leaf in ("info.txt", "split.txt"):
                sources[room + "/" + leaf] = hashlib.sha256(archive.read(room + "/" + leaf)).hexdigest()
            image_path = room_path / "images.npz"
            with image_path.open("xb") as stream:
                np.savez_compressed(stream, images=np.stack([np.asarray(p) for p in photos]),
                                    c2w=np.stack([poses[int(n)] for n in targets]), K=ck)
            packets = {side: packet(archive, room, targets, offset, poses, (csize, dsize), (ck, dk),
                                    room_path / f"{side}.npz", photos, f"{room}/{side}")
                       for side, offset in SIDES.items()}
        records[environment] = dict(room=room, rooms=rooms, capture=capture, stride=stride,
                                    target_frames=targets.tolist(), baseline_m=baseline,
                                    archive_sha256=sha(archive_path), archive_bytes=archive_path.stat().st_size,
                                    image_sha256=sha(image_path), source_hashes=sources, packets=packets)
        print(environment, room, capture, targets.tolist(), [v["folds"]["fold_counts"] for v in packets.values()],
              flush=True)
    write_new(destination / "inputs.json", dict(environments=records, fold_seed=FOLD_SEED,
              sampling="centered 8x8; half-pixel resize; ties-to-even; [0.2,10] m inclusive; nearest axial z-buffer",
              license="CC BY-NC-SA 4.0", source="https://graphics.stanford.edu/projects/reloc/"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("--attempt", default="intake01", choices=("intake01", "intake02"))
    args = parser.parse_args()
    prepare(args.archive_dir, RAW / args.attempt)


if __name__ == "__main__":
    main()
