"""Project signed neighboring-frame packets independently for the MASt3R screen."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from neighbor_sparse_depth import FRAMES, SCENES, SCALE, ROOT, project_samples, resized_k, read, sha, write_arrays
from evaluation_stability import dump

PUBLIC = ROOT / "results/mast3r_partner_shape"
RAW = ROOT / "reports/mast3r_partner_shape"
SIDES = {"earlier": -4, "later": 4}


def prepare(assets):
    sources = read(ROOT / "results/source_frame_spacing/inputs.json")["frames"]
    calibration = np.asarray(read(ROOT / "manifests/sensor_frame_v14.json")["calibration"])
    k = resized_k(525, 640, 480)
    target_k = resized_k(525)
    target_k[[0, 1], [0, 1]] *= SCALE
    u, v = np.meshgrid(np.arange(40, 640, 80), np.arange(30, 480, 60))
    uv = np.column_stack([u.ravel(), v.ravel()])
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(k).T
    PUBLIC.mkdir(parents=True, exist_ok=True)
    origins = {}
    for side, offset in SIDES.items():
        records = {}
        for scene in SCENES:
            def checked(frame, suffix):
                path = assets / scene / f"frame-{frame:06d}.{suffix}"
                expected = sources[scene][str(frame)]["original"][suffix]
                if sha(path) != expected:
                    raise ValueError(f"Source identity changed: {path}")
                origins[path.relative_to(assets).as_posix()] = expected
                return path

            def camera(frame):
                return calibration @ np.linalg.inv(np.loadtxt(checked(frame, "pose.txt")))

            for target in FRAMES:
                source = target + offset
                relative = camera(target) @ np.linalg.inv(camera(source))
                with Image.open(checked(source, "depth.proj.png")) as im:
                    if im.size != (640, 480):
                        raise ValueError("Unexpected sensor size")
                    mm = np.asarray(im)[uv[:, 1], uv[:, 0]]
                projected = project_samples(mm, uv, rays, relative, target_k, source)
                nearest = {}
                for row in sorted(projected, key=lambda a: a[2]):
                    nearest.setdefault(tuple(row[:2]), row)
                records[f"{scene.split('/')[0]}_{target}"] = np.stack(list(nearest.values()))
        write_arrays(PUBLIC / f"{side}.npz", records)
    dump(PUBLIC / "inputs.json", dict(source_files=origins, packets={s:sha(PUBLIC/f"{s}.npz") for s in SIDES}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    prepare(parser.parse_args().assets)
