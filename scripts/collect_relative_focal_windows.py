"""Collect the predeclared offset-25 replications with original RGB, VGGT and DISK/LightGlue."""
import argparse
import hashlib
import io
from pathlib import Path
import sys
import time

import numpy as np

from evaluation_stability import dump, sha
from focal_zoom_observations import (MODEL_SHA, ROOT, UPSTREAM, batches, camera_model, check_plan,
                                     image_digest, lightglue)
from score_saved_geometry import read


STUDY = "relative_focal_windows"
OUT = ROOT / "results" / STUDY
RAW = ROOT / "reports" / STUDY
PLAN = ROOT / "manifests" / (STUDY + ".json")
FRAMES = (25, 225, 425, 625, 825)
SCENES = ("heads/seq-01", "chess/seq-03", "fire/seq-03")


def source(archives, name):
    from PIL import Image
    from replay_sensor_windows_from_zips import sequence_archive
    from datasets.utils.cropping import resize_image
    scene, seq = name.split("/")
    pixels, poses, hashes = [], [], {}
    with sequence_archive(archives, scene, seq) as z:
        for frame in FRAMES:
            base = f"{seq}/frame-{frame:06d}"
            image = z.read(base + ".color.png")
            pose = z.read(base + ".pose.txt")
            for ext, value in (("color.png", image), ("pose.txt", pose)):
                hashes[base + "." + ext] = hashlib.sha256(value).hexdigest()
            with Image.open(io.BytesIO(image)) as img:
                resized = resize_image(img.convert("RGB"), (518, 392))
                pixels.append((np.asarray(resized, dtype=np.float32) / 255).transpose(2, 0, 1))
            poses.append(np.loadtxt(io.BytesIO(pose)))
    rgb = np.stack(pixels)
    return rgb, np.stack(poses), dict(files=hashes, input_rgb_sha256=image_digest(rgb))


def fixed():
    paths = ["docs/RELATIVE_FOCAL_CONFIRMATION.md", "docs/RELATIVE_FOCAL_WINDOWS.md",
             "scripts/relative_focal_confirmation.py", "scripts/collect_relative_focal_windows.py",
             "scripts/focal_zoom_gradient.py", "scripts/focal_zoom_observations.py",
             "scripts/learned_fixed_support.py", "manifests/focal_zoom_gradient.json",
             "manifests/visual_focal_baseline.json", "results/relative_focal_confirmation/summary.json"]
    return dict(frames=list(FRAMES), scenes=list(SCENES), model_sha256=MODEL_SHA,
                sources={p:sha(ROOT/p) for p in paths}, additional_cloud_usd=0)


def check():
    check_plan()
    spec = read(PLAN)
    if spec["fixed"] != fixed() or set(spec["inputs"]) != set(SCENES):
        raise ValueError("Frozen replication plan differs")
    return spec


def collect(args):
    import cv2
    import torch
    from learned_fixed_support import build
    spec = check()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261111)
    np.random.seed(20261111)
    cv2.setNumThreads(1)
    model = camera_model(args.checkpoint) if args.action == "camera" else None
    matcher = lightglue(args.support) if args.action == "tracks" else None
    for scene in SCENES:
        rgb, _, provenance = source(args.archives, scene)
        if provenance != spec["inputs"][scene]:
            raise ValueError("Public windows changed: " + scene)
        for tag, images in batches(rgb):
            folder = RAW / scene.replace("/", "_") / tag
            folder.mkdir(exist_ok=True, parents=True)
            path = folder / (args.action + ".json")
            identity = dict(scene=scene, batch=tag, rgb_sha256=image_digest(images), manifest_sha256=sha(PLAN))
            if path.exists():
                old = read(path)
                if any(old[k] != v for k,v in identity.items()):
                    raise ValueError("Conflicting observation")
                if matcher and sha(folder/"tracks.npz") != old["matches_sha256"]:
                    raise ValueError("Conflicting feature tracks")
                continue
            start = time.monotonic()
            if model is not None:
                with torch.inference_mode():
                    tokens, _ = model.aggregator(torch.from_numpy(images)[None])
                    enc = model.camera_head(tokens)[-1].cpu().numpy()
                    del tokens
                if enc.shape != (1,5,9) or not np.isfinite(enc).all() or enc.dtype != np.float32:
                    raise ValueError("Invalid replica camera encoding")
                value = dict(encoding=enc.tolist(), dtype=str(enc.dtype), device="cpu")
            else:
                pairs, meta = build(images, *matcher)
                part = folder / "tracks.partial.npz"
                np.savez_compressed(part, **{f"{n}_{k}":p[k] for n,p in enumerate(pairs) for k in ("xy_i","xy_j","fold")})
                part.replace(folder/"tracks.npz")
                value = dict(matches_sha256=sha(folder/"tracks.npz"), metadata=meta,
                             pair_order=[[p["i"],p["j"]] for p in pairs])
            duration = time.monotonic() - start
            pending = path.with_suffix(".partial.json")
            dump(pending, dict(**identity, **value, elapsed_seconds=duration))
            pending.replace(path)
            print(args.action, scene, tag, round(duration,2), "seconds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze","camera","tracks"))
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(UPSTREAM))
    if args.action == "freeze":
        check_plan()
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN, dict(fixed=fixed(), inputs={s:source(args.archives,s)[2] for s in SCENES}))
    else:
        collect(args)


if __name__ == "__main__":
    main()
