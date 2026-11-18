"""Register fresh-room inputs and disjoint optimization, selection and scoring roles."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image

from native_lora_direction_inputs import (ROOT, SIDES, WIDTH, HEIGHT, calibration, frame_path,
                                          project, read_window)
from prepare_profiled_inputs import grouped_folds, read, sha, verify_grouping
from run_profiled_d0 import IMAGE, write_new
from score_native_lora_direction import data_packet, held_mask


RAW = ROOT / "reports/native_lora_step_v1"
PUBLIC = ROOT / "results/native_lora_step_v1"
MANIFEST = ROOT / "manifests/native_lora_step_v1.json"
ROOMS = {"apt1": "apt1/living", "apt2": "apt2/kitchen", "office2": "office2/5b"}
STATES = {"zero": 0., "quarter": .25, "half": .5, "full": 1.}
EXPECTED_RUNTIME = ROOT / "results/lora_profiled_v1/environment.json"
MIN_PARTITION, MIN_HELD = 8, 16
EPSILON_MM = .05
BASE_GAIN = .01
SOURCE_PATHS = (
    "scripts/native_lora_step_inputs.py", "scripts/run_native_lora_step.py",
    "scripts/run_native_lora_step_local.py", "scripts/score_native_lora_step.py",
    "scripts/replay_native_lora_step.py", "scripts/test_native_lora_step.py",
    "scripts/run_native_lora_direction.py", "scripts/native_lora_direction_inputs.py",
    "scripts/score_native_lora_direction.py", "scripts/prepare_profiled_inputs.py",
    "scripts/profiled_network.py", "scripts/profiled_calibration.py", "scripts/profiled_runtime.py",
    "scripts/mast3r_network_capture.py", "scripts/mast3r_network_priors.py",
    "scripts/check_profiled_endpoints.py", "scripts/mast3r_joint_prior_loss.py",
    "scripts/mast3r_calibrated_alignment.py", "scripts/mast3r_alignment_hook.py",
    "scripts/run_profiled_lora.py", "scripts/run_profiled_d0_local.py",
    "scripts/run_profiled_d1_local.py", "third_party/TCO/peft/lora.py",
    "third_party/mast3r/mast3r/cloud_opt/sparse_ga.py",
    "docs/NATIVE_LORA_STEP_SELECTION.md",
)


def cases():
    return [(env, side) for env in ROOMS for side in SIDES]


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<i8").tobytes()).hexdigest()


def grids(width, height, side):
    phase = {"earlier": .25, "later": .75}[side]
    x, y = np.meshgrid((np.arange(8) + phase) * width / 8 - .5,
                       (np.arange(8) + phase) * height / 8 - .5)
    return np.rint(np.column_stack((x.ravel(), y.ravel()))).astype(np.int64)


def subset(views, masks):
    return [{key: value[mask] for key, value in view.items()} for view, mask in zip(views, masks)]


def split(views, case):
    def labels(packets, role):
        folds, _ = grouped_folds(packets, 20261118, case + "/" + role)
        verify_grouping(packets, folds)
        return [fold for _, fold in folds]

    half = [f == 1 for f in labels(views, "outer")]
    second = labels(subset(views, half), "selector")
    selector = [mask.copy() for mask in half]
    for mask, group in zip(selector, second):
        mask[mask] = group == 1
    adapt = [~mask for mask in selector]
    training = labels(subset(views, adapt), "support_query")
    result = [np.full(len(view["xy"]), 2, dtype=np.int8) for view in views]
    for folds, available, fitted in zip(result, adapt, training):
        folds[available] = fitted
    counts = [[int((fold == role).sum()) for fold in result] for role in range(3)]
    if min(min(views_count) for views_count in counts) < MIN_PARTITION:
        raise ValueError(f"Insufficient preflight fitting or selection coverage for {case}: {counts}")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        check_views = subset(views, [(fold == a) | (fold == b) for fold in result])
        check_labels = [(None, (fold[(fold == a) | (fold == b)] == b).astype(np.int8)) for fold in result]
        verify_grouping(check_views, check_labels)
    return result, counts


def extract(archives, intake):
    intake.mkdir(parents=True, exist_ok=False)
    report = {}
    for env, room in ROOMS.items():
        items = intake / env
        items.mkdir()
        archive_path = archives / f"{env}.zip"
        with zipfile.ZipFile(archive_path) as z:
            names = set(z.namelist())
            info = z.read(room + "/info.txt")
            color_size, depth_size, color_k, depth_k = calibration(info)
            capture, targets, poses, baseline, stride = read_window(z, room, names)
            source_files = {}

            def asset(frame, suffix):
                path = frame_path(room, frame, suffix)
                raw = z.read(path)
                source_files[path] = hashlib.sha256(raw).hexdigest()
                return raw

            rgb = []
            for frame in targets:
                with Image.open(io.BytesIO(asset(frame, "color.jpg"))) as image:
                    if image.size != color_size:
                        raise ValueError("Unexpected publisher RGB dimensions")
                    rgb.append(np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)))
            image_path = items / "images.npz"
            np.savez_compressed(image_path, images=np.stack(rgb), K=color_k,
                                c2w=np.stack([poses[int(frame)] for frame in targets]))
            for frame in sorted({int(t) + d for t in targets for d in (-1, 0, 1)}):
                asset(frame, "pose.txt")
            packets = {}
            for side, delta in SIDES.items():
                views = []
                for frame in targets:
                    source = int(frame) + delta
                    with Image.open(io.BytesIO(asset(source, "depth.png"))) as image:
                        if image.size != depth_size:
                            raise ValueError("Unexpected publisher depth dimensions")
                        values = np.asarray(image)
                    if values.dtype.kind not in "ui" or values.ndim != 2 or values.min() < 0 or values.max() > 65535:
                        raise ValueError("Unexpected axial depth image encoding")
                    views.append(project(values, source, grids(*depth_size, side), depth_k, color_k,
                                         poses[source], poses[int(frame)]))
                labels, counts = split(views, f"{room}/{side}")
                arrays = {}
                for i, (view, fold) in enumerate(zip(views, labels)):
                    arrays |= {f"xy_{i}": view["xy"], f"source_ids_{i}": view["source_ids"],
                               f"depth_{i}": view["rows"][:, 2], f"fold_{i}": fold}
                path = items / f"{side}.npz"
                np.savez_compressed(path, **arrays)
                packets[side] = dict(sha256=sha(path), source_frames=(targets+delta).tolist(), counts=counts)
            for leaf in ("info.txt", "split.txt"):
                name = room + "/" + leaf
                source_files[name] = hashlib.sha256(z.read(name)).hexdigest()
        earlier, later = (data_packet(items / f"{side}.npz", packets[side]["sha256"]) for side in SIDES)
        for side, (first, second) in zip(SIDES, ((earlier, later), (later, earlier))):
            coverage = []
            for view, (a, b) in enumerate(zip(first, second)):
                mask = held_mask(a, b)
                if mask.sum() < MIN_HELD:
                    raise ValueError(f"Held coverage preflight failed for {room}/{side}/{view}")
                coverage.append(dict(view=view, count=int(mask.sum()), excluded=int((~mask).sum()),
                                     xy_sha256=array_digest(b["xy"][mask])))
            packets[side]["held_coverage"] = coverage
        print(room, targets.tolist(),
              {side: [view["count"] for view in item["held_coverage"]] for side, item in packets.items()},
              flush=True)
        report[env] = dict(room=room, capture=capture, stride=stride, baseline_m=baseline,
                           target_frames=targets.tolist(), archive_sha256=sha(archive_path),
                           archive_bytes=archive_path.stat().st_size, image_sha256=sha(image_path),
                           packets=packets, source_files=source_files)
    write_new(intake / "intake.json", dict(environments=report, preflight="counts and coordinate hashes only"))


def freeze(weights):
    ref = read(EXPECTED_RUNTIME)
    if sha(weights / "model.safetensors") != ref["checkpoint_sha256"]:
        raise ValueError("Unexpected checkpoint bytes")
    prepared = RAW / "intake/intake.json"
    manifest = dict(schema="native_lora_step_v1", source="https://graphics.stanford.edu/projects/reloc/",
                    reserved_environment="office1", rooms=ROOMS, cases=cases(), states=STATES,
                    input_sha256=sha(prepared), inputs=read(prepared),
                    source_sha256_lf={name: sha(ROOT / name, text=True) for name in SOURCE_PATHS},
                    image=IMAGE, checkpoint_sha256=ref["checkpoint_sha256"],
                    config_sha256=sha(weights / "config.json"), runtime_sha256_lf=sha(EXPECTED_RUNTIME, text=True),
                    selection_tolerance_mm=EPSILON_MM, consistency_relative=BASE_GAIN,
                    rental_compute_usd=0, min_held=MIN_HELD, min_partition=MIN_PARTITION)
    write_new(MANIFEST, manifest)
    verify_registration()


def verify_registration(committed=False):
    item = read(MANIFEST)
    if (item["rooms"] != ROOMS or item["cases"] != [list(case) for case in cases()] or
            item["states"] != STATES or item["image"] != IMAGE or item["rental_compute_usd"] != 0 or
            item["min_held"] != MIN_HELD or item["min_partition"] != MIN_PARTITION or
            item["selection_tolerance_mm"] != EPSILON_MM or item["consistency_relative"] != BASE_GAIN):
        raise ValueError("Registration disagrees with the frozen design")
    inputs = RAW / "intake/intake.json"
    if item["input_sha256"] != sha(inputs) or item["inputs"] != read(inputs):
        raise ValueError("Input registration changed")
    for name, expected in item["source_sha256_lf"].items():
        if sha(ROOT / name, text=True) != expected:
            raise ValueError(f"Registered source changed: {name}")
    if sha(EXPECTED_RUNTIME, text=True) != item["runtime_sha256_lf"]:
        raise ValueError("Runtime reference changed")
    if committed:
        path = MANIFEST.relative_to(ROOT).as_posix()
        old = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(ROOT),
                                       "show", "HEAD:" + path])
        if old.replace(b"\r\n", b"\n") != MANIFEST.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Commit the frozen registration before scientific use")
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("intake", "freeze"))
    parser.add_argument("--archives", type=Path, default=Path("E:/ResearchData/TCO/data/stanford12"))
    parser.add_argument("--weights", type=Path, default=Path("E:/ResearchData/TCO/checkpoints/MASt3R/hf"))
    args = parser.parse_args()
    if args.stage == "intake":
        extract(args.archives, RAW / "intake")
    else:
        freeze(args.weights)
