"""Lock the development winner and prepare its independent evaluation inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
import random
import shutil
import subprocess
import sys

from replay_tco_one_percent import GEOMETRY, GEOMETRY_CONTROLS, analyze_development
from run_profiled_d0 import read, sha, write_new
from tco_one_percent_protocol import CONFIRM_MANIFEST, DATA, FRAMES, MANIFEST, PUBLIC, ROOT, TIME_MANIFEST
from tco_one_percent_protocol import collect, file_hashes, job_for, read_committed, verify


ETH3D_ROOT = DATA / "data" / "eth3d"
ASSETS = DATA / "preparation" / "tco_one_percent_eth3d"
PREP_REVISION = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"
PREP_URL = f"https://raw.githubusercontent.com/facebookresearch/vggt-omega/{PREP_REVISION}/eval/prepare.py"
ETH3D_URL = "https://www.eth3d.net/data/"
TCO_MAP = ROOT / "third_party/TCO/datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf200.json"
CAPTURES = ROOT / "results/heldout_sparse_depth/captures"


def exposure():
    planned = read(TCO_MAP)
    prior = {row["scene"]: row["target_frames"] for row in map(read, CAPTURES.glob("*.json"))}
    rooms = read(ROOT / "manifests/holdout_policy_v1.json")["heldout_scenes"]
    chosen = {}
    for room in rooms:
        frames = [0, 100, 200, 300, 400] if room == "stairs" else FRAMES
        sequences = sorted(scene for scene in planned if scene.startswith(room + "/"))
        unused = [scene for scene in sequences if not set(frames).intersection(prior.get(scene, []))]
        sequence = (unused or sequences)[0]
        chosen[sequence] = dict(room=room, frames=frames,
                                previously_seen_room=any(scene.startswith(room + "/") for scene in prior),
                                prior_frame_ids=prior.get(sequence, []),
                                shared_frame_ids=sorted(set(frames).intersection(prior.get(sequence, []))))
    return chosen


def freeze():
    plan = verify()
    quality = read_committed(PUBLIC / "qualification.json")
    timing = read_committed(TIME_MANIFEST)
    development = [job for job in plan["jobs"] if job["arm"] != "cached14" or quality["last14_screen"]["eligible"]]
    development.extend(job for job in timing["jobs"] if job["phase"] == "equal_time")
    evidence = collect(development)
    report = analyze_development(plan, quality, evidence, timing)
    if report["verdict"] != "ready_for_confirmation":
        raise ValueError("A qualified development winner is required before opening either confirmation dataset")
    candidate = report["nominated"]
    budget = timing["locked_by_candidate"][candidate]
    comparator = report["candidates"][candidate]["matched_control"]
    arms = {"four_40", "four_10", "four_5", "zero_0", candidate, comparator}
    if budget:
        arms.add(f"four_{budget}")
    if budget > 1:
        arms.add(f"four_{budget - 1}")
    if candidate == GEOMETRY:
        arms.update(GEOMETRY_CONTROLS)
    selected = exposure()
    eth_frames = plan["confirmation"]["eth3d_frames"]
    cases = {"7scenes": {scene: row["frames"] for scene, row in selected.items()},
             "eth3d": dict.fromkeys(plan["confirmation"]["eth3d_scenes"], eth_frames)}
    jobs = []
    order = random.Random(plan["launch_order_seed"])
    for dataset, scenes in cases.items():
        for seed in plan["confirmation"]["seeds"]:
            block = []
            for scene, frames in scenes.items():
                for recipe in sorted(arms):
                    arm, steps = recipe.rsplit("_", 1)
                    block.append(job_for(scene, arm, int(steps), seed, "confirmation", dataset, frames))
            order.shuffle(block)
            jobs.extend(block)
    write_new(CONFIRM_MANIFEST, dict(
        candidate=candidate, locked_full_budget=budget, development_equal_time_control=comparator,
        development_sha256=sha(MANIFEST, text=True), timing_sha256=sha(TIME_MANIFEST, text=True),
        preparation_source_sha256=sha(__file__, text=True),
        physical_rooms={"7scenes": {scene: row["room"] for scene, row in selected.items()},
                        "eth3d": {scene: scene for scene in cases["eth3d"]}},
        transfer_exposure=selected, exposure_source=CAPTURES.relative_to(ROOT).as_posix(),
        eth3d_scenes=cases["eth3d"], eth3d_preparation=dict(module=PREP_URL, archive_base=ETH3D_URL),
        input_source_sha256=file_hashes(DATA / "data", cases["7scenes"]), jobs=jobs, rental_compute_usd=0))
    print(f"Locked {candidate} with {len(jobs)} cases on {sum(map(len, cases.values()))} scenes.")


def fetch(url, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.is_file():
        working = file.with_suffix(file.suffix + ".partial")
        subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error", "--retry", "2", "--continue-at", "-",
                        "--connect-timeout", "20", "-o", str(working), url], check=True)
        working.replace(file)
    return file


def preparation_module():
    path = fetch(PREP_URL, ASSETS / f"vggt_omega_prepare_{PREP_REVISION}.py")
    spec = importlib.util.spec_from_file_location("pinned_eth3d_preparation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, sha(path)


def archives_for(scene):
    scene_root = ASSETS / "raw" / scene
    entries = {}
    for kind in ("undistorted", "jpg", "depth"):
        name = f"{scene}_dslr_{kind}.7z"
        url = ETH3D_URL + name
        archive = fetch(url, ASSETS / "archives" / name)
        checksum = sha(archive)
        finished = ASSETS / "extracted" / f"{name}.json"
        if finished.is_file():
            if read(finished)["sha256"] != checksum:
                raise ValueError(f"Staged archive changed: {name}")
        else:
            extract(archive, scene_root, kind)
            write_new(finished, dict(sha256=checksum))
        entries[kind] = dict(url=url, sha256=checksum)
    return scene_root, entries


def extract(archive, root, kind):
    import py7zr

    with py7zr.SevenZipFile(archive) as handle:
        targets = handle.getnames()
        if kind == "jpg":
            targets = [name for name in targets if "dslr_calibration_jpg" in name
                       and name.rsplit("/", 1)[-1] in ("images.txt", "cameras.txt")]
        if not targets:
            raise ValueError(f"Archive lacks the required {kind} files: {archive.name}")
        handle.extract(path=root, targets=targets)


def prepare_scene(scene, frames, module):
    import numpy as np
    from PIL import Image

    extracted, archives = archives_for(scene)
    root = extracted if (extracted / "dslr_calibration_undistorted").is_dir() else module._find_scene(extracted, scene)
    distorted = root / "dslr_calibration_jpg"
    undistorted = root / "dslr_calibration_undistorted"
    original_cameras = module._read_colmap_cameras(distorted / "cameras.txt")
    original_views = camera_catalog(module, distorted / "images.txt")
    cameras = module._read_colmap_cameras(undistorted / "cameras.txt")
    views = camera_catalog(module, undistorted / "images.txt")
    sources = sorted((root / "images/dslr_images_undistorted").glob("*.JPG"))
    output = ETH3D_ROOT / scene
    rgb, depth, cam = (output / name for name in ("images/custom_undistorted",
                                                  "ground_truth_depth/custom_undistorted",
                                                  "custom_undistorted_cam"))
    for folder in (rgb, depth, cam):
        folder.mkdir(parents=True, exist_ok=True)
    for source in sources:
        linked = rgb / source.name
        if not linked.is_file():
            try:
                os.link(source, linked)
            except OSError:
                shutil.copy2(source, linked)
    prepared = {}
    for frame in frames:
        source = sources[frame]
        name = source.name
        view = views[name]
        camera = cameras[view["camera_id"]]
        original_view = original_views[name]
        original = original_cameras[original_view["camera_id"]]
        raw = root / "ground_truth_depth" / "dslr_images" / name
        map_path = depth / name
        camera_file = cam / f"{source.stem}.npz"
        if not map_path.is_file():
            if raw.stat().st_size != 4 * original.height * original.width:
                raise ValueError(f"Unexpected raw depth dimensions: {scene}/{name}")
            data = np.memmap(raw, dtype="<f4", mode="r", shape=(original.height, original.width))
            result = module._undistort_depth(data, original, camera)
            result = np.nan_to_num(result, nan=0, neginf=0, posinf=0)
            if np.count_nonzero(result > 1e-4) < 3:
                raise ValueError(f"Insufficient valid depth at {scene}/{name}")
            staged = map_path.with_suffix(".raw.partial")
            result.astype("<f4", copy=False).tofile(staged)
            staged.replace(map_path)
        if not camera_file.is_file():
            np.savez(camera_file, extrinsics=view["extrinsic"][:3], intrinsics=camera.intrinsic)
        with Image.open(source) as image:
            if image.size != (camera.width, camera.height) or map_path.stat().st_size != 4 * image.width * image.height:
                raise ValueError(f"Misaligned camera, image or depth in {scene}/{name}")
        prepared.setdefault("filenames", []).append(name)
        prepared.setdefault("hashes", {}).update(
            {path.relative_to(DATA / "data").as_posix(): sha(path) for path in (rgb / name, map_path, camera_file)})
    return dict(prepared, archives=archives, catalog_sha256=sha(undistorted / "images.txt"))


def camera_catalog(module, path):
    views = module._read_colmap_images(path)
    views_by_name = {name.rsplit("/", 1)[-1]: row for name, row in views.items()}
    if len(views_by_name) != len(views):
        raise ValueError(f"Duplicate image name in {path}")
    return views_by_name


def prepare_eth3d():
    import py7zr

    lock = read_committed(CONFIRM_MANIFEST)
    if lock["preparation_source_sha256"] != sha(__file__, text=True) or (
            lock["development_sha256"] != sha(MANIFEST, text=True)):
        raise ValueError("Confirmation no longer matches its frozen preparation")
    module, digest = preparation_module()
    evidence = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {scene: pool.submit(prepare_scene, scene, frames, module) for scene, frames in lock["eth3d_scenes"].items()}
        for scene, future in futures.items():
            evidence[scene] = future.result()
            print(f"{scene}: {len(evidence[scene]['filenames'])} frames ready", flush=True)
    lock["eth3d_preparation"].update(helper_sha256=digest, scenes=evidence,
                                    numpy=module.np.__version__, opencv=module.cv2.__version__, py7zr=py7zr.__version__)
    for scene, info in evidence.items():
        lock["input_source_sha256"].update(info["hashes"])
        for job in lock["jobs"]:
            if job["dataset"] == "eth3d" and job["scene"] == scene:
                job["images_expected"] = info["filenames"]
    next_manifest = CONFIRM_MANIFEST.with_suffix(".prepared.json")
    write_new(next_manifest, lock)
    print(f"Verify and commit {next_manifest.name} at {CONFIRM_MANIFEST.name} before independent evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "eth3d"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        prepare_eth3d()
