"""Register the local Courtyard traces and native-batch camera checkpoint study."""

import argparse
from collections import defaultdict
import random
import re
import subprocess

from run_profiled_d0 import read, sha, write_new
from tco_full40_recompute_protocol import LOCKED as PRIOR_LOCKED
from tco_full40_recompute_protocol import runtime_pins, verify as verify_prior
from tco_frozen_prefix_protocol import CHECKPOINT, FRAMES, IMAGE, ROOT, SCENES, TCO
from tco_one_percent_protocol import CONFIRM_MANIFEST, DATA, file_hashes, read_committed


STUDY = "tco_courtyard_camera_v1"
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
LOCKED = ROOT / "manifests/tco_courtyard_camera_locked_v1.json"
REGISTRATIONS = {"initial": INITIAL, "locked": LOCKED}
HANDOFF = "docs/TCO_COURTYARD_CAMERA_HANDOFF.md"
PREPARED = PRIVATE / "prepared_inputs.json"
INPUT_AUDIT = PRIVATE / "decoded_inputs.json"
REFERENCE = "c8_off"
CAMERA = "camera"
DEPTH_ONLY = ("c1_off", "c1_on")
COURTYARD_FRAMES = (0, 5, 10, 15, 20)
COURTYARD_IMAGES = tuple(f"DSC_{number:04d}.JPG" for number in (286, 291, 296, 301, 306))
TRACE_MILESTONES = (0, 1, 5, 10, 39, 40)
CONFIRM_SEEDS = (7, 77, 777)
TEN_FRAMES = tuple(range(0, 1000, 100))
TEN_WINDOWS = {"heads/seq-01": (42, 43, 44), "office/seq-02": CONFIRM_SEEDS,
               "pumpkin/seq-01": CONFIRM_SEEDS}
FIVE_DEFAULT = (49, 249, 449, 649, 849)
FIVE_7SCENES = dict.fromkeys(("chess/seq-05", "fire/seq-04", "heads/seq-01", "office/seq-06",
                             "pumpkin/seq-07", "redkitchen/seq-04"), FIVE_DEFAULT)
FIVE_7SCENES["stairs/seq-04"] = (49, 149, 249, 349, 449)
FIVE_ETH3D = {scene: tuple(range(start, start + 21, 5))
              for scene, start in (("delivery_area", 23), ("electro", 24), ("facade", 55),
                                   ("kicker", 9), ("office", 4), ("playground", 17), ("terrains", 21))}
ORDER_SEED = 20261125
BOOTSTRAP_SAMPLES = 10_000
MARGIN = 1.01
CPU_PROFILES = {"fixture", "archived", "diagnosis"}
SOURCES = (
    HANDOFF,
    "scripts/tco_courtyard_camera_protocol.py",
    "scripts/run_tco_courtyard_camera_local.py",
    "scripts/run_tco_courtyard_camera.py",
)
# Trace and camera source modules are added by explicit path when registering.
# The lock also binds the independent scorer and replay code.
LOCK_SOURCES = ("scripts/prepare_tco_one_percent_confirmation.py",
                "scripts/run_tco_courtyard_input_audit.py")
AUTHORITIES = ("diagnosis.json", "qualification.json", "screen.json")


def case(identifier, cohort, policy=REFERENCE, *, scene=SCENES[0], frames=FRAMES, steps=40,
         seed=42, dataset="7scenes", width=518, profile=False, score=False, probe=None, **fields):
    job = dict(id=identifier, cohort=cohort, policy=policy, dataset=dataset, scene=scene,
               frames=list(frames), steps=steps, seed=seed, width=width, arm="four",
               phase="confirmation", profile=profile, score=score, **fields)
    if probe is not None:
        job["probe"] = probe
    return job


def courtyard(identifier, cohort, policy=REFERENCE, **kwargs):
    return case(identifier, cohort, policy, dataset="eth3d", scene="courtyard",
                frames=COURTYARD_FRAMES, images_expected=list(COURTYARD_IMAGES), **kwargs)


def initial_jobs():
    jobs = [courtyard("archived_rescore", "rescore", steps=0, profile="archived")]
    order = random.Random(ORDER_SEED)
    for seed in (777, 7, 77):
        policies = [REFERENCE, *DEPTH_ONLY, "native_repeat"]
        order.shuffle(policies)
        for token in policies:
            policy = REFERENCE if token == "native_repeat" else token
            trace = f"trace_{token}_{seed}"
            jobs.append(courtyard(trace, "trace", policy, seed=seed, profile="trace", score=True))
            jobs.append(courtyard(f"readout_{token}_{seed}", "readout", policy, seed=seed,
                                  profile="readout", trace_job=trace))
    jobs.extend((courtyard("trace_clean_777", "trace_clean", seed=777, score=True),
                 courtyard("diagnose_courtyard", "diagnosis", steps=0, profile="diagnosis"),
                 case("camera_fixture", "fixture", CAMERA, steps=0, profile="fixture")))
    for policy in (REFERENCE, CAMERA):
        jobs.extend((case(f"identity_{policy}", "qualification", policy, steps=0, probe="identity"),
                     case(f"fixed_{policy}", "qualification", policy, steps=2, width=294, probe="fixed"),
                     case(f"gate_{policy}", "gate", policy, steps=2, frames=TEN_FRAMES),
                     case(f"profile10_{policy}", "gate", policy, steps=2, frames=TEN_FRAMES, profile="timeline"),
                     case(f"screen5_{policy}", "screen5", policy, steps=2)))
    jobs.extend((case("leak_camera", "qualification", CAMERA, steps=2, width=294, probe="leak"),
                 case("promote_camera", "promotion", CAMERA, frames=TEN_FRAMES)))
    return jobs


def window_jobs(eth_images):
    jobs = [case(f"window_7scenes_{scene.split('/')[0]}", "confirmation", scene=scene, frames=frames)
            for scene, frames in FIVE_7SCENES.items()]
    jobs.extend(case(f"window_eth3d_{scene}", "confirmation", dataset="eth3d", scene=scene,
                     frames=frames, images_expected=eth_images[scene]) for scene, frames in FIVE_ETH3D.items())
    jobs.append(courtyard("window_courtyard", "revisit"))
    return jobs


def locked_jobs(windows):
    jobs = []
    order = random.Random(ORDER_SEED)
    for window in windows:
        dataset, scene, frames, cohort = (window[key] for key in ("dataset", "scene", "frames", "cohort"))
        label = f"{dataset}_{scene.split('/')[0]}" if cohort == "confirmation" else "courtyard"
        fields = {"images_expected": window["images_expected"]} if "images_expected" in window else {}
        for seed in CONFIRM_SEEDS:
            pair = [case(f"{cohort}_{label}_{policy}_{seed}", cohort, policy, dataset=dataset,
                         scene=scene, frames=frames, seed=seed, score=True, **fields)
                    for policy in (REFERENCE, CAMERA)]
            order.shuffle(pair)
            jobs.extend(pair)
    for scene, seeds in TEN_WINDOWS.items():
        room = scene.split("/")[0]
        for index, seed in enumerate(seeds):
            pair = [
                case(f"ten_{room}_{CAMERA}_{seed}", "ten", CAMERA, scene=scene,
                     frames=TEN_FRAMES, seed=seed, score=True),
                case(f"ten_{room}_{REFERENCE}_{seed}", "ten", scene=scene,
                     frames=TEN_FRAMES, seed=seed, score=True,
                     **({"first_gate": f"ten_{room}_{REFERENCE}_{seeds[0]}"} if index else {})),
            ]
            order.shuffle(pair)
            jobs.extend(pair)
    return jobs


def hashes_for(jobs, data_root=None):
    root = data_root or DATA / "data"
    seven = defaultdict(set)
    eth = defaultdict(dict)
    for job in jobs:
        if job["dataset"] == "7scenes":
            seven[job["scene"]].update(job["frames"])
        elif job["dataset"] == "eth3d":
            if len(job["frames"]) != len(job["images_expected"]):
                raise ValueError(f"Wrong number of image filenames in {job['id']}")
            for index, filename in zip(job["frames"], job["images_expected"]):
                if index in eth[job["scene"]] and eth[job["scene"]][index] != filename:
                    raise ValueError(f"Conflicting filenames for {job['scene']} index {index}")
                eth[job["scene"]][index] = filename
        else:
            raise ValueError(f"Unknown dataset in {job['id']}")
    hashes = file_hashes(root, {scene: sorted(frames) for scene, frames in sorted(seven.items())})
    for scene, frames in sorted(eth.items()):
        base = root / "eth3d" / scene
        catalog = sorted(path.name for path in (base / "images/custom_undistorted").glob("*.JPG"))
        for index, name in sorted(frames.items()):
            if catalog[index] != name:
                raise ValueError(f"Unexpected lexical ETH3D index {scene}/{index}")
            for key in (f"images/custom_undistorted/{name}", f"ground_truth_depth/custom_undistorted/{name}",
                        f"custom_undistorted_cam/{name.removesuffix('.JPG')}.npz"):
                hashes[f"eth3d/{scene}/{key}"] = sha(base / key)
    return hashes


def prior_frame_exposure(windows):
    """Audit identifiable sequence/frame pairs in prior project registrations and capture records."""
    indexed = defaultdict(lambda: defaultdict(set))
    files = [*sorted((ROOT / "manifests").glob("*.json")),
             *sorted((ROOT / "results/heldout_sparse_depth/captures").glob("*.json"))]
    seven_scene = re.compile(r"^(chess|fire|heads|office|pumpkin|redkitchen|stairs)/seq-\d\d$")
    file_frame = re.compile(r"(?:^|/)(?P<scene>[^/]+/seq-\d\d)/frame-(?P<frame>\d+)\.")

    def add(dataset, scene, frame, source):
        if type(frame) is int and frame >= 0:
            indexed[(dataset, scene)][frame].add(source)

    def walk(value, source, inherited_scene=None, inherited_dataset=None):
        if isinstance(value, list):
            for item in value:
                walk(item, source, inherited_scene, inherited_dataset)
        elif isinstance(value, dict):
            scene = value.get("scene", value.get("sequence_name", value.get("sequence", inherited_scene)))
            dataset = value.get("dataset", inherited_dataset)
            if isinstance(scene, str):
                scene = scene.lower().replace("\\", "/")
                if seven_scene.fullmatch(scene):
                    dataset = "7scenes"
                elif scene in {*FIVE_ETH3D, "courtyard"}:
                    dataset = "eth3d"
                if dataset in {"7scenes", "eth3d"}:
                    for field in ("frames", "frame_ids", "target_frames", "source_frames", "support_frames",
                                  "frame", "frame_id", "reference_frame", "target_frame"):
                        items = value.get(field, [])
                        for frame in items if isinstance(items, list) else [items]:
                            add(dataset, scene, frame, source)
            for key, item in value.items():
                match = file_frame.search(key.replace("\\", "/"))
                if match and seven_scene.fullmatch(match["scene"]):
                    add("7scenes", match["scene"], int(match["frame"]), source)
                if seven_scene.fullmatch(key):
                    if isinstance(item, list):
                        for frame in item:
                            add("7scenes", key, frame, source)
                    walk(item, source, key, "7scenes")
                elif key in FIVE_ETH3D or key == "courtyard":
                    if isinstance(item, list):
                        for frame in item:
                            add("eth3d", key, frame, source)
                    walk(item, source, key, "eth3d")
                elif isinstance(item, (dict, list)):
                    walk(item, source, scene, dataset)

    authorities = []
    for path in files:
        if path in REGISTRATIONS.values():
            continue
        source = path.relative_to(ROOT).as_posix()
        walk(read(path), source)
        authorities.append(dict(file=source, sha256_lf=sha(path, text=True)))

    rows = []
    for job in windows:
        dataset, scene = job["dataset"], job["scene"]
        overlaps = [{"frame": frame, "sources": sorted(indexed[(dataset, scene)][frame])}
                    for frame in job["frames"] if indexed[(dataset, scene)][frame]]
        rows.append(dict(dataset=dataset, scene=scene, frames=job["frames"],
                         known_physical_location=True, observed_exact_overlaps=overlaps))
    return dict(method="exact scene/frame fields and registered 7-Scenes file paths",
                limitations="Exact indexed ID audit over available tracked manifests and held-out captures."
                            " Adjacent views and unindexed historical inputs can share content.",
                authorities=authorities, windows=rows)


def source_pins(prior, extras):
    files = set(prior["source_sha256_lf"]) | set(SOURCES) | set(extras)
    for name in files:
        if not name.startswith(("scripts/", "docs/", "third_party/", "configs/", "environment/")):
            raise ValueError(f"Invalid source path {name}")
    return {name: sha(ROOT / name, text=True) for name in sorted(files)}


def register_initial(extra_sources):
    prior = verify_prior("locked")
    actual = subprocess.check_output(["git", "-C", str(ROOT / "third_party/TCO"), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != TCO or sha(DATA / "checkpoints/VGGT-1B/model.safetensors") != CHECKPOINT:
        raise ValueError("TCO or VGGT weight pin differs")
    jobs = initial_jobs()
    hashes = hashes_for(jobs)
    previous = prior["input_source_sha256"]
    if any(previous[path] != digest for path, digest in hashes.items()):
        raise ValueError("Diagnostic and engineering inputs differ from the old locked study")
    known_images = {scene: [path.name for path in sorted((DATA / f"data/eth3d/{scene}/images/custom_undistorted").glob("*.JPG"))]
                    for scene in FIVE_ETH3D}
    fresh_windows = window_jobs({scene: [known_images[scene][index] for index in frames]
                                 for scene, frames in FIVE_ETH3D.items()})
    write_new(INITIAL, dict(study=STUDY, registration="initial", **runtime_pins(),
                            previous_sha256=sha(PRIOR_LOCKED, text=True), reference=REFERENCE,
                            candidate=CAMERA, launch_order_seed=ORDER_SEED, bootstrap_seed=ORDER_SEED,
                            bootstrap_samples=BOOTSTRAP_SAMPLES, margin=MARGIN, trace_milestones=TRACE_MILESTONES,
                            max_attempts=2, historical_rescore_tolerance_mm=0.01, planned_windows=fresh_windows,
                            exposure=prior_frame_exposure(fresh_windows),
                            jobs=jobs, input_source_sha256=hashes,
                            source_sha256_lf=source_pins(prior, extra_sources)))
    print(f"Registered {len(jobs)} initial cases and proposed {len(fresh_windows)} five-view windows")


def preparation():
    """Materialize only the missing public maps and cameras from existing verified caches."""
    from prepare_tco_one_percent_confirmation import ASSETS, PREP_REVISION, preparation_module, prepare_scene

    if PREPARED.exists():
        raise ValueError(f"An input preparation receipt already exists at {PREPARED}")
    initial = verify("initial")
    helper = ASSETS / f"vggt_omega_prepare_{PREP_REVISION}.py"
    expected = read_committed(CONFIRM_MANIFEST)["eth3d_preparation"]["helper_sha256"]
    if sha(helper) != expected:
        raise ValueError("The cached ETH3D preparation helper differs from the earlier input registration")
    for scene in FIVE_ETH3D:
        for kind in ("undistorted", "jpg", "depth"):
            name = f"{scene}_dslr_{kind}.7z"
            if not (ASSETS / "archives" / name).is_file() or not (ASSETS / "extracted" / f"{name}.json").is_file():
                raise FileNotFoundError(f"The extracted cached ETH3D authority is unavailable for {name}")
    module, helper_sha = preparation_module()
    decoded = {}
    raw = {}
    for scene, frames in FIVE_ETH3D.items():
        raw_root = ASSETS / "raw" / scene
        root = raw_root if (raw_root / "dslr_calibration_undistorted").is_dir() else module._find_scene(raw_root, scene)
        selected = next(job["images_expected"] for job in initial["planned_windows"] if job["scene"] == scene)
        for filename in selected:
            for directory in ("images/dslr_images_undistorted", "ground_truth_depth/dslr_images"):
                path = root / directory / filename
                raw[path.relative_to(ASSETS).as_posix()] = sha(path)
        for directory in ("dslr_calibration_undistorted", "dslr_calibration_jpg"):
            for name in ("cameras.txt", "images.txt"):
                path = root / directory / name
                raw[path.relative_to(ASSETS).as_posix()] = sha(path)
        decoded[scene] = prepare_scene(scene, frames, module)
        if decoded[scene]["filenames"] != selected:
            raise ValueError(f"Preparation used other image indices in {scene}")
        print(f"Prepared {len(frames)} cached views of {scene}", flush=True)
    write_new(PREPARED, dict(study=STUDY, registration_sha256=sha(INITIAL, text=True),
                             helper_sha256=helper_sha, raw_sha256=raw, eth3d=decoded, rental_compute_usd=0))


def window_key(job):
    return (job["dataset"], job["scene"], tuple(job["frames"]), job["width"])


def register_locked(five_only, extra_sources):
    initial = verify("initial")
    prepared, decoded = read(PREPARED), read(INPUT_AUDIT)
    initial_digest = sha(INITIAL, text=True)
    for receipt in (prepared, decoded):
        if (receipt.get("study"), receipt.get("registration_sha256"), receipt.get("rental_compute_usd")) != (
                STUDY, initial_digest, 0):
            raise ValueError("Input receipt differs from initial registration")
    windows = initial["planned_windows"]
    signatures = decoded["windows"]
    if len(signatures) != len(windows) or set(map(window_key, signatures)) != set(map(window_key, windows)):
        raise ValueError("Decoded audit differs from the registered five-view windows")
    jobs = locked_jobs(windows)
    actual = hashes_for(jobs)
    for info in prepared["eth3d"].values():
        if any(actual[path] != digest for path, digest in info["hashes"].items()):
            raise ValueError("Prepared ETH3D bytes changed before lock")
    expected_audit = hashes_for(windows)
    if decoded.get("input_source_sha256") != expected_audit or any(
            actual[path] != digest for path, digest in expected_audit.items()):
        raise ValueError("Audited inputs changed before lock")
    authorities = {name: sha(PUBLIC / name, text=True) for name in AUTHORITIES}
    source_hashes = source_pins(initial, [*LOCK_SOURCES, *extra_sources])
    write_new(LOCKED, dict(study=STUDY, registration="locked", **runtime_pins(),
                           initial_sha256=initial_digest, reference=REFERENCE, candidate=CAMERA,
                           five_only=five_only, launch_order_seed=ORDER_SEED, bootstrap_seed=ORDER_SEED,
                           bootstrap_samples=BOOTSTRAP_SAMPLES, margin=MARGIN, max_attempts=2, jobs=jobs,
                           expected_confirmation_signatures=signatures, input_source_sha256=actual,
                           authority_sha256_lf=authorities, preparation=prepared, source_sha256_lf=source_hashes))
    print(f"Registered {len(jobs)} locked cases (84 primary five-view, 6 Courtyard, 18 ten-view)")


def verify(registration):
    plan = read_committed(REGISTRATIONS[registration])
    if (plan["study"], plan["registration"]) != (STUDY, registration):
        raise ValueError("Wrong registered study or phase")
    if any(plan.get(name) != expected for name, expected in runtime_pins().items()):
        raise ValueError("Runtime pins differ from registration")
    if registration == "locked":
        if plan["initial_sha256"] != sha(INITIAL, text=True):
            raise ValueError("Locked roster is detached from the initial registration")
        if any(sha(PUBLIC / name, text=True) != digest for name, digest in plan["authority_sha256_lf"].items()):
            raise ValueError("The diagnosis, qualification or screen authority changed")
    changed = [name for name, digest in plan["source_sha256_lf"].items() if sha(ROOT / name, text=True) != digest]
    if changed:
        raise ValueError(f"Registered sources changed: {changed}")
    identifiers = [job["id"] for job in plan["jobs"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate study case ID")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register-initial", "prepare-inputs", "register-locked", "verify"))
    parser.add_argument("--registration", choices=REGISTRATIONS, default="initial")
    parser.add_argument("--source", action="append", default=[], help="Bind an additional effective source file")
    parser.add_argument("--five-only", action="store_true", help="Record a closed ten-view gate")
    args = parser.parse_args()
    if args.action == "register-initial":
        register_initial(args.source)
    elif args.action == "prepare-inputs":
        preparation()
    elif args.action == "register-locked":
        register_locked(args.five_only, args.source)
    else:
        plan = verify(args.registration)
        print(f"Verified {plan['study']} {plan['registration']} with {len(plan['jobs'])} registered cases")


if __name__ == "__main__":
    main()
