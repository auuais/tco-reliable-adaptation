"""Freeze the local TCO render-lifetime study before launching its jobs."""

import argparse
import random
import subprocess

from run_profiled_d0 import read, sha, write_new
from tco_courtyard_camera_protocol import (
    COURTYARD_FRAMES, COURTYARD_IMAGES, REGISTRATIONS as OLD_REGISTRATIONS,
    TEN_FRAMES, TEN_WINDOWS, hashes_for, prior_frame_exposure,
    verify as verify_previous, window_key,
)
from tco_full40_recompute_protocol import runtime_pins
from tco_one_percent_protocol import CONFIRM_MANIFEST, DATA, ROOT, read_committed


STUDY = "tco_render_lifetime_v1"
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
LOCKED = ROOT / "manifests" / f"{STUDY.removesuffix('_v1')}_locked_v1.json"
REGISTRATIONS = {"initial": INITIAL, "locked": LOCKED}
IMAGE = runtime_pins()["docker_image"]
CPU_PROFILES = {"fixture", "rescore", "input_audit"}
REFERENCE = "native"
CAMERA = "camera"
FIVE_FRAMES = (0, 200, 400, 600, 800)
FIVE_SEEDS = (7, 77, 777)
ORDER_SEED = 20261125
MAIN = ("native", "camera", "n", "g", "ng")
FALLBACK = ("ngd", "camera_d")
SEVEN = {"chess/seq-05": (124, 200), "fire/seq-04": (124, 200),
         "office/seq-06": (124, 200), "pumpkin/seq-07": (124, 200),
         "redkitchen/seq-04": (124, 200), "heads/seq-01": (150, 200),
         "stairs/seq-04": (74, 100)}
ETH = {"delivery_area": 1, "electro": 2, "facade": 27, "kicker": 2,
       "office": 2, "playground": 1, "terrains": 3}
SOURCES = (
    "docs/TCO_RENDER_LIFETIME_HANDOFF.md",
    "scripts/tco_render_lifetime_protocol.py",
    "scripts/run_tco_render_lifetime_local.py",
    "scripts/run_tco_render_lifetime.py",
    "scripts/tco_render_lifetime.py",
    "scripts/check_tco_render_lifetime.py",
)
RELEASE_SOURCES = (
    "scripts/archive_tco_render_lifetime.py",
    "scripts/replay_tco_render_lifetime.py",
    "scripts/test_tco_render_lifetime_replay.py",
)


def job(identifier, cohort, policy="native", *, scene="heads/seq-01", frames=FIVE_FRAMES,
        seed=42, steps=40, dataset="7scenes", width=518, profile=False, score=False,
        probe=None, **extra):
    return dict(id=identifier, cohort=cohort, policy=policy, dataset=dataset, scene=scene,
                frames=list(frames), width=width, seed=seed, steps=steps,
                arm="four", phase="confirmation", profile=profile, score=score, probe=probe, **extra)


def initial_jobs():
    order = random.Random(ORDER_SEED)
    jobs = [job("cpu_fixture", "qualification", profile="fixture", probe="fixture", steps=0),
            job("trace_camera_10", "trace", "camera", steps=2, frames=TEN_FRAMES, profile="timeline"),
            job("screen_camera_1", "screen", "camera", steps=2, frames=TEN_FRAMES, replica=1),
            job("screen_native_1", "screen", steps=2, frames=TEN_FRAMES, replica=1),
            job("renderer_capture", "qualification", width=294, steps=2, probe="capture"),
            job("renderer_replay", "qualification", width=294, steps=0, probe="renderer",
                capture_job="renderer_capture"),
            job("real_native", "qualification", width=294, steps=2, probe="real"),
            job("real_native_2", "qualification", width=294, steps=2, probe="real")]
    for policy in (*MAIN, *FALLBACK):
        stage = "fallback_probe" if policy in FALLBACK else "qualification"
        jobs.extend([
            job(f"identity_{policy}", stage, policy, steps=0, probe="identity"),
            job(f"fixed_{policy}", stage, policy, width=294, steps=2, probe="fixed"),
            job(f"matched_{policy}", stage, policy, width=294, steps=2,
                probe="real", matched_state_from="real_native"),
        ])
        if policy != "native":
            jobs.extend([
                job(f"real_{policy}", stage, policy, width=294, steps=2, probe="real"),
                job(f"leak_{policy}", stage, policy, width=294, steps=2, probe="leak"),
            ])
    for policy in MAIN:
        jobs.append(job(f"trace_{policy}_5", "trace", policy, steps=2, profile="timeline"))
        if policy not in {"native", "camera"}:
            jobs.append(job(f"trace_{policy}_10", "trace", policy, steps=2, frames=TEN_FRAMES, profile="timeline"))
    for replica in (1, 2):
        block = [job(f"screen_{policy}_{replica}", "screen", policy, steps=2,
                     frames=TEN_FRAMES, replica=replica)
                 for policy in MAIN[1:] if (policy, replica) != ("camera", 1)]
        order.shuffle(block)
        jobs.extend(block)
    for policy in FALLBACK:
        jobs.append(job(f"trace_{policy}_10", "fallback", policy, steps=2,
                        frames=TEN_FRAMES, profile="timeline"))
        for replica in (1, 2):
            jobs.append(job(f"screen_{policy}_{replica}", "fallback", policy, steps=2,
                            frames=TEN_FRAMES, replica=replica))
    jobs.extend(job(f"pilot_{policy}", "pilot", policy, frames=TEN_FRAMES)
                for policy in ("n", "g", "ng", "ngd"))
    for room in ("chess", "courtyard"):
        fields = (dict(scene="chess/seq-03") if room == "chess" else
                  dict(scene="courtyard", dataset="eth3d", frames=COURTYARD_FRAMES,
                       images_expected=list(COURTYARD_IMAGES)))
        for seed in FIVE_SEEDS:
            pair = [job(f"repeat_{room}_{seed}_{replica}", "repeat", seed=seed, replica=replica,
                        score=True, **fields) for replica in (1, 2)]
            order.shuffle(pair)
            jobs.extend(pair)
            jobs.extend(job(f"rescore_{item['id']}", "rescore", seed=seed, profile="rescore",
                            probe="rescore", source_job=item["id"], steps=0, **fields) for item in pair)
    return jobs


def confirmation_windows():
    data = DATA / "data"
    eth_images = {scene: catalog(data, scene) for scene in ETH}
    return [
        job(f"window_7scenes_{scene.split('/')[0]}", "confirmation", scene=scene,
            frames=range(start, start + 5 * stride, stride))
        for scene, (start, stride) in SEVEN.items()
    ] + [
        job(f"window_eth3d_{scene}", "confirmation", dataset="eth3d", scene=scene,
            frames=range(start, start + 25, 5),
            images_expected=[eth_images[scene][idx] for idx in range(start, start + 25, 5)])
        for scene, start in ETH.items()
    ]


def catalog(data, scene):
    return sorted(p.name for p in (data / "eth3d" / scene / "images/custom_undistorted").glob("*.JPG"))


def locked_jobs(candidate, windows):
    if candidate is None:
        return []
    order = random.Random(ORDER_SEED)
    jobs = []
    for scene, seeds in TEN_WINDOWS.items():
        room = scene.split("/")[0]
        for index, seed in enumerate(seeds):
            block = []
            for policy in (candidate, "native", "camera"):
                extra = ({"first_gate": f"ten_{room}_{policy}_{seeds[0]}"}
                         if policy != candidate and index else {})
                block.append(job(f"ten_{room}_{policy}_{seed}", "ten", policy, scene=scene,
                                 seed=seed, frames=TEN_FRAMES, score=True, **extra))
            order.shuffle(block)
            jobs.extend(block)
    for window in windows:
        prefix = window["id"].replace("window_", "conf_", 1)
        fields = {key: window[key] for key in ("scene", "dataset", "frames", "width", "images_expected") if key in window}
        for seed in FIVE_SEEDS:
            pair = [job(f"{prefix}_{policy}_{seed}", "confirmation", policy,
                        seed=seed, score=True, **fields) for policy in ("native", candidate)]
            order.shuffle(pair)
            jobs.extend(pair)
    return jobs


def prepare():
    from prepare_tco_one_percent_confirmation import ASSETS, PREP_REVISION, preparation_module, prepare_scene

    helper = ASSETS / f"vggt_omega_prepare_{PREP_REVISION}.py"
    expected = read_committed(CONFIRM_MANIFEST)["eth3d_preparation"]["helper_sha256"]
    if sha(helper) != expected:
        raise ValueError("The offline ETH3D preparation helper differs from its authority")
    for scene in ETH:
        for kind in ("undistorted", "jpg", "depth"):
            name = f"{scene}_dslr_{kind}.7z"
            if not (ASSETS / "archives" / name).is_file() or not (ASSETS / "extracted" / f"{name}.json").is_file():
                raise FileNotFoundError(f"Missing offline archive and extraction receipt for {name}")
    module, _ = preparation_module()
    receipt = {}
    for scene, start in ETH.items():
        receipt[scene] = prepare_scene(scene, range(start, start + 25, 5), module)
        print(f"{scene}: prepared five registered public depth/camera files", flush=True)
    write_new(PRIVATE / "preparation.json", dict(helper_sha256=expected, eth3d=receipt, rental_compute_usd=0))


def register_initial():
    from unittest.mock import patch

    previous = verify_previous("locked")
    current_upstream = subprocess.check_output(["git", "-C", str(ROOT / "third_party/TCO"), "rev-parse", "HEAD"],
                                               text=True).strip()
    pins = runtime_pins()
    if current_upstream != pins["upstream"] or sha(DATA / "checkpoints/VGGT-1B/model.safetensors") != pins["checkpoint_sha256"]:
        raise ValueError("Upstream or checkpoint differs from the prior study")
    jobs, windows = initial_jobs(), confirmation_windows()
    # The older helper excluded its own registrations. Include them and exclude this study instead.
    with patch.dict(OLD_REGISTRATIONS, REGISTRATIONS, clear=True):
        exposure = prior_frame_exposure(windows)
    old_hashes = previous["source_sha256_lf"]
    paths = set(SOURCES) | set(old_hashes)
    paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "patches").glob("*render-lifetime*"))
    hashes = hashes_for([*jobs, *windows, *locked_jobs("n", [])])
    write_new(INITIAL, dict(
        study=STUDY, registration="initial", **pins, reference=REFERENCE, margin=1.01, max_attempts=2,
        bootstrap_seed=ORDER_SEED, bootstrap_samples=10000, launch_order_seed=ORDER_SEED,
        selection_order=["n", "g", "ng", "ngd"], jobs=jobs, planned_windows=windows, exposure=exposure,
        thresholds=dict(loss_atol=1e-6, output_rtol=1e-5, gradient_rtol=1e-4, rescore_mm=0.01),
        prior_sha256=sha(OLD_REGISTRATIONS["locked"], text=True),
        preparation=read(PRIVATE / "preparation.json"),
        input_source_sha256=hashes, source_sha256_lf={name: sha(ROOT / name, text=True) for name in sorted(paths)},
    ))
    print(f"Registered {len(jobs)} initial jobs and {len(windows)} confirmation windows")


def register_lock():
    initial = verify("initial")
    screen = read_committed(PUBLIC / "screen.json")
    candidate = screen["promoted_policy"]
    if candidate not in (*initial["selection_order"], None):
        raise ValueError("Screen yielded an unknown policy")
    if screen.get("pending"):
        raise ValueError("The capacity decision is still open")
    authorities = {name: sha(PUBLIC / name, text=True) for name in ("qualification.json", "screen.json")}
    jobs = locked_jobs(candidate, initial["planned_windows"])
    sources = {**initial["source_sha256_lf"], **{name: sha(ROOT / name, text=True) for name in RELEASE_SOURCES}}
    signatures = []
    if candidate is not None:
        audit_path = PRIVATE / "decoded_inputs.json"
        audit = read(audit_path)
        if (audit["registration_sha256"] != sha(INITIAL, text=True)
                or audit["rental_compute_usd"] != 0
                or audit["input_source_sha256"] != hashes_for(initial["planned_windows"])
                or set(map(window_key, audit["windows"])) != set(map(window_key, initial["planned_windows"]))):
            raise ValueError("The decoded input audit differs from registration")
        if any(initial["input_source_sha256"][key] != value for key, value in audit["input_source_sha256"].items()):
            raise ValueError("Prepared public frame bytes changed after registration")
        signatures = audit["windows"]
    write_new(LOCKED, dict(
        study=STUDY, registration="locked", **runtime_pins(), reference=REFERENCE, candidate=candidate,
        initial_sha256=sha(INITIAL, text=True), authority_sha256_lf=authorities, jobs=jobs,
        source_sha256_lf=sources, input_source_sha256=initial["input_source_sha256"],
        bootstrap_seed=ORDER_SEED, bootstrap_samples=10000, margin=1.01, max_attempts=2,
        launch_order_seed=ORDER_SEED, planned_windows=initial["planned_windows"],
        expected_confirmation_signatures=signatures,
    ))
    print(f"Locked {len(jobs)} capacity and quality jobs for {candidate}")


def verify(registration):
    plan = read_committed(REGISTRATIONS[registration])
    if (plan["study"], plan["registration"]) != (STUDY, registration):
        raise ValueError("Incorrect study or phase in registration")
    if any(plan.get(key) != value for key, value in runtime_pins().items()):
        raise ValueError("Runtime pins differ")
    if registration == "locked":
        if plan["initial_sha256"] != sha(INITIAL, text=True):
            raise ValueError("Lock differs from initial registration")
        if any(sha(PUBLIC / name, text=True) != digest for name, digest in plan["authority_sha256_lf"].items()):
            raise ValueError("Screen or qualification changed after lock")
    if len({item["id"] for item in plan["jobs"]}) != len(plan["jobs"]):
        raise ValueError("Duplicate case IDs")
    changed = [key for key, digest in plan["source_sha256_lf"].items() if sha(ROOT / key, text=True) != digest]
    if changed:
        raise ValueError(f"Frozen files changed: {changed}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "register-initial", "register-locked", "verify"))
    parser.add_argument("--registration", choices=REGISTRATIONS, default="initial")
    args = parser.parse_args()
    if args.command == "verify":
        print(f"Verified {len(verify(args.registration)['jobs'])} {args.registration} jobs")
    else:
        {"prepare": prepare, "register-initial": register_initial, "register-locked": register_lock}[args.command]()


if __name__ == "__main__":
    main()
