"""Register the RTX 3080 five-view experiment and prepare its public inputs."""

import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import random
import subprocess
from unittest.mock import patch

from run_profiled_d0 import read, sha, write_new
from tco_courtyard_camera_protocol import (
    COURTYARD_FRAMES, COURTYARD_IMAGES, REGISTRATIONS as CAMERA_REGISTRATIONS,
    hashes_for, prior_frame_exposure, window_key,
)
from tco_full40_recompute_protocol import runtime_pins
from tco_one_percent_protocol import CONFIRM_MANIFEST, DATA, ROOT, read_committed
from tco_render_lifetime_protocol import catalog


STUDY = "tco_five_view_replay_cache_v1"
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
PHASES = ("initial", "screen", "combination", "development", "locked")
REGISTRATIONS = {stage: INITIAL if stage == "initial" else
                 INITIAL.with_name(f"tco_five_view_replay_cache_{stage}_v1.json") for stage in PHASES}
LOCKED = REGISTRATIONS["locked"]
IMAGE = runtime_pins()["docker_image"]
CPU_PROFILES = {"fixture", "input_audit", "rescore"}
ORDER_SEED = 20261126
DEV_SEEDS = (42, 43, 44)
FIVE_SEEDS = (7, 77, 777)
FIVE_FRAMES = (0, 200, 400, 600, 800)
SEVEN = {"chess/seq-05": (162, 200), "fire/seq-04": (162, 200), "heads/seq-01": (75, 200),
         "office/seq-06": (162, 200), "pumpkin/seq-07": (162, 200),
         "redkitchen/seq-04": (162, 200), "stairs/seq-04": (87, 100)}
ETH = {"delivery_area": 2, "electro": 21, "facade": 3, "kicker": 6, "office": 1,
       "playground": 3, "terrains": 19, "courtyard": 17}


def job(identifier, cohort, policy="native", *, scene="heads/seq-01", frames=FIVE_FRAMES,
        seed=42, steps=40, dataset="7scenes", width=518, profile=False, score=False,
        probe=None, **extra):
    camera = {"native": "native", "camera": "checkpoint", "last": "last", "retention": "native",
              "combined": "last", "uniform": "last", "camera_retention": "checkpoint"}.get(policy, "native")
    result = dict(id=identifier, cohort=cohort, policy=policy, dataset=dataset, scene=scene,
                  frames=list(frames), width=width, seed=seed, steps=steps, camera=camera,
                  retention="original", sites=[], operation="attention", arm="four",
                  phase="confirmation", profile=profile, score=score, probe=probe)
    result.update(extra)
    return result


def confirmation_windows():
    windows = []
    for scene, (start, stride) in SEVEN.items():
        frames = range(start, start + 5 * stride, stride)
        windows.append(job(f"window_7scenes_{scene.split('/')[0]}", "confirmation", scene=scene, frames=frames))
    for scene, start in ETH.items():
        names = catalog(DATA / "data", scene)
        frames = range(start, start + 25, 5)
        windows.append(job(f"window_eth3d_{scene}", "confirmation", dataset="eth3d", scene=scene,
                           frames=frames, images_expected=[names[index] for index in frames]))
    return windows


def development_windows():
    return [job("window_heads", "development"), job("window_chess", "development", scene="chess/seq-03"),
            job("window_courtyard", "development", dataset="eth3d", scene="courtyard",
                frames=COURTYARD_FRAMES, images_expected=list(COURTYARD_IMAGES))]


def rgb_path(dataset, scene, frame, catalogs):
    root = DATA / "data" / dataset / scene
    if dataset == "7scenes":
        return root / f"frame-{frame:06d}.color.png"
    return root / "images/custom_undistorted" / catalogs[scene][frame]


def historical_exposure(windows):
    """Include reserved windows, scored/instrumented raw attempts and exact source-image bytes."""
    root = DATA / "data"
    data = {}
    for seq in (root / "7scenes").glob("*/seq-*"):
        ids = sorted(int(path.name[6:12]) for path in seq.glob("frame-*.color.png"))
        if ids:
            data["7scenes", seq.relative_to(root / "7scenes").as_posix()] = ids
    catalogs = {scene: catalog(root, scene) for scene in ETH}
    data.update({("eth3d", scene): list(range(len(names))) for scene, names in catalogs.items()})
    all_slots = [dict(dataset=dataset, scene=scene, frames=frames) for (dataset, scene), frames in data.items()]
    with patch.dict(CAMERA_REGISTRATIONS, REGISTRATIONS, clear=True):
        exposure = prior_frame_exposure(all_slots)
    occupied = defaultdict(lambda: defaultdict(set))
    for row in exposure["windows"]:
        for hit in row["observed_exact_overlaps"]:
            occupied[row["dataset"], row["scene"]][hit["frame"]].update(hit["sources"])
    raw_files = sorted((ROOT / "reports").glob("*/cases/**/complete.json"))
    receipt_hash = hashlib.sha256()
    receipt_count = 0
    for path in raw_files:
        if PRIVATE in path.parents:
            continue
        row = read(path)
        if not isinstance(row, dict):
            continue
        key = row.get("dataset"), row.get("scene")
        if key not in data or not isinstance(row.get("frames"), list):
            continue
        label = path.relative_to(ROOT).as_posix()
        receipt_hash.update(label.encode() + b"\0" + bytes.fromhex(sha(path)))
        receipt_count += 1
        for idx in row["frames"]:
            if type(idx) is int:
                occupied[key][idx].add(label)
    for field in development_windows():
        for idx in field["frames"]:
            occupied[field["dataset"], field["scene"]][idx].add("current development/diagnostic window")
    old_bytes = defaultdict(set)
    for (dataset, scene), ids in occupied.items():
        for frame in ids:
            path = rgb_path(dataset, scene, frame, catalogs)
            if not path.is_file():
                raise ValueError(f"Historical registered RGB source is unavailable: {path}")
            old_bytes[sha(path)].add((dataset, scene, frame))
    selected = []
    details = []
    selected_bytes = set()
    for window in windows:
        dataset, scene = key = window["dataset"], window["scene"]
        prior_ids = occupied[key]
        available = set(data[key])
        source_hash = {}
        def fresh(ids):
            for idx in ids:
                if idx not in source_hash:
                    source_hash[idx] = sha(rgb_path(dataset, scene, idx, catalogs))
            digests = [source_hash[idx] for idx in ids]
            return (len(set(digests)) == len(ids) and not any(idx in prior_ids for idx in ids)
                    and not any(digest in old_bytes or digest in selected_bytes for digest in digests))
        requested = window["frames"]
        stride = requested[1] - requested[0]
        chosen = requested
        if not fresh(requested):
            possibilities = [tuple(start + i * stride for i in range(5)) for start in sorted(available)]
            possibilities = [ids for ids in possibilities if all(idx in available for idx in ids)]
            def rank(ids):
                near = [min((abs(idx - old) for old in prior_ids), default=0) for idx in ids]
                return (-min(near), -sum(near), ids[0])
            chosen = next((list(ids) for ids in sorted(possibilities, key=rank) if fresh(ids)), [])
        details.append(dict(dataset=dataset, scene=scene, requested_frames=requested, selected_frames=chosen,
                            selected_rgb_sha256=[source_hash[idx] for idx in chosen],
                            known_indices_checked=len(prior_ids)))
        if chosen:
            output = dict(window, frames=chosen)
            if dataset == "eth3d":
                output["images_expected"] = [catalogs[scene][idx] for idx in chosen]
            selected.append(output)
            selected_bytes.update(source_hash[idx] for idx in chosen)
    return selected, dict(raw_receipt_count=receipt_count, raw_receipt_path_content_sha256=receipt_hash.hexdigest(),
                          previous_materials=exposure["authorities"], known_rgb_hash_count=len(old_bytes),
                          windows=details, method="Known historical indices plus exact original RGB SHA-256, "
                          "including prepared and gated registrations and current development/diagnostic views. "
                          "The physical rooms were studied before; adjacent images may depict the same surfaces.")


def window_fields(window):
    return {key: window[key] for key in ("scene", "dataset", "frames", "width", "images_expected")
            if key in window}


def locked_jobs(candidate, windows, configuration=None):
    if candidate is None:
        return []
    order = random.Random(ORDER_SEED)
    jobs = []
    for window in windows:
        for index, seed in enumerate(FIVE_SEEDS):
            rows = [job(f"conf_{window['id'][7:]}_{policy}_{seed}", "confirmation", policy, seed=seed, score=True,
                        **window_fields(window), **(configuration if policy == candidate else {}))
                    for policy in ("native", candidate)]
            if index == 0:
                reverse_first = order.randrange(2)
            if (index + reverse_first) % 2:
                rows.reverse()
            jobs.extend(rows)
    for dataset in ("7scenes", "eth3d"):
        first = next((item for item in windows if item["dataset"] == dataset), None)
        if first is None:
            continue
        for item in [row for row in jobs if row["cohort"] == "confirmation"
                     and row["seed"] == FIVE_SEEDS[0] and window_key(row) == window_key(first)]:
            jobs.append(job(f"rescore_{item['id']}", "rescore", item["policy"], **window_fields(item),
                            seed=item["seed"], steps=0, score=False, profile="rescore", probe="rescore",
                            source_job=item["id"]))
    return jobs


def persist_once(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Existing evidence differs: {path}")
    else:
        write_new(path, payload)


def prepare():
    """Keep canonical older inputs, prepare unique frames in a study-owned directory."""
    import prepare_tco_one_percent_confirmation as prep

    helper = prep.ASSETS / f"vggt_omega_prepare_{prep.PREP_REVISION}.py"
    expected = read_committed(CONFIRM_MANIFEST)["eth3d_preparation"]["helper_sha256"]
    if not helper.is_file() or sha(helper) != expected:
        raise ValueError("Missing or changed cached ETH3D preparation helper")
    staged = DATA / "preparation" / STUDY
    receipts = {}
    for scene in ETH:
        for kind in ("undistorted", "jpg", "depth"):
            name = f"{scene}_dslr_{kind}.7z"
            if not (prep.ASSETS / "archives" / name).is_file() or not (prep.ASSETS / "extracted" / f"{name}.json").is_file():
                raise FileNotFoundError(f"Missing offline ETH3D archive or extraction receipt for {name}")
    module, _ = prep.preparation_module()
    windows, exposure = historical_exposure(confirmation_windows())
    persist_once(PRIVATE / "exposure.json", exposure)
    persist_once(PRIVATE / "windows.json", windows)
    for window in windows:
        if window["dataset"] != "eth3d":
            continue
        scene = window["scene"]
        with patch.object(prep, "DATA", staged), patch.object(prep, "ETH3D_ROOT", staged / "data/eth3d"):
            receipt = prep.prepare_scene(scene, window["frames"], module)
        if receipt["filenames"] != window["images_expected"]:
            raise ValueError(f"Unexpected ETH3D lexical frame selection at {scene}")
        for relative, digest in receipt["hashes"].items():
            src = staged / "data" / relative
            dst = DATA / "data" / relative
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
            if sha(dst) != digest:
                raise ValueError(f"Previously prepared canonical input differs at {relative}")
        receipts[scene] = receipt
        print(f"Verified five public camera, RGB and depth inputs for {scene}", flush=True)
    inputs = hashes_for(windows + development_windows())
    persist_once(PRIVATE / "preparation.json", dict(study=STUDY, source_helper_sha256=expected,
                 eth3d=receipts, input_source_sha256=inputs, rental_compute_usd=0))


def gradient_jobs():
    jobs = []
    def grad(identifier, mode, backend, **extra):
        return job(identifier, "gradient", width=294, steps=2, profile=False, probe="real",
                   backend=backend, gradient_mode=mode, **extra)
    anchor = "grad_production_capture"
    jobs.append(grad(anchor, "capture", "production"))
    jobs.extend(grad(f"grad_production_control_{i}", "control", "production", replica=i) for i in (1, 2))
    jobs.extend(grad(f"grad_production_regenerate_{i}", "regenerate", "production", source_job=anchor,
                     q_index=i, replica=i) for i in (1, 2, 3))
    for backend in ("production", "deterministic"):
        for idx in (1, 2, 3):
            jobs.append(grad(f"grad_{backend}_fixed_{idx}", "replay", backend,
                             source_job=anchor, replica=idx))
            jobs.append(grad(f"grad_{backend}_vary_{idx}", "replay", backend,
                             source_job=anchor, q_source_job=f"grad_production_regenerate_{idx}", q_index=idx))
    jobs.append(grad("grad_production_sign", "negative", "production", source_job=anchor))
    fallback = "grad_deterministic_capture"
    def alternate(identifier, mode, **extra):
        return grad(identifier, mode, "deterministic", conditional="deterministic_forward_anchor_mismatch", **extra)
    jobs.append(alternate(fallback, "capture"))
    for idx in (1, 2, 3):
        regenerated = f"grad_deterministic_regenerate_{idx}"
        jobs.append(alternate(regenerated, "regenerate", source_job=fallback, q_index=idx, replica=idx))
        jobs.append(alternate(f"grad_deterministic_fallback_fixed_{idx}", "replay", source_job=fallback, replica=idx))
        jobs.append(alternate(f"grad_deterministic_fallback_vary_{idx}", "replay", source_job=fallback,
                              q_source_job=regenerated, q_index=idx))
    return jobs


def initial_jobs():
    jobs = gradient_jobs()
    jobs.append(job("cpu_fixture", "fixture", steps=0, probe="camera_fixture", profile="fixture"))
    jobs.append(job("gpu_fixture", "fixture", steps=0, probe="camera_fixture_gpu"))
    for policy in ("native", "camera", "last"):
        jobs.append(job(f"identity_{policy}", "qualification", policy, probe="identity", steps=0))
        jobs.append(job(f"leak_{policy}", "qualification", policy, probe="reference_substitution", steps=0))
        jobs.append(job(f"matched_{policy}", "qualification", policy, probe="real", width=294, steps=2,
                        gradient_mode="replay", backend="production", source_job="grad_production_capture"))
    order = random.Random(ORDER_SEED)
    for window in (development_windows()[0], development_windows()[2]):
        room = window["scene"].split("/")[0]
        for tag, retention in (("native", "original"), ("zero", "zero")):
            jobs.append(job(f"profile_{room}_{tag}", "profile", profile="timeline", diagnostic=True,
                            retention=retention, steps=40, **window_fields(window)))
        for replica in (1, 2):
            block = [job(f"baseline_{room}_{policy}_{replica}", "baseline", policy, steps=2,
                         replica=replica, **window_fields(window)) for policy in ("native", "camera", "last")]
            order.shuffle(block)
            jobs.extend(block)
    return jobs


def audit_inside():
    import tempfile
    import numpy as np
    import torch
    from run_tco_one_percent import DATA_ROOT, load_case, signature

    if torch.cuda.is_available():
        raise RuntimeError("Input decoder must run with GPU devices disabled")
    rows = []
    windows = read(PRIVATE / "windows.json") + development_windows()
    for window in windows:
        with tempfile.TemporaryDirectory() as out:
            supplied, reference = load_case(window, Path(out), window["width"])
        arrays = {"rgb": supplied["images"].numpy(), "poses": supplied["extrs"], "intrinsics": supplied["intrs"]}
        surface = {"mask": reference["valid_mask"], "surface": reference["pointclouds"]}
        counts = np.count_nonzero(surface["mask"], axis=(1, 2))
        if (len(counts) != 5 or not (counts > 0).all() or any(not np.isfinite(arr).all() for arr in arrays.values())
                or not np.isfinite(surface["surface"][surface["mask"]]).all()):
            raise ValueError(f"Invalid decoded public input at {window['scene']}")
        rows.append(dict(**window_fields(window), input_sha256=signature(arrays), reference_sha256=signature(surface),
                         raster_shape=list(arrays["rgb"].shape), valid_points=int(counts.sum()),
                         valid_points_by_view=counts.tolist()))
        print(f"{window['dataset']} {window['scene']}: {counts.tolist()} valid pixels", flush=True)
    persist_once(PRIVATE / "decoded_inputs.json", dict(study=STUDY, windows=rows, rental_compute_usd=0,
                 input_source_sha256=hashes_for(windows, DATA_ROOT)))


def audit_host():
    from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped

    command = [*DOCKER, "create", "--pull", "never", "--network", "none", "--shm-size", "2g",
                    "--mount", f"type=bind,src={ROOT.as_posix()},dst=/workspace", "--workdir", "/workspace",
                    "--mount", f"type=bind,src={DATA.as_posix()},dst=/data,readonly",
                    "--env", "OMP_NUM_THREADS=4", "--env", "OPENBLAS_NUM_THREADS=4",
                    "--env", "MKL_NUM_THREADS=4", "--env", "HF_HUB_OFFLINE=1", IMAGE, "python", "-B",
                    "-u", "scripts/tco_five_view_replay_cache_protocol.py", "audit-inside"]
    container = subprocess.check_output(command, text=True, timeout=60).strip()
    try:
        snapshot, = docker_json(["inspect", container])
        fields = dict(image=snapshot["Image"], network=snapshot["HostConfig"]["NetworkMode"],
                      device_requests=snapshot["HostConfig"].get("DeviceRequests") or [],
                      data_read_only=not next(m["RW"] for m in snapshot["Mounts"] if m["Destination"] == "/data"))
        if fields != dict(image=IMAGE, network="none", device_requests=[], data_read_only=True):
            raise ValueError("Expected pinned offline CPU decoder with read-only data")
        persist_once(PRIVATE / "audit_runtime.json", fields)
        subprocess.run([*DOCKER, "start", "--attach", container], check=True, timeout=1200)
    finally:
        final, errors = ensure_stopped(container)
        if final is None or final["State"]["Running"]:
            raise RuntimeError(f"Could not observe the CPU audit container stopped: {errors}")
        subprocess.run([*DOCKER, "rm", container], check=True, timeout=30)


def validate_jobs(jobs):
    if len({row["id"] for row in jobs}) != len(jobs):
        raise ValueError("Duplicate scheduled job ID")
    for row in jobs:
        if len(row["frames"]) != 5 or len(set(row["frames"])) != 5:
            raise ValueError(f"Expected five distinct frames at {row['id']}")
        if row["camera"] not in ("native", "checkpoint", "last") or row["retention"] not in ("original", "zero", "op", "block"):
            raise ValueError(f"Invalid forward implementation at {row['id']}")
        if row["steps"] not in (0, 2, 40) or row["width"] not in (294, 518):
            raise ValueError(f"Unsupported number of TCO updates or raster in {row['id']}")


def initial_registration():
    from tco_render_lifetime_protocol import verify as previous_verify

    parent = previous_verify("locked")
    observed = subprocess.check_output(["git", "-C", str(ROOT / "third_party/TCO"), "rev-parse", "HEAD"],
                                       text=True).strip()
    pins = runtime_pins()
    if observed != pins["upstream"] or sha(DATA / "checkpoints/VGGT-1B/model.safetensors") != pins["checkpoint_sha256"]:
        raise ValueError("Upstream model or safetensors differs from the registered release")
    windows = read(PRIVATE / "windows.json")
    decoded = read(PRIVATE / "decoded_inputs.json")
    prepared = read(PRIVATE / "preparation.json")
    exposure = read(PRIVATE / "exposure.json")
    inputs = hashes_for(windows + development_windows())
    if inputs != decoded["input_source_sha256"] or inputs != prepared["input_source_sha256"]:
        raise ValueError("Raw input bytes changed after caching or CPU decoding")
    if set(map(window_key, windows + development_windows())) != set(map(window_key, decoded["windows"])):
        raise ValueError("Decoded signatures do not cover the registered windows")
    jobs = initial_jobs()
    validate_jobs(jobs)
    sources = set(parent["source_sha256_lf"])
    sources.add("docs/TCO_FIVE_VIEW_REPLAY_CACHE_HANDOFF.md")
    sources.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "scripts").glob("*tco_five_view*.py"))
    for required in ("scripts/run_tco_five_view_replay_cache.py", "scripts/run_tco_five_view_replay_cache_local.py",
                     "scripts/check_tco_five_view_replay_cache.py"):
        if required not in sources:
            raise FileNotFoundError(f"Expected source before GPU registration: {required}")
    plan = dict(study=STUDY, registration="initial", **pins, max_attempts=2, reference="native",
                launch_order_seed=ORDER_SEED, bootstrap_seed=ORDER_SEED, bootstrap_samples=10000, margin=1.01,
                development_seeds=list(DEV_SEEDS), confirmation_seeds=list(FIVE_SEEDS),
                requested_confirmation_windows=confirmation_windows(), planned_windows=windows,
                development_windows=development_windows(), historical_exposure=exposure,
                preparation=prepared, cpu_input_audit_runtime=read(PRIVATE / "audit_runtime.json"),
                decoded_windows=decoded["windows"], input_source_sha256=inputs,
                expected_confirmation_signatures=[row for row in decoded["windows"] if window_key(row)
                                                  in set(map(window_key, windows))],
                source_sha256_lf={name: sha(ROOT / name, text=True) for name in sorted(sources)},
                jobs=jobs, budget=dict(runpod_usd=0, only_gpu="NVIDIA GeForce RTX 3080", views=5),
                thresholds=dict(loss_atol=1e-6, output_rtol=1e-5, gradient_rtol=1e-4, rescore_mm=0.01,
                                uniform_peak_difference_MiB=64, primary=1.01, secondary=[1.03, 1.05]),
                intended_development_policies=["native", "camera", "last", "retention", "combined", "uniform"],
                head=dict(tap_indices=[4, 11, 17, 23], stage_pairs=24, adapter_tensors=384, rank=4,
                          alpha=16, dropout=0, adam_lr=0.0003, clip=1, depth_batch=[5]),
                stats=dict(primary="hierarchical paired physical-scene and seed resampling within dataset",
                           ratio="candidate/native three-seed median per room and arithmetic equal-room mean",
                           pass_condition="estimate and 95th percentile <=1.01 independently on both metrics per dataset"),
                timing="Fresh CPU-resident model/inputs through placement, 40 real updates, final FP32 unprojection. "
                       "GPU-synchronized boundaries. Score and instrumentation outside the timer.")
    persist_once(INITIAL, plan)
    return plan


def register_stage(stage):
    if stage == "initial":
        return initial_registration()
    prior_stage = PHASES[PHASES.index(stage) - 1]
    prior = verify(prior_stage)
    decision_path = PUBLIC / f"{stage}_plan.json"
    decision = read_committed(decision_path)
    parent_sha = sha(REGISTRATIONS[prior_stage], text=True)
    if decision.get("study") != STUDY or decision.get("parent_registration_sha256") != parent_sha:
        raise ValueError(f"{stage} decision is not bound to {prior_stage} registration")
    authority = {decision_path.relative_to(ROOT).as_posix(): sha(decision_path, text=True)}
    for name, digest in decision.get("authority_sha256_lf", {}).items():
        path = PUBLIC / name if Path(name).name == name else ROOT / name
        read_committed(path)
        if sha(path, text=True) != digest:
            raise ValueError(f"Stale decision authority {name}")
        authority[path.relative_to(ROOT).as_posix()] = digest
    if stage == "locked":
        candidate = decision["candidate"]
        configuration = decision.get("configuration", {})
        if candidate is not None and ("camera" not in configuration or "retention" not in configuration
                                      or "sites" not in configuration or "operation" not in configuration):
            raise ValueError("Incomplete locked candidate configuration")
        jobs = locked_jobs(candidate, prior["planned_windows"], configuration)
    else:
        candidate, configuration = None, None
        jobs = decision["jobs"]
    validate_jobs(jobs)
    plan = {key: prior[key] for key in ("study", "reference", "max_attempts", "margin", "source_sha256_lf",
            "launch_order_seed", "bootstrap_seed", "bootstrap_samples", "input_source_sha256", "planned_windows",
            "development_windows", "expected_confirmation_signatures", "decoded_windows", "thresholds")}
    plan.update(runtime_pins(), registration=stage, previous_registration_sha256=parent_sha,
                initial_sha256=sha(INITIAL, text=True), authority_sha256_lf=authority,
                jobs=jobs, candidate=candidate, configuration=configuration)
    persist_once(REGISTRATIONS[stage], plan)
    return plan


def verify(registration):
    plan = read_committed(REGISTRATIONS[registration])
    if plan["study"] != STUDY or plan["registration"] != registration:
        raise ValueError("Incorrect study or registration")
    if any(plan.get(name) != value for name, value in runtime_pins().items()):
        raise ValueError("Runtime pin mismatch")
    validate_jobs(plan["jobs"])
    if registration != "initial":
        parent = REGISTRATIONS[PHASES[PHASES.index(registration) - 1]]
        if (sha(parent, text=True) != plan["previous_registration_sha256"]
                or sha(INITIAL, text=True) != plan["initial_sha256"]):
            raise ValueError("Registration ancestry changed")
    for name, digest in {**plan["source_sha256_lf"], **plan.get("authority_sha256_lf", {})}.items():
        if sha(ROOT / name, text=True) != digest:
            raise ValueError(f"Changed registered source or evidence: {name}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "audit", "audit-inside", "verify",
                                          *[f"register-{stage}" for stage in PHASES]])
    parser.add_argument("--registration", default="initial", choices=PHASES)
    args = parser.parse_args()
    if args.action == "verify":
        plan = verify(args.registration)
    elif args.action.startswith("register-"):
        plan = register_stage(args.action.removeprefix("register-"))
    else:
        {"prepare": prepare, "audit-inside": audit_inside, "audit": audit_host}[args.action]()
        return
    print(f"{plan['registration']}: {len(plan['jobs'])} registered jobs")


if __name__ == "__main__":
    main()
