"""Register benchmark inputs and immutable rosters for full TCO recomputation."""

import argparse
from collections import defaultdict
import random
import subprocess

from run_profiled_d0 import read, sha, write_new
from tco_frozen_prefix_protocol import CHECKPOINT, FRAMES, IMAGE, ROOT, SCENES, TCO
from tco_one_percent_protocol import ALLOCATOR, CONFIRM_MANIFEST, DATA, GPU_MEMORY_FRACTION
from tco_one_percent_protocol import MANIFEST as PRIOR_MANIFEST
from tco_one_percent_protocol import PUBLIC as PRIOR_PUBLIC
from tco_one_percent_protocol import file_hashes, read_committed, verify as verify_prior


STUDY = "tco_full40_recompute_v1"
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
LOCKED = ROOT / "manifests/tco_full40_recompute_locked_v1.json"
REGISTRATIONS = {"initial": INITIAL, "locked": LOCKED}
HANDOFF = "docs/TCO_FULL40_RECOMPUTE_HANDOFF.md"
NEW_SOURCES = (
    HANDOFF,
    "scripts/tco_full40_recompute.py",
    "scripts/run_tco_full40_recompute.py",
    "scripts/run_tco_full40_recompute_local.py",
    "scripts/tco_full40_recompute_protocol.py",
)
TEN_FRAMES = tuple(range(0, 1000, 100))
TEN_SCENES = (SCENES[0], "office/seq-02", "pumpkin/seq-01")
REFERENCE = "c8_off"
SEEDS = (42, 43, 44)
CONFIRM_SEEDS = (7, 77, 777)
POLICIES = tuple(f"c{chunk}_{mode}" for chunk in (8, 4, 2, 1) for mode in ("off", "on"))
OFF_POLICIES = tuple(policy for policy in POLICIES if policy.endswith("_off"))
ORDER_SEED = 20261124
TIMEOUT = 1200


def case(identifier, cohort, policy=REFERENCE, *, scene=SCENES[0], frames=FRAMES, steps=40, seed=42,
         dataset="7scenes", width=518, profile=False, score=False, probe=None):
    job = dict(id=identifier, cohort=cohort, policy=policy, dataset=dataset, scene=scene,
               frames=list(frames), steps=steps, seed=seed, width=width, arm="four",
               phase="confirmation", profile=profile, score=score)
    if probe is not None:
        job["probe"] = probe
    return job


def initial_jobs():
    jobs = [
        case("reference_5", "reference"),
        case("reference_10", "reference", frames=TEN_FRAMES),
        case("profile_5", "profile", profile="timeline"),
        case("profile_10", "profile", frames=TEN_FRAMES, profile="timeline"),
        case("head_fixture", "fixture", steps=0, profile="fixture"),
    ]
    for policy in POLICIES:
        jobs.extend((
            case(f"screen_{policy}", "screen", policy, frames=TEN_FRAMES, steps=2),
            case(f"screen5_{policy}", "screen5", policy, steps=2),
            case(f"fixed_{policy}", "qualification", policy, steps=2, width=294, probe="fixed"),
            case(f"leak_{policy}", "qualification", policy, steps=2, width=294, probe="leak"),
            case(f"identity_{policy}", "qualification", policy, steps=0, probe="identity"),
        ))
        jobs.extend(case(f"promote_{policy}_r{repeat}", "resources", policy, frames=TEN_FRAMES)
                    for repeat in (1, 2, 3))
    jobs.extend(case(f"repeat_{scene.split('/')[0]}_r{repeat}", "repeat", scene=scene, score=True)
                for scene in SCENES[:2] for repeat in (1, 2))
    return jobs


def hashes_for(jobs, prior_confirmation):
    captures = defaultdict(set)
    for job in jobs:
        if job["dataset"] == "7scenes":
            captures[job["scene"]].update(job["frames"])
    hashes = file_hashes(DATA / "data", {scene: sorted(frames) for scene, frames in sorted(captures.items())})
    previous = prior_confirmation["input_source_sha256"]
    eth_scenes = {job["scene"] for job in jobs if job["dataset"] == "eth3d"}
    hashes.update({name: digest for name, digest in previous.items()
                   if any(name.startswith(f"eth3d/{scene}/") for scene in eth_scenes)})
    for job in jobs:
        prefix = f"{job['dataset']}/{job['scene']}/"
        if not any(name.startswith(prefix) for name in hashes):
            raise ValueError(f"Missing input hashes for {job['id']}")
    differing = [name for name in hashes.keys() & previous.keys() if hashes[name] != previous[name]]
    if differing:
        raise ValueError(f"Inputs changed since the prior benchmark: {differing}")
    return hashes


def runtime_pins():
    return dict(upstream=TCO, checkpoint_sha256=CHECKPOINT, docker_image=IMAGE,
                allocator=ALLOCATOR, gpu_memory_fraction=GPU_MEMORY_FRACTION,
                gpu="NVIDIA GeForce RTX 3080", launch_timeout_seconds=TIMEOUT, rental_compute_usd=0)


def register_initial():
    prior = verify_prior()
    observed = subprocess.check_output(["git", "-C", str(ROOT / "third_party/TCO"), "rev-parse", "HEAD"],
                                       text=True).strip()
    if observed != TCO or sha(DATA / "checkpoints/VGGT-1B/model.safetensors") != CHECKPOINT:
        raise ValueError("The upstream revision or local checkpoint differs from the prior benchmark")
    previous = read_committed(CONFIRM_MANIFEST)
    sources = {name: sha(ROOT / name, text=True) for name in (*prior["source_sha256_lf"], *NEW_SOURCES)}
    jobs = initial_jobs()
    write_new(INITIAL, dict(study=STUDY, registration="initial", **runtime_pins(),
                            previous_sha256=sha(PRIOR_MANIFEST, text=True), reference=REFERENCE,
                            jobs=jobs, input_source_sha256=hashes_for(jobs, previous),
                            source_sha256_lf=sources))
    print(f"Registered {len(jobs)} engineering and resource cases in {INITIAL.name}")


def lock_jobs(candidate, chunk_control, prior_confirmation):
    ten_policies = [candidate, *OFF_POLICIES]
    jobs = []
    order = random.Random(ORDER_SEED)
    for scene_index, scene in enumerate(TEN_SCENES):
        seeds = SEEDS if scene_index == 0 else CONFIRM_SEEDS
        room = scene.split("/")[0]
        for seed_index, seed in enumerate(seeds):
            block = []
            for policy in ten_policies:
                job = case(f"ten_{room}_{policy}_{seed}", "ten", policy, scene=scene,
                           frames=TEN_FRAMES, seed=seed, score=True)
                if seed_index and policy in OFF_POLICIES:
                    job["first_gate"] = f"ten_{room}_{policy}_{seeds[0]}"
                block.append(job)
            order.shuffle(block)
            jobs.extend(block)
    for scene in SCENES:
        room = scene.split("/")[0]
        for seed in SEEDS:
            block = [case(f"dev_{room}_{policy}_{seed}", "development", policy, scene=scene, seed=seed,
                          score=True) for policy in dict.fromkeys((REFERENCE, candidate, chunk_control))]
            order.shuffle(block)
            jobs.extend(block)
    previously_registered = {(job["dataset"], job["scene"]): job for job in prior_confirmation["jobs"]}
    for (dataset, scene), original in sorted(previously_registered.items()):
        room = scene.split("/")[0]
        for seed in CONFIRM_SEEDS:
            block = []
            for policy in (REFERENCE, candidate):
                job = case(f"conf_{dataset}_{room}_{policy}_{seed}", "confirmation", policy, scene=scene,
                           dataset=dataset, frames=original["frames"], seed=seed, score=True)
                if "images_expected" in original:
                    job["images_expected"] = original["images_expected"]
                block.append(job)
            order.shuffle(block)
            jobs.extend(block)
    return jobs, ten_policies


def register_locked(candidate, chunk_control, best_off, five_only):
    initial = verify("initial")
    if candidate not in POLICIES or not candidate.endswith("_on"):
        raise ValueError("Select a registered recomputation-on policy")
    if chunk_control != candidate.replace("_on", "_off") or best_off not in OFF_POLICIES:
        raise ValueError("Use a same-chunk control and an off policy for the resource frontier")
    previous = read_committed(CONFIRM_MANIFEST)
    signatures = read_committed(PRIOR_PUBLIC / "confirmation_inputs.json")
    if signatures["confirmation_sha256"] != sha(CONFIRM_MANIFEST, text=True):
        raise ValueError("Previous decoded inputs differ from their registration")
    jobs, ten_policies = lock_jobs(candidate, chunk_control, previous)
    source_hashes = initial["source_sha256_lf"] | {
        "scripts/replay_tco_full40_recompute.py": sha(ROOT / "scripts/replay_tco_full40_recompute.py", text=True),
    }
    write_new(LOCKED, dict(study=STUDY, registration="locked", **runtime_pins(),
                           initial_sha256=sha(INITIAL, text=True), candidate=candidate, reference=REFERENCE,
                           chunk_control=chunk_control, best_off=best_off, ten_policies=ten_policies,
                           five_only=five_only, launch_order_seed=ORDER_SEED, bootstrap_seed=ORDER_SEED,
                           bootstrap_samples=10_000, margin=1.01, jobs=jobs,
                           expected_confirmation_signatures=signatures["scenes"],
                           input_source_sha256=hashes_for(jobs, previous), source_sha256_lf=source_hashes))
    print(f"Registered {len(jobs)} locked cases in {LOCKED.name}, including conditional ten-view controls")


def verify(registration):
    plan = read_committed(REGISTRATIONS[registration])
    if plan["study"] != STUDY or plan["registration"] != registration:
        raise ValueError("Wrong registered study or phase")
    if any(plan.get(name) != expected for name, expected in runtime_pins().items()):
        raise ValueError("Runtime pins differ from registration")
    if registration == "locked" and plan["initial_sha256"] != sha(INITIAL, text=True):
        raise ValueError("Locked roster is detached from initial registration")
    changed = [name for name, digest in plan["source_sha256_lf"].items() if sha(ROOT / name, text=True) != digest]
    if changed:
        raise ValueError(f"Registered sources changed: {changed}")
    identifiers = [job["id"] for job in plan["jobs"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate study case ID")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register-initial", "register-locked", "verify"))
    parser.add_argument("--registration", choices=REGISTRATIONS, default="initial")
    parser.add_argument("--candidate")
    parser.add_argument("--chunk-control")
    parser.add_argument("--best-off", choices=OFF_POLICIES)
    parser.add_argument("--five-only", action="store_true", help="Register scientific five-view runs after a ten-view failure")
    args = parser.parse_args()
    if args.action == "register-initial":
        register_initial()
    elif args.action == "register-locked":
        if not args.candidate or not args.chunk_control:
            parser.error("Required: --candidate and --chunk-control")
        register_locked(args.candidate, args.chunk_control, args.best_off or args.chunk_control, args.five_only)
    else:
        plan = verify(args.registration)
        print(f"Verified {plan['study']} {plan['registration']} with {len(plan['jobs'])} registered cases")


if __name__ == "__main__":
    main()
