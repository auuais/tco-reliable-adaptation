"""Register jobs and bind local receipts for the TCO one-percent study."""

import argparse
from pathlib import Path
import random
import subprocess

from run_profiled_d0 import read, sha, write_new
from tco_frozen_prefix_protocol import CHECKPOINT, FRAMES, IMAGE, ROOT, SCENES, SOURCES, TCO


STUDY = "tco_one_percent_v1"
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
MANIFEST = ROOT / "manifests" / f"{STUDY}.json"
TIME_MANIFEST = ROOT / "manifests" / "tco_one_percent_time_v1.json"
CONFIRM_MANIFEST = ROOT / "manifests" / "tco_one_percent_confirmation_v1.json"
DATA = Path("E:/ResearchData/TCO")
METADATA = ROOT / "environment" / "development_dataset_cache_v7.npy"
ALLOCATOR = "garbage_collection_threshold:0.75"
THRESHOLDS = dict(scene_seconds=0.50, peak_allocated=0.80,
                  accuracy_mean_m=1.01, completeness_mean_m=1.01)
SEEDS = (42, 43, 44)
ORDER_SEED = 20261123
ETH3D = ("courtyard", "delivery_area", "electro", "facade", "kicker", "office", "playground", "terrains")
QUALIFICATION = (
    ("q_zero_release", "zero", 0, "identity"),
    ("q_four_release", "four", 0, "identity"),
    ("q_plain_fixed", "cached12", 1, "fixed"),
    ("q_geo_fixed", "geometry", 1, "fixed"),
    ("q_leak_fixed", "geometry", 1, "leak"),
    ("q_last14_fixed", "cached14", 1, "fixed"),
    ("q_un14_fixed", "uncached14", 1, "fixed"),
    ("q_geo_real_a", "geometry", 2, "repeat"),
    ("q_geo_real_b", "geometry", 2, "repeat"),
    ("q_full40_peak", "four", 40, "memory"),
    ("q_last14_peak", "cached14", 10, "memory"),
)
FAMILIES = {
    "fixed": (("zero", 0), ("four", 5), ("four", 10), ("four", 40),
              ("cached12", 10), ("cached12", 15), ("cached12", 20), ("cached14", 10)),
    "boundary": (("own", 10), ("geometry", 10), ("shuffle", 10)),
}
SOURCES = (*SOURCES, "scripts/tco_one_percent_geometry.py", "scripts/tco_one_percent_protocol.py",
           "scripts/run_tco_one_percent.py", "scripts/run_tco_one_percent_local.py",
           "scripts/check_tco_one_percent.py", "docs/TCO_ONE_PERCENT_HANDOFF.md",
           "third_party/TCO/datasets/sevenscenes.py", "third_party/TCO/datasets/eth3d.py",
           "third_party/TCO/datasets/utils/cropping.py",
           "third_party/TCO/evaluation/mv_recon/utils.py",
           "third_party/TCO/datasets/seq-id-maps/ETH3D_mv-recon_seq-id-map-kf5.json")


def read_committed(path):
    committed = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"])
    if committed.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError(f"Registration or qualification is different from Git: {path.name}")
    return read(path)


def job_for(scene, arm, steps, seed, phase, dataset="7scenes", frames=FRAMES, *, name=None, probe=None):
    scene_id = scene.replace("/", "_").replace("-", "")
    prefix = {"fixed": "", "boundary": "", "timing": "t_", "equal_time": "e_", "confirmation": f"p_{dataset}_"}
    identifier = name or f"{prefix[phase]}{scene_id}_{arm}_{steps:02d}_{seed}"
    job = dict(id=identifier, dataset=dataset, scene=scene, frames=list(frames), arm=arm, steps=steps,
               seed=seed, phase=phase)
    if probe is not None:
        job["probe"] = probe
    return job


def development_jobs():
    jobs = []
    order = random.Random(ORDER_SEED)
    for phase, rooms in (("fixed", SCENES), ("boundary", SCENES[:2]), ("boundary", SCENES[2:])):
        for seed in SEEDS:
            block = [job_for(scene, arm, steps, seed, phase)
                     for scene in rooms for arm, steps in FAMILIES[phase]]
            order.shuffle(block)
            jobs.extend(block)
    return jobs


def file_hashes(root, scenes):
    hashes = {}
    for scene, frames in scenes.items():
        for frame in frames:
            for kind in ("color.png", "depth.proj.png", "pose.txt"):
                key = f"7scenes/{scene}/frame-{frame:06d}.{kind}"
                hashes[key] = sha(root / key)
    return hashes


def design():
    return dict(study=STUDY, rental_compute_usd=0, upstream=TCO, checkpoint_sha256=CHECKPOINT,
                docker_image=IMAGE, gpu="NVIDIA GeForce RTX 3080", allocator=ALLOCATOR,
                launch_timeout_seconds=1200, launch_order_seed=ORDER_SEED, thresholds=THRESHOLDS,
                suffix=dict(rank=4, alpha=16, dropout=0, learning_rate=0.0003, max_grad_norm=1.0),
                boundary=dict(rank=4, alpha=4, stage=11, relative_depth_tolerance=0.05,
                              reverse_error_patch_units=0.5, min_pair_baseline=1e-6),
                fixed_adjoint_width=294,
                physical_rooms={scene: scene.split("/")[0] for scene in SCENES},
                confirmation=dict(seeds=[7, 77, 777], eth3d_scenes=list(ETH3D), eth3d_frames=[0, 5, 10, 15, 20],
                                  bootstrap_seed=ORDER_SEED, bootstrap_samples=10000),
                timer="Synchronized CPU-ready model and supplied inputs through final FP32 point map",
                probes=[job_for(SCENES[0], arm, steps, SEEDS[0], "qualification", name=name, probe=probe)
                        for name, arm, steps, probe in QUALIFICATION],
                jobs=development_jobs(),
                metadata_sha256=sha(METADATA),
                source_sha256_lf={file: sha(ROOT / file, text=True) for file in SOURCES})


def verify():
    plan = read_committed(MANIFEST)
    if {key: val for key, val in plan.items() if key != "input_source_sha256"} != design():
        raise ValueError("The local sources or design differ from registration")
    return plan


def register():
    if sha(DATA / "checkpoints/VGGT-1B/model.safetensors") != CHECKPOINT:
        raise ValueError("The checkpoint differs from the prior qualified run")
    upstream = subprocess.check_output(["git", "-C", "third_party/TCO", "rev-parse", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    if upstream != TCO:
        raise ValueError("Wrong upstream TCO base revision")
    prior = read(ROOT / "manifests/tco_frozen_prefix_v1.json")
    current = design()
    changed = [name for name, checksum in prior["source_sha256_lf"].items()
               if checksum != current["source_sha256_lf"][name]]
    if changed or current["metadata_sha256"] != prior["metadata_sha256"]:
        raise ValueError(f"Prior TCO source or metadata changed: {changed}")
    raw = file_hashes(DATA / "data", dict.fromkeys(SCENES, FRAMES))
    if raw != {f"7scenes/{name}": checksum for name, checksum in prior["input_source_sha256"].items()}:
        raise ValueError("Development input files differ from the previous study")
    if MANIFEST.exists():
        earlier = read_committed(MANIFEST)
        if earlier["jobs"] != current["jobs"] or any((PRIVATE / "launches" / job["id"]).exists()
                                                     for job in earlier["jobs"]):
            raise ValueError("The scientific schedule is locked before its first launch")
        target = MANIFEST.with_suffix(".next.json")
        write_new(target, dict(**current, input_source_sha256=raw))
        target.replace(MANIFEST)
    else:
        write_new(MANIFEST, dict(**current, input_source_sha256=raw))
    print(f"Registered {len(current['jobs'])} scientific jobs and {len(current['probes'])} probes")


def check_case_inputs(job, hashes, root):
    keys = [name for name in hashes if name.startswith(f"{job['dataset']}/{job['scene']}/")]
    if not keys or any(sha(root / key) != hashes[key] for key in keys):
        raise ValueError("Registered image, camera or evaluation file changed")


def is_success(receipt):
    return receipt["stopped"] and receipt["exit_code"] == 0 and not receipt["error"]


def case_receipts(job):
    key = job["id"]
    rows, attempts = [], []
    for path in sorted((PRIVATE / "launches" / key).glob("*/terminal.json")):
        receipt = read(path)
        if receipt["job"] != key or receipt["attempt"] != path.parent.name:
            raise ValueError("The case and launch identities differ")
        if is_success(receipt):
            complete = PRIVATE / "cases" / key / path.parent.name / "complete.json"
            result = read(complete)
            if any(result[field] != value for field, value in job.items()):
                raise ValueError(f"Finished result differs from registered job {key}")
            rows.append(dict(**result, attempt=receipt["attempt"], launch_wall_seconds=receipt["wall_seconds"]))
        attempts.append(receipt)
    if len(rows) > 1:
        raise ValueError(f"Multiple successes for {key}")
    return rows, attempts


def collect(jobs):
    rows, attempts = [], []
    for job in jobs:
        results, receipts = case_receipts(job)
        rows.extend(results)
        attempts.extend(receipts)
    return dict(rows=rows, attempts=attempts, rental_compute_usd=0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("register", "verify"))
    args = cli.parse_args()
    if args.action == "register":
        register()
    else:
        print(f"Verified {verify()['study']}")
