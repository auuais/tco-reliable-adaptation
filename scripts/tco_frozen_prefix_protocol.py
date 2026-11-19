"""Freeze the local TCO efficiency experiment and independently replay its decision."""
import argparse
import json
import math
import random
import subprocess
from pathlib import Path
from statistics import mean, median

from run_profiled_d0 import read, sha, write_new

ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_frozen_prefix_v1"
PRIVATE = ROOT/"reports"/STUDY
PUBLIC = ROOT/"results"/STUDY
MANIFEST = ROOT/"manifests"/f"{STUDY}.json"
IMAGE = "sha256:541dd1e09a32b8da7851929e73810882415aa30967b0ae2acd5473c9cf5ec3f1"
CHECKPOINT = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
TCO = "65387333877723cf99086f7f29089379dd50c59a"
SCENES = ("heads/seq-01", "chess/seq-03", "fire/seq-03")
FRAMES = [0, 200, 400, 600, 800]
ARMS = dict(zero=[0], original=[5, 10, 40], four=[5, 10, 40],
            cached12=[5, 10, 40], cached6=[5, 10, 40], uncached12=[10], uncached6=[10])
SOURCES = (
    "scripts/run_tco_frozen_prefix.py", "scripts/run_tco_frozen_prefix_local.py",
    "scripts/tco_frozen_prefix.py", "scripts/tco_frozen_prefix_protocol.py", "scripts/check_tco_frozen_prefix.py",
    "scripts/run_experiment.py", "configs/7scenes_baseline.yaml", "docs/TCO_FROZEN_PREFIX_HANDOFF.md",
    "third_party/TCO/tco_vggt_lora.py", "third_party/TCO/tco_loss.py", "third_party/TCO/objective.py",
    "third_party/TCO/utils/interfaces.py", "third_party/TCO/utils/geometry.py", "third_party/TCO/peft/lora.py",
    "third_party/TCO/base_models/vggt/vggt/heads/dpt_head.py",
    "third_party/TCO/base_models/vggt/vggt/heads/camera_head.py",
)


def design():
    items = []
    rng = random.Random(20261120)
    for seed in (42, 43, 44):
        block = []
        for scene in SCENES:
            for arm, counts in ARMS.items():
                for steps in counts:
                    key = f"{scene.split('/')[0]}_{arm}_{steps:02d}_{seed}"
                    block.append(dict(id=key, scene=scene, arm=arm, steps=steps, seed=seed))
        rng.shuffle(block)
        items.extend(block)
    return dict(study=STUDY, upstream=TCO, checkpoint_sha256=CHECKPOINT, docker_image=IMAGE,
                adapter=dict(rank=4, alpha=16, dropout=0, learning_rate=.0003),
                frames=FRAMES, image_shape=[5, 3, 392, 518], rental_compute_usd=0,
                jobs=items, thresholds=dict(scene_seconds=.50, peak_allocated=.80,
                                            accuracy_mean_m=1.01, completeness_mean_m=1.01),
                launch_permutations_seed=20261120,
                timer="CPU-loaded model and inputs through complete final geometry with cache construction and transfers",
                placement="original/four full GPU and patch offload; suffix both stacks stream one frozen stage at a time",
                source_sha256_lf={name: sha(ROOT/name, text=True) for name in SOURCES},
                metadata_sha256=sha(ROOT/"environment/development_dataset_cache_v7.npy"))


def verify():
    plan = read(MANIFEST)
    existing = dict(plan)
    existing.pop("input_source_sha256")
    if existing != design():
        raise ValueError("Execution differs from the registered design")
    committed = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(ROOT), "show",
                                         f"HEAD:manifests/{STUDY}.json"])
    if committed.replace(b"\r\n", b"\n") != MANIFEST.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("Commit the preregistration before running")
    return plan


def register(data):
    plan = design()
    if sha(data/"checkpoints/VGGT-1B/model.safetensors") != CHECKPOINT:
        raise ValueError("VGGT checkpoint differs from the registered reproduction")
    inputs = {}
    for scene in SCENES:
        for frame in FRAMES:
            for leaf in ("color.png", "depth.proj.png", "pose.txt"):
                relative = f"{scene}/frame-{frame:06d}.{leaf}"
                inputs[relative] = sha(data/"data/7scenes"/relative)
    write_new(MANIFEST, dict(**plan, input_source_sha256=inputs))


def collect(plan):
    rows, failed = [], []
    for job in plan["jobs"]:
        key = job["id"]
        receipts = sorted((PRIVATE/"launches"/key).glob("*/terminal.json"))
        valid = []
        for path in receipts:
            receipt = read(path)
            if receipt["job"] != key or receipt["attempt"] != path.parent.name:
                raise ValueError("Launch identity differs")
            if receipt["stopped"] and receipt["exit_code"] == 0 and receipt["error"] is None:
                result = read(PRIVATE/"cases"/key/receipt["attempt"]/"complete.json")
                if any(result[field] != value for field, value in job.items()):
                    raise ValueError("Completed experiment disagrees with job registration")
                valid.append(dict(**result, launch_wall_seconds=receipt["wall_seconds"]))
            else:
                failed.append(receipt)
        if len(valid) > 1:
            raise ValueError("Multiple successful runs for one registered job")
        rows.extend(valid)
    return dict(rows=rows, failed=failed, registered_jobs=len(plan["jobs"]), rental_compute_usd=0)


def decision(plan, evidence):
    fields = tuple(plan["thresholds"])
    quality = ("accuracy_mean_m", "completeness_mean_m")
    observed = {row["id"]: row for row in evidence["rows"]}
    if len(observed) != len(evidence["rows"]) or not observed.keys() <= {j["id"] for j in plan["jobs"]}:
        raise ValueError("Duplicate or unexpected evidence")
    if len(observed) != len(plan["jobs"]):
        return dict(verdict="incomplete", finished=len(observed), planned=len(plan["jobs"]))
    groups = {}
    for job in plan["jobs"]:
        row = observed[job["id"]]
        if any(row[key] != value for key, value in job.items()) or row["rental_compute_usd"]:
            raise ValueError("Wrong run binding or nonzero spend")
        group = (job["scene"], job["arm"], job["steps"])
        groups.setdefault(group, []).append(row)
    metrics = {}
    for scene in SCENES:
        scene_rows = [row for row in observed.values() if row["scene"] == scene]
        if len({r["input_sha256"] for r in scene_rows}) != 1:
            raise ValueError("Inconsistent inputs within a matched scene")
    for key, rows in groups.items():
        metrics[key] = {}
        for field in fields:
            values = [row[field] if field in row else row["metrics"][field] for row in rows]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Invalid measurement: {key}/{field}")
            metrics[key][field] = median(values)
    comparisons = {}
    for arm, steps_list in ARMS.items():
        for steps in steps_list:
            scene_ratios = {scene: {field: metrics[scene, arm, steps][field]/metrics[scene, "four", 40][field]
                                   for field in fields} for scene in SCENES}
            ratios = {field: mean([r[field] for r in scene_ratios.values()]) for field in fields}
            comparisons[f"{arm}_{steps}"] = dict(ratios=ratios, scene_ratios=scene_ratios)
    advances = []
    for key, record in comparisons.items():
        if not key.startswith("cached"):
            continue
        ratios = record["ratios"]
        feasible = [name for name, other in comparisons.items() if name.startswith(("original_", "four_"))
                    and other["ratios"]["scene_seconds"] <= ratios["scene_seconds"]]
        reference = min(feasible, key=lambda name: (
            mean(comparisons[name]["ratios"][f] for f in quality),
            comparisons[name]["ratios"]["scene_seconds"], comparisons[name]["ratios"]["peak_allocated"]
        )) if feasible else None
        resource_quality = all(ratios[field] <= plan["thresholds"][field] for field in fields)
        against_equal_time = None
        if reference:
            stronger = comparisons[reference]["ratios"]
            against_equal_time = (mean(ratios[f] for f in quality) < mean(stronger[f] for f in quality)
                                  and all(ratios[f] <= 1.01*stronger[f] for f in quality))
        record.update(time_matched_control=reference, matches_resource_and_quality=resource_quality,
                      beats_time_matched_control=against_equal_time)
        if resource_quality and against_equal_time:
            advances.append(key)
    return dict(verdict="advance" if advances else "missed_targets", comparisons=comparisons,
                nominated=min(advances, key=lambda k: (comparisons[k]["ratios"]["scene_seconds"],
                                                       comparisons[k]["ratios"]["peak_allocated"], k)) if advances else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "collect", "replay"))
    args = parser.parse_args()
    if args.mode == "register":
        register(Path("E:/ResearchData/TCO"))
    elif args.mode == "collect":
        plan = verify()
        result = collect(plan)
        print(json.dumps(decision(plan, result), indent=2))
        if len(result["rows"]) == len(plan["jobs"]):
            write_new(PUBLIC/"evidence.json", result)
            write_new(PUBLIC/"decision.json", decision(plan, result))
    else:
        report = decision(read(MANIFEST), read(PUBLIC/"evidence.json"))
        if report != read(PUBLIC/"decision.json"):
            raise ValueError("Public decision does not reproduce")
        print(json.dumps(dict(verified=True, verdict=report["verdict"])))


if __name__ == "__main__":
    main()
