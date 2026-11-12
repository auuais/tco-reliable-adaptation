"""Register or verify the MASt3R crossover run list using archived evidence. CPU only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/mast3r_neural_transfer"
PROTOCOL = ROOT / "docs/MAST3R_GEOMETRY_MATCHING_GOAL.md"
MANIFEST = ROOT / "manifests/mast3r_geometry_matching_v1.json"
PARENT_COMMIT = "afef4e998ac0ef3786e7a0f04844c52187a8fa07"
SCENES = {"heads": "heads/seq-01", "chess": "chess/seq-03", "fire": "fire/seq-03"}
GRIDS = ("centered", "shifted")
POSES = ("clean", "noisy")
MATCH_FOLDER = "corres_conf=desc_conf_subsample=8"
SOURCE_RECORDS = (
    "results/mast3r_neural_transfer/plan.json",
    "results/mast3r_neural_transfer/session.json",
    "results/mast3r_neural_transfer/archive.json",
    "results/mast3r_neural_transfer/predictions.json",
    "results/mast3r_neural_transfer/observations.json",
    "results/mast3r_neural_transfer/metrics.json",
    "results/mast3r_neural_transfer/scores.json",
    "results/mast3r_neural_transfer/pose_packets.npz",
    "results/neighbor_sparse_depth/measurements.npz",
    "results/shifted_sparse_grid/measurements.npz",
    "manifests/dense_focal_inputs.json",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_pairs(scene, rgb_hash):
    names = [f"{scene}_base_{rgb_hash[:12]}_{index}" for index in range(5)]
    ids = [hashlib.md5(name.encode("utf-8")).hexdigest() for name in names]
    paths = set()
    for first, second in itertools.permutations(ids, 2):
        paths.add(f"forward/{first}/{second}.pth")
        paths.add(f"{MATCH_FOLDER}/{first}-{second}.pth")
    return paths


def cache_inventory(parent_plan):
    archive = read(PARENT / "archive.json")
    inventory = {}
    for scene in SCENES:
        expected = expected_pairs(scene, parent_plan["rgb_sha256"][scene])
        keys = [f"{scene}_zero"] + [f"{scene}_{grid}_D" for grid in GRIDS]
        for key in keys:
            records = [item for item in archive["matching_caches"][key]
                       if item["file"].split("/")[0] in ("forward", MATCH_FOLDER)]
            paths = {item["file"] for item in records}
            require(len(records) == 40 and paths == expected, f"Incomplete ordered pairs in {key}")
            inventory[key] = records
    return inventory


def run_matrix():
    predictions = read(PARENT / "predictions.json")["rows"]
    lookup = {row["tag"]: row for row in predictions}
    rows = []
    # Reconstruct both historical endpoints before building the two counterfactuals.
    for geometry, bundle in ((0, 0), (1, 1), (1, 0), (0, 1)):
        for scene, grid, pose in itertools.product(SCENES, GRIDS, POSES):
            frozen, adapted = f"{scene}_zero", f"{scene}_{grid}_D"
            row = dict(
                id=f"{scene}_{grid}_{pose}_G{geometry}B{bundle}", scene=scene, grid=grid,
                pose_packet=f"{scene}_{pose}", alignment=f"KDP_{pose}",
                geometry=geometry, bundle=bundle,
                coordinates_from=adapted if geometry else frozen,
                confidences_and_matches_from=adapted if bundle else frozen,
            )
            if geometry == bundle:
                case = adapted if geometry else f"{scene}_zero_{grid}"
                tag = f"{case}_KDP_{pose}"
                old = lookup[tag]
                require(old["case"] == case and old["post"] == row["alignment"], f"Unexpected endpoint {tag}")
                row["endpoint"] = dict(tag=tag, native_sha256=old["native_sha256"], point_digest=old["prediction"])
            rows.append(row)
    require(len({r["id"] for r in rows}) == 48, "Expected 48 distinct alignments")
    require(sum("endpoint" in r for r in rows) == 24, "Expected 24 endpoint controls")
    return rows


def expected_plan():
    parent = read(PARENT / "plan.json")
    session = read(PARENT / "session.json")
    scores = read(PARENT / "scores.json")["grids"]
    return dict(
        study="mast3r_geometry_matching_v1", parent_commit=PARENT_COMMIT,
        protocol_sha256_lf=text_sha256(PROTOCOL),
        parent_artifact_sha256={name: sha256(ROOT / name) for name in SOURCE_RECORDS},
        upstream_commits=session["upstream_commits"], image_sha256=session["image_sha256"],
        checkpoint_sha256=parent["checkpoint_sha256"], scenes=SCENES,
        frames=[25, 225, 425, 625, 825], native_hw=[384, 512], known_K=parent["intrinsic"],
        adapted_arm="D", references=["registered_rgb", "released"],
        alignment=dict(seed=42, subsample=8, mode="avg-angle", kinematic_mode="hclust-ward",
                       lr1=0.07, niter1=300, lr2=0.01, niter2=300, opt_depth=True,
                       shared_intrinsics=False, matching_conf_thr=0.0,
                       depth_weight=0.01, intrinsics_weight=0.01, pose_weight=0.01),
        bridge_tolerance=dict(rtol=1e-5, atol=1e-6),
        historical_best={grid: {pose: scores[grid][pose]["best_control"] for pose in POSES} for grid in GRIDS},
        branch_selection=dict(primary="registered_rgb", alternate="released", extent="one_branch",
                              margin_floor=1e-6, max_replay_error_multiplier=2,
                              tie_break="largest_worst_primary_relative_gain_then_geometry"),
        repeat_design=dict(fresh_processes=3, optimizer_seed=42, scene_order=list(SCENES),
                           original_pose_seed_base=20261112, new_pose_seed_bases=[20261130, 20261230],
                           rotation_degrees=5, translation_fraction_of_diameter=0.02),
        spending_rule=dict(min_relative_gain=0.05, min_scene_gain=0.0, max_component_ratio=1.02,
                           max_alternate_score_ratio=1.0, positive_gain_each_paired_repeat=True,
                           additional_runpod_compute_usd=0,
                           reserved_transfer_usd=25),
        runs=run_matrix(),
    )


def validate_public(plan):
    for row in plan["runs"]:
        if "endpoint" in row:
            old = row["endpoint"]
            path = PARENT / "native_outputs" / (old["tag"] + ".npz")
            require(sha256(path) == old["native_sha256"], f"Changed endpoint {old['tag']}")
    parent_plan = read(PARENT / "plan.json")
    require(sha256(PARENT / "pose_packets.npz") == parent_plan["pose_packets_sha256"], "Pose packet mismatch")
    for grid, name in (("centered", "neighbor_sparse_depth"), ("shifted", "shifted_sparse_grid")):
        require(sha256(ROOT / "results" / name / "measurements.npz") == parent_plan["packets"][grid],
                f"Changed sparse measurements on {grid}")
    return cache_inventory(parent_plan)


def validate_cache(root, inventory):
    files = 0
    size = 0
    for name, records in inventory.items():
        for item in records:
            path = root / name / item["file"]
            require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"Changed {path}")
            files += 1
            size += item["bytes"]
    print(f"Verified {files} archived forward/match files ({size:,} bytes) under {root}.")


def validate_local_inputs():
    registry = read(ROOT / "manifests/dense_focal_inputs.json")
    local = ROOT / "reports/dense_focal_pilot"
    for scene in SCENES:
        for group, directory in (("inputs", "inference"), ("references", "evaluation")):
            record = registry[group][scene + "_base"]
            path = local / directory / record["file"]
            require(sha256(path) == record["sha256"], f"Input mismatch for {path}")
    print("Verified 3 RGB archives and 3 offline reference archives by file hash.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="Create the initial manifest if absent.")
    parser.add_argument("--cache-root", action="append", type=Path, default=[], help="Hash a local cache copy.")
    parser.add_argument("--local-inputs", action="store_true", help="Check local RGB/reference bytes.")
    args = parser.parse_args()
    plan = expected_plan()
    inventory = validate_public(plan)
    if args.register and not MANIFEST.exists():
        with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    require(read(MANIFEST) == plan, "Registered manifest differs from its protocol or parent evidence")
    print("Verified 12 scene/grid/pose combinations, 24 archived controls and 24 new hybrid jobs.")
    if args.local_inputs:
        validate_local_inputs()
    for location in args.cache_root:
        validate_cache(location, inventory)


if __name__ == "__main__":
    main()
