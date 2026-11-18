"""Replay registered LoRA folds and challenge source/cell grouping on CPU.

The default check requires only NumPy and public repository artifacts. Optional
original-pose verification reads no depth images; --verify-native additionally
checks the existing PyTorch native_samples function on CPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_profiled_inputs import (ROOT, DEST, PLAN, read, sha, array_hash,
    build_splits, grouped_folds, verify_grouping, verify_native_samples)


OUTPUT = DEST / "split_checks.json"


def fixture_checks():
    # A and B share a cell in view 0; A repeats in view 1 and shares a cell
    # with C there. All four observations must form one transitive component.
    # The last observation has equal pixel coordinates across views but distinct
    # source identities; anchor-cell identity must remain view-specific.
    views = [dict(rows=np.arange(4), xy=np.array([[1,1],[2,2],[50,50],[70,70]]),
                  source_ids=np.array([[21,40,30],[21,120,30],[21,280,30],[21,360,30]])),
             dict(rows=np.arange(4), xy=np.array([[30,40],[31,41],[80,80],[70,70]]),
                  source_ids=np.array([[21,40,30],[21,200,30],[21,440,30],[21,520,30]]))]
    labels, summary = grouped_folds(views, 20261118, "transitive_fixture")
    verify_grouping(views, labels)
    component = labels[0][0][0]
    assert labels[0][0][1] == labels[1][0][0] == labels[1][0][1] == component
    assert labels[0][0][3] != labels[1][0][3]
    assert summary["components"] == 5 and summary["largest_component"] == 4

    permutations = [([3,2,1,0], [3,2,1,0]), ([2,0,3,1], [1,3,0,2]), ([1,2,0,3], [2,0,3,1])]
    for orders in permutations:
        changed = [{key:value[order].copy() for key,value in view.items()}
                   for view,order in zip(views,orders)]
        actual, other_summary = grouped_folds(changed, 20261118, "transitive_fixture")
        verify_grouping(changed, actual)
        assert other_summary == summary
        for i, order in enumerate(orders):
            inverse = np.argsort(order)
            np.testing.assert_array_equal(actual[i][0][inverse], labels[i][0])
            np.testing.assert_array_equal(actual[i][1][inverse], labels[i][1])

    # Split the two view-1 cell neighbors together: only the cross-view source
    # identity is now leaked. Then split B alone: only cell identity is leaked.
    rejected = []
    for name, flips in (("source_only", ((1,0),(1,1))), ("cell_only", ((0,1),))):
        invalid = [(group.copy(),np.zeros(len(fold),dtype=np.int8)) for group,fold in labels]
        for i,j in flips:
            invalid[i][1][j] = 1
        try:
            verify_grouping(views,invalid)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"Leakage check accepted {name}")
    return dict(transitive_component_observations=4, expected_components=5,
        view_specific_cells_pass=True, permutation_checks=len(permutations),
        provenance_to_fold_invariant=True, isolated_leakage_cases_rejected=rejected)


def registered_checks():
    arrays, core, packets = build_splits()
    record = read(DEST/"inputs.json")
    if core != record["registration"]:
        raise ValueError("Input registration changed")
    if sha(DEST/"splits.npz") != record["splits_binary_sha256"]:
        raise ValueError("Split archive bytes changed")
    with np.load(DEST/"splits.npz",allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("Unexpected split fields")
        for key,value in arrays.items():
            np.testing.assert_array_equal(saved[key],value)
        if array_hash({key:saved[key] for key in saved.files}) != core["splits_semantic_sha256"]:
            raise ValueError("Split semantic hash changed")
    minimum = min(min(v["fold_counts"]) for case in core["cases"].values() for v in case["views"])
    report = dict(input_registration_sha256=sha(DEST/"inputs.json"),
        splits_binary_sha256=record["splits_binary_sha256"], splits_semantic_sha256=core["splits_semantic_sha256"],
        scene_side_cases=len(core["cases"]), views=sum(len(c["views"]) for c in core["cases"].values()),
        observations=core["total_selected"], minimum_fold_observations=minimum,
        all_source_and_anchor_intersections_zero=all(c["source_id_intersection"]==c["anchor_cell_intersection"]==0
                                                    for c in core["cases"].values()))
    assert report["all_source_and_anchor_intersections_zero"]
    return report,core,packets


def clean_pose_checks(assets,core):
    plan = read(PLAN)
    reference = read(ROOT/plan["runtime_reference"])
    calibration = np.asarray(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    results = {}
    with np.load(ROOT/"results/mast3r_neural_transfer/pose_packets.npz",allow_pickle=False) as poses:
        for slug in plan["scenes"]:
            expected = []
            for frame in plan["target_frames"]:
                relative = f"{reference['scenes'][slug]}/frame-{frame:06d}.pose.txt"
                path = assets/relative
                if sha(path) != core["source_files"][relative]:
                    raise ValueError(f"Original target pose changed: {relative}")
                expected.append(calibration @ np.linalg.inv(np.loadtxt(path)))
            actual = poses[slug+"_clean"]
            np.testing.assert_allclose(actual,np.asarray(expected),rtol=0,atol=1e-12)
            results[slug] = float(np.max(np.abs(actual-np.asarray(expected))))
    return dict(convention="world_to_RGB_camera = depth_to_RGB_calibration @ inv(original_camera_to_world_pose)",
                checked_poses=len(plan["scenes"])*len(plan["target_frames"]),
                maximum_absolute_errors=results, tolerance_absolute=1e-12, target_depth_files_opened=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--assets",type=Path)
    parser.add_argument("--verify-native",action="store_true")
    args = parser.parse_args()
    original_input_hash = sha(DEST/"inputs.json")
    registered,core,packets = registered_checks()
    portable = dict(registered=registered, fixtures=fixture_checks())
    optional = {}
    if args.assets:
        optional["clean_poses"] = clean_pose_checks(args.assets,core)
    if args.verify_native:
        optional["native_samples"] = verify_native_samples(packets)
    assert sha(DEST/"inputs.json") == original_input_hash
    if args.verify:
        previous = read(OUTPUT)
        if previous["portable"] != portable:
            raise ValueError("Portable split check changed; preserve the registered result")
        for key,value in optional.items():
            if key == "clean_poses":
                if previous["optional_at_registration"][key]["checked_poses"] != value["checked_poses"]:
                    raise ValueError("Pose-check inventory changed")
            elif previous["optional_at_registration"].get(key) != value:
                raise ValueError(f"Optional check changed: {key}")
    else:
        with OUTPUT.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(dict(portable=portable, optional_at_registration=optional, pass_all=True),
                      stream,indent=2,allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(pass_all=True, portable_checks="registered arrays, transitive unions, view-specific cells, three permutations, isolated source/cell leakage rejection",
                         optional_checks_this_run=optional, input_registration_unchanged=True)))


if __name__ == "__main__":
    main()
