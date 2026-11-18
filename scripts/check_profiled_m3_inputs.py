"""CPU synthetic checks for signed input preparation; no dataset arrays opened."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_profiled_m3_inputs import (ROOT, SCENES, FRAMES, SIDES, AllowedReader,
    build_shifted_splits, dump_new, fixed_noisy_w2c, project_samples, resized_k,
    sensor_grid, source_names, validate_c2w)
from prepare_profiled_inputs import sha


def rejected(function):
    try:
        function()
    except (ValueError,AssertionError):
        return True
    raise AssertionError("Invalid input unexpectedly accepted")


def synthetic_records(side):
    uv = sensor_grid((20,15))
    rays = np.column_stack([uv,np.ones(len(uv))]) @ np.linalg.inv(resized_k(525,640,480)).T
    target = resized_k(525,518,392)
    target[[0,1],[0,1]] *= 1.12
    records = {}
    for scene in SCENES:
        for frame in FRAMES:
            # Unique depths make an input-order invariance check unambiguous.
            mm = 2000 + np.arange(len(uv))
            records[f"{scene}_{frame}"] = project_samples(mm,uv,rays,np.eye(4),target,frame+SIDES[side])
    return records


def fold_assignments(arrays):
    result = {}
    for key,folds in arrays.items():
        if key.endswith("__fold"):
            prefix = key.removesuffix("__fold")
            result[prefix] = {tuple(source.tolist()+xy.tolist()):int(fold)
                for source,xy,fold in zip(arrays[prefix+"__source_ids"],arrays[prefix+"__xy"],folds)}
    return result


def run():
    checks = {}
    # Load only the pure reference function's syntax: importing its historical
    # orchestration module would pull unrelated model/evaluation dependencies.
    tree = ast.parse((ROOT/"scripts/neighbor_sparse_depth.py").read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="project_samples")
    namespace = {"np":np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),"pinned_project_samples","exec"),namespace)
    uv = sensor_grid((20,15))
    rays = np.column_stack([uv,np.ones(len(uv))]) @ np.linalg.inv(resized_k(525,640,480)).T
    relative = np.eye(4)
    relative[:3,:3] = Rotation.from_rotvec([.01,-.03,.02]).as_matrix()
    relative[:3,3] = [.15,-.03,.08]
    mm = 2000+np.arange(len(uv))*21
    mm[:3] = [0,10001,500]
    target = resized_k(525,518,392); target[[0,1],[0,1]] *= 1.12
    first = project_samples(mm,uv,rays,relative,target,21)
    second = namespace["project_samples"](mm,uv,rays,relative,target,21)
    np.testing.assert_array_equal(first,second)
    assert (first[:,3]==21).all() and (first[:,6]>0).all() and (first[:,6]<=10000).all()
    checks["projection_exact_pinned_arithmetic_and_provenance"] = True

    identity = project_samples(np.full(len(uv),2000),uv,rays,np.eye(4),resized_k(525,518,392),21)
    expected_xy = np.rint((uv+.5)*[518/640,392/480]-.5)
    np.testing.assert_array_equal(identity[:,:2],expected_xy)
    np.testing.assert_array_equal(identity[:,2],np.full(len(uv),2.))
    checks["independent_pixel_center_identity_and_metric_units"] = True

    records = synthetic_records("earlier")
    arrays,cases = build_shifted_splits(records,"earlier")
    repeated,_ = build_shifted_splits(records,"earlier")
    for key in arrays:
        np.testing.assert_array_equal(arrays[key],repeated[key])
    shuffled = {key:value[::-1].copy() for key,value in records.items()}
    reversed_arrays,_ = build_shifted_splits(shuffled,"earlier")
    assert fold_assignments(arrays)==fold_assignments(reversed_arrays)
    assert min(min(v["fold_counts"]) for c in cases.values() for v in c["views"])>=8
    checks["grouped_folds_deterministic_and_packet_order_independent"] = True
    later_arrays,later_cases = build_shifted_splits(synthetic_records("later"),"later")
    assert len(later_arrays)==len(arrays) and all("_later_shifted" in key for key in later_cases)
    checks["later_packet_prepared_independently"] = True
    bad = copy.deepcopy(records); bad["heads_25"][0,3]=29
    checks["opposite_source_rejected"] = rejected(lambda:build_shifted_splits(bad,"earlier"))
    bad = copy.deepcopy(records); bad["heads_25"][0,4]=40
    checks["centered_source_coordinate_rejected"] = rejected(lambda:build_shifted_splits(bad,"earlier"))
    tiny = {key:value[:10] for key,value in records.items()}
    checks["insufficient_support_fails_without_seed_search"] = rejected(lambda:build_shifted_splits(tiny,"earlier"))
    checks["changed_split_seed_rejected"] = rejected(lambda:build_shifted_splits(records,"earlier",seed=99))
    # Deliberately merge all points into one 8x8 cell in each view: native count
    # can still exceed 16 but grouped support must fail, not search another seed.
    clustered = copy.deepcopy(records)
    for key,value in clustered.items():
        n = min(24,len(value)); value = value[:n].copy()
        value[:,:2] = np.column_stack([np.arange(n)%6+100,np.arange(n)//6+100])
        clustered[key]=value
    checks["large_leakage_group_minimum_rejected"] = rejected(lambda:build_shifted_splits(clustered,"earlier"))

    folders = {s:s+"/seq-01" for s in SCENES}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        allowed = {}
        for name,record in source_names(folders,"earlier").items():
            payload = name.encode()
            path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(payload)
            allowed[name] = dict(record,sha256=hashlib.sha256(payload).hexdigest())
        reader = AllowedReader(root,folders,"earlier",allowed)
        assert reader.access==[]
        checks["target_depth_read_rejected_before_open"] = rejected(lambda:reader.bytes("heads/seq-01/frame-000025.depth.proj.png"))
        checks["opposite_depth_read_rejected_before_open"] = rejected(lambda:reader.bytes("heads/seq-01/frame-000029.depth.proj.png"))
        checks["path_escape_rejected_before_open"] = rejected(lambda:reader.bytes("../forbidden"))
        assert reader.access==[]
        reader.preflight()
        assert len(reader.access)==45
        checks["exact_selected_side_whitelist"] = True
        wrong = copy.deepcopy(allowed)
        wrong["heads/seq-01/frame-000025.depth.proj.png"] = next(iter(wrong.values()))
        checks["broadened_whitelist_rejected"] = rejected(lambda:AllowedReader(root,folders,"earlier",wrong))
        (root/next(iter(allowed))).write_bytes(b"changed")
        checks["source_hash_mismatch_rejected_before_decode"] = rejected(lambda:AllowedReader(root,folders,"earlier",allowed).preflight())

    c2w = np.tile(np.eye(4),(5,1,1))
    for view in range(5):
        c2w[view,:3,:3]=Rotation.from_rotvec([.03*view,-.01*view,.02*view]).as_matrix()
        c2w[view,:3,3]=[.2*view,.03*view**2,-.02*view]
    clean = np.linalg.inv(c2w)
    saved = clean.copy()
    noisy = fixed_noisy_w2c(clean,0)
    for _ in range(3):
        np.testing.assert_array_equal(noisy,fixed_noisy_w2c(clean,0))
    np.testing.assert_array_equal(saved,clean)
    # Inversion can introduce machine-epsilon differences in anchored view0.
    noisy_c2w=np.linalg.inv(noisy)
    np.testing.assert_allclose(noisy_c2w[0],c2w[0],rtol=0,atol=1e-15)
    magnitude=Rotation.from_matrix(c2w[1:,:3,:3].transpose(0,2,1)@noisy_c2w[1:,:3,:3]).magnitude()
    np.testing.assert_allclose(magnitude,np.deg2rad(5),rtol=0,atol=2e-15)
    diameter=np.linalg.norm(c2w[:,None,:3,3]-c2w[None,:,:3,3],axis=-1).max()
    np.testing.assert_allclose(np.linalg.norm(noisy_c2w[1:,:3,3]-c2w[1:,:3,3],axis=-1),.02*diameter,rtol=0,atol=1e-15)
    assert not np.array_equal(noisy,fixed_noisy_w2c(clean,1))
    checks["fixed_pose_repeats_exact_and_clean_input_unchanged"] = True
    checks["independent_pose_angle_translation_anchor_and_scene_seed"] = True
    approximate=clean.copy(); approximate[:,:3,:3]*=.9998
    unchanged=approximate.copy()
    geometry=validate_c2w(approximate)
    fixed_noisy_w2c(approximate,0)
    np.testing.assert_array_equal(approximate,unchanged)
    assert geometry["max_rotation_orthogonality_error"]>2e-5 and geometry["rotation_modified"] is False
    checks["pinned_approximate_rotations_preserved_and_diagnosed"] = True
    reflected=clean.copy(); reflected[0,:3,0]*=-1
    checks["left_handed_camera_rejected"] = rejected(lambda:validate_c2w(reflected))
    degenerate=np.tile(np.eye(4),(5,1,1))
    checks["zero_camera_diameter_rejected"] = rejected(lambda:fixed_noisy_w2c(degenerate,0))
    return dict(schema="profiled_m3_input_checks_v1",pass_all=all(checks.values()),checks=checks,
        fixture_scope="Synthetic CPU geometry, grouped splits, strict reads and fixed pose construction only",
        real_dataset_arrays_opened=0,target_depth_files_opened=0,model_experiments_run=0,
        source_sha256_lf={name:sha(ROOT/name,text=True) for name in
            ("scripts/check_profiled_m3_inputs.py","scripts/prepare_profiled_m3_inputs.py",
             "scripts/prepare_profiled_inputs.py","scripts/neighbor_sparse_depth.py")},numpy=np.__version__)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    report=run()
    if args.output:
        dump_new(args.output,report)
    print(json.dumps(report,indent=2))
