"""Seal source provenance and label-independent grouped folds for profiled LoRA.

Only signed neighboring-frame depth packets are read. Target depth labels and
held-source accuracy reports are never opened. Public verification uses NumPy;
full source reconstruction additionally needs Pillow and the original assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/lora_profiled_v1"
PLAN = ROOT / "manifests/lora_profiled_design_v1.json"
PACKETS = ROOT / "results/mast3r_partner_shape"
SIDES = {"earlier": -4, "later": 4}
CODE = ("scripts/prepare_profiled_inputs.py", "scripts/prepare_partner_sources.py",
        "scripts/neighbor_sparse_depth.py", "scripts/mast3r_network_priors.py",
        "scripts/calibration_audit.py", "scripts/focal_zoom_observations.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        header = json.dumps([name, value.dtype.str, list(value.shape)], separators=(",", ":"))
        digest.update(header.encode() + b"\0" + value.tobytes())
    return digest.hexdigest()


def native_selection(values):
    """Preserve the exact remap, nearest-depth order and dedup in native_samples."""
    pixels = np.rint((values[:, :2] + .5) * [512 / 518, 384 / 392] - .5).astype(np.int64)
    inside = ((pixels >= 0) & (pixels < [512, 384])).all(axis=1)
    nearest = {}
    for row in np.argsort(values[:, 2]):
        if inside[row]:
            nearest.setdefault(tuple(pixels[row]), row)
    selected = np.asarray(list(nearest.values()), dtype=np.int64)
    return selected, pixels[selected], int((~inside).sum()), int(inside.sum() - len(selected))


def grouped_folds(views, seed, case):
    """Union source rays across views and native 8x8 cells within each view.

    Assignment uses source identifiers, native coordinates and group counts only.
    One seeded pass balances counts; it never searches seeds or reads depth values.
    """
    nodes = [(i, j) for i, view in enumerate(views) for j in range(len(view["rows"]))]
    parent = list(range(len(nodes)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    source_seen, cell_seen = {}, {}
    for k, (i, j) in enumerate(nodes):
        source = tuple(views[i]["source_ids"][j])
        cell = (i, *tuple(views[i]["xy"][j] // 8))
        for key, seen in ((source, source_seen), (cell, cell_seen)):
            if key in seen:
                union(k, seen[key])
            else:
                seen[key] = k
    groups = {}
    for k in range(len(nodes)):
        groups.setdefault(find(k), []).append(k)
    # Native-sample order follows depth. Canonicalize components by provenance
    # and coordinates so that this order cannot influence fold assignment.
    def component_key(members):
        return tuple(sorted((nodes[k][0], *views[nodes[k][0]]["source_ids"][nodes[k][1]].tolist(),
                             *views[nodes[k][0]]["xy"][nodes[k][1]].tolist()) for k in members))
    components = sorted(groups.values(), key=component_key)
    context = int.from_bytes(hashlib.sha256(case.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, context]))
    priority = rng.permutation(len(components))
    order = sorted(priority.tolist(), key=lambda k: -len(components[k]))
    totals = np.asarray([len(v["rows"]) for v in views], dtype=np.int64)
    counts = np.zeros((2, len(views)), dtype=np.int64)
    assignments = np.empty(len(nodes), dtype=np.int8)
    group_ids = np.empty(len(nodes), dtype=np.int64)
    for group in order:
        members = components[group]
        sizes = np.bincount([nodes[k][0] for k in members], minlength=len(views))
        costs = [float(np.square((counts[0] - counts[1] + sign * sizes) / totals).sum())
                 for sign in (1, -1)]
        fold = int(rng.integers(2)) if costs[0] == costs[1] else int(np.argmin(costs))
        counts[fold] += sizes
        assignments[members] = fold
        group_ids[members] = group
    output = []
    start = 0
    for view in views:
        end = start + len(view["rows"])
        output.append((group_ids[start:end], assignments[start:end]))
        start = end
    return output, dict(components=len(components), largest_component=max(map(len, components)),
                       component_size_histogram={str(n):sum(len(v) == n for v in components)
                                                 for n in sorted(set(map(len, components)))},
                       fold_counts=counts.tolist())


def verify_grouping(views, labels):
    source_sets, cell_sets = [set(), set()], [set(), set()]
    for i, (view, (_, folds)) in enumerate(zip(views, labels)):
        for j, fold in enumerate(folds):
            source_sets[fold].add(tuple(view["source_ids"][j]))
            cell_sets[fold].add((i, *tuple(view["xy"][j] // 8)))
    if source_sets[0] & source_sets[1] or cell_sets[0] & cell_sets[1]:
        raise ValueError("Source-ray or native-cell leakage across spatial folds")


def build_splits():
    plan, origin = read(PLAN), read(PACKETS / "inputs.json")
    reference = read(ROOT / plan["runtime_reference"])
    expected_keys = {f"{s}_{f}" for s in plan["scenes"] for f in plan["target_frames"]}
    arrays, cases, packets = {}, {}, {}
    for side, offset in SIDES.items():
        path = PACKETS / f"{side}.npz"
        if sha(path) != origin["packets"][side]:
            raise ValueError(f"Packet changed: {side}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise ValueError("Unexpected packet inventory")
            packets[side] = {k:archive[k] for k in archive.files}
        for scene in plan["scenes"]:
            views = []
            for frame in plan["target_frames"]:
                data = packets[side][f"{scene}_{frame}"]
                if data.ndim != 2 or data.shape[1] != 7 or not np.isfinite(data).all():
                    raise ValueError("Invalid source packet schema")
                integers = data[:, [0, 1, 3, 4, 5, 6]]
                if not np.array_equal(integers, np.rint(integers)) or not (data[:, 2] > 0).all():
                    raise ValueError("Invalid packet coordinates or depths")
                if not (data[:, 3] == frame + offset).all():
                    raise ValueError("Wrong source frame identity")
                if not (((data[:, 4] - 40) % 80 == 0) & ((data[:, 5] - 30) % 60 == 0)).all():
                    raise ValueError("Wrong original sensor grid")
                if not ((data[:, 4:6] >= 0) & (data[:, 4:6] < [640, 480])).all():
                    raise ValueError("Source ray outside image")
                if not ((data[:, 6] > 0) & (data[:, 6] <= 10000)).all():
                    raise ValueError("Source sensor range outside registered validity")
                rows, xy, outside, duplicates = native_selection(data)
                views.append(dict(rows=rows, xy=xy, source_ids=data[rows, 3:6].astype(np.int64),
                                  frame=frame, original_rows=len(data), outside=outside, duplicates=duplicates))
            key = f"{scene}_{side}"
            labels, summary = grouped_folds(views, plan["spatial_fold_seed"], key)
            verify_grouping(views, labels)
            summary["views"] = []
            for view, (groups, folds) in zip(views, labels):
                counts = np.bincount(folds, minlength=2)
                required = max(plan["minimum_support_per_view"], plan["minimum_query_per_view"])
                if counts.min() < required:
                    raise ValueError(f"Insufficient grouped support in {key}/{view['frame']}: {counts.tolist()}")
                prefix = f"{key}_{view['frame']}"
                for field, value in (("rows", view["rows"]), ("xy", view["xy"]),
                                     ("source_ids", view["source_ids"]), ("group", groups), ("fold", folds)):
                    arrays[f"{prefix}__{field}"] = value
                summary["views"].append(dict(frame=view["frame"], packet_key=f"{scene}_{view['frame']}",
                    prefix=prefix, packet_rows=view["original_rows"], retained_rows=len(folds),
                    outside_native=view["outside"], native_pixel_duplicates_removed=view["duplicates"],
                    fold_counts=counts.tolist()))
            summary["source_id_intersection"] = summary["anchor_cell_intersection"] = 0
            cases[key] = summary
    source_files = {}
    index = read(ROOT / "results/source_frame_spacing/inputs.json")["frames"]
    for name, expected in origin["source_files"].items():
        scene, filename = name.rsplit("/", 1)
        frame_text, suffix = filename.split(".", 1)
        frame = int(frame_text.removeprefix("frame-"))
        if suffix == "depth.proj.png" and frame in plan["target_frames"]:
            raise ValueError("Target depth must remain evaluation-only")
        if index[scene][str(frame)]["original"][suffix] != expected:
            raise ValueError("Conflicting source provenance")
        source_files[name] = expected
    core = dict(schema_version=1, purpose="Grouped adaptation folds; no accuracy scores or target depth labels",
        seed=plan["spatial_fold_seed"], packet_columns=["target_u_518", "target_v_392", "axial_depth_m",
            "source_frame", "source_u_640", "source_v_480", "source_depth_mm"],
        arrays_schema={"prefix":"scene_side_targetframe", "rows":"int64 indices into immutable source packet in native_samples order",
            "xy":"int64 N by 2 native pixel coordinates in x,y order", "source_ids":"int64 N by 3 source frame,u,v",
            "group":"int64 connected-component ID within scene and temporal packet", "fold":"int8 0/1; support=step parity, query=1-parity"},
        native_wh=[512,384], packet_wh=[518,392], anchor_cell_wh=[8,8],
        assignment="Single context-seeded count-balancing pass over union components; no depth-based assignment or seed search",
        exclusion="Both temporal packets are prepared independently; each run loads only its permitted side. No held-source score is read.",
        source_files=source_files, packet_sha256=origin["packets"],
        immutable_sha256={name:sha(ROOT/name) for name in ("manifests/lora_profiled_design_v1.json",
            plan["runtime_reference"], "manifests/sensor_frame_v14.json", "results/mast3r_partner_shape/inputs.json",
            "results/source_frame_spacing/inputs.json", "results/mast3r_neural_transfer/pose_packets.npz")},
        code_sha256_lf={name:sha(ROOT/name, text=True) for name in CODE},
        expected_checkpoint_sha256=reference["checkpoint_sha256"], expected_upstream_commits=reference["upstream_commits"],
        cases=cases, splits_semantic_sha256=array_hash(arrays), total_selected=sum(len(v) for k,v in arrays.items() if k.endswith("__rows")))
    return arrays, core, packets


def resized_k(focal, width, height):
    sx, sy = width / 640., height / 480.
    return np.array([[focal*sx, 0, (320+.5)*sx-.5], [0, focal*sy, (240+.5)*sy-.5], [0,0,1.]])


def verify_original_sources(assets, core, packets):
    from PIL import Image
    for name, expected in core["source_files"].items():
        if sha(assets / name) != expected:
            raise ValueError(f"Original source identity changed: {name}")
    plan = read(PLAN)
    reference = read(ROOT / plan["runtime_reference"])
    calibration = np.asarray(read(ROOT / "manifests/sensor_frame_v14.json")["calibration"])
    u, v = np.meshgrid(np.arange(40,640,80), np.arange(30,480,60))
    uv = np.column_stack([u.ravel(),v.ravel()])
    rays = np.column_stack([uv,np.ones(len(uv))]) @ np.linalg.inv(resized_k(525,640,480)).T
    target_k = resized_k(525,518,392)
    target_k[[0,1],[0,1]] *= 1.12
    maximum_error = 0.
    for side, offset in SIDES.items():
        for scene in plan["scenes"]:
            folder = assets / reference["scenes"][scene]
            for target in plan["target_frames"]:
                source = target + offset
                def camera(frame):
                    return calibration @ np.linalg.inv(np.loadtxt(folder/f"frame-{frame:06d}.pose.txt"))
                relative = camera(target) @ np.linalg.inv(camera(source))
                with Image.open(folder/f"frame-{source:06d}.depth.proj.png") as image:
                    if image.size != (640,480):
                        raise ValueError("Unexpected original sensor resolution")
                    mm = np.asarray(image)[uv[:,1],uv[:,0]]
                xyz = (rays*(mm/1000.)[:,None]) @ relative[:3,:3].T + relative[:3,3]
                valid = (mm>0) & (mm<=10000) & (xyz[:,2]>0)
                xy = (xyz[valid] @ target_k.T)[:,:2] / xyz[valid,2:3]
                pixel = np.rint(xy).astype(int)
                keep = ((pixel>=0)&(pixel<[518,392])).all(axis=1)
                projected = np.column_stack([pixel, xyz[valid,2], np.full(len(pixel),source),uv[valid],mm[valid]])[keep]
                nearest = {}
                for row in sorted(projected, key=lambda a:a[2]):
                    nearest.setdefault(tuple(row[:2]),row)
                actual = np.stack(list(nearest.values()))
                old = packets[side][f"{scene}_{target}"]
                np.testing.assert_array_equal(actual[:,[0,1,3,4,5,6]], old[:,[0,1,3,4,5,6]])
                np.testing.assert_allclose(actual[:,2],old[:,2],rtol=1e-12,atol=1e-12)
                maximum_error = max(maximum_error,float(np.max(np.abs(actual[:,2]-old[:,2]))))
    return dict(original_files_checked=len(core["source_files"]), reconstructed_packets=30,
                maximum_projected_depth_difference_m=maximum_error, target_depth_files_opened=0)


def verify_runtime(core, weights):
    if sha(weights) != core["expected_checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    revisions = {}
    for name, expected in core["expected_upstream_commits"].items():
        actual = subprocess.check_output(["git","-C",str(ROOT/"third_party"/name),"rev-parse","HEAD"],text=True).strip()
        if actual != expected:
            raise ValueError(f"Upstream revision changed: {name}")
        revisions[name] = actual
    return dict(checkpoint_sha256=core["expected_checkpoint_sha256"], upstream_commits=revisions)


def verify_native_samples(packets):
    from mast3r_network_priors import native_samples
    checked = 0
    for packet in packets.values():
        for values in packet.values():
            rows, xy, _, _ = native_selection(values)
            actual_xy, actual_depth = native_samples(values, "cpu")
            np.testing.assert_array_equal(actual_xy.numpy().T,xy)
            np.testing.assert_array_equal(actual_depth.numpy(),values[rows,2].astype(np.float32))
            checked += 1
    return dict(packet_views_checked=checked, native_samples_agreement=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--verify-native", action="store_true")
    args = parser.parse_args()
    arrays, core, packets = build_splits()
    report_path, split_path = DEST/"inputs.json", DEST/"splits.npz"
    if args.verify:
        old = read(report_path)
        if old["registration"] != core:
            raise ValueError("Immutable registration changed; preserve the original and version the revision")
        if sha(split_path) != old["splits_binary_sha256"]:
            raise ValueError("Sealed split archive changed")
        with np.load(split_path,allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("Split inventory changed")
            for key,value in arrays.items():
                np.testing.assert_array_equal(saved[key],value)
    elif report_path.exists() or split_path.exists():
        raise FileExistsError("Preserve registered artifacts; use --verify")
    verification = {}
    if args.assets:
        verification["source_reconstruction"] = verify_original_sources(args.assets,core,packets)
    if args.weights:
        verification["runtime_identity"] = verify_runtime(core,args.weights)
    if args.verify_native:
        verification["native_bridge"] = verify_native_samples(packets)
    if not args.verify:
        if not args.assets or not args.weights or not args.verify_native:
            parser.error("Registration requires --assets, --weights and --verify-native")
        DEST.mkdir(parents=True,exist_ok=True)
        with split_path.open("xb") as stream:
            np.savez_compressed(stream,**arrays)
        report = dict(registration=core, verification_at_registration=verification,
                      splits_binary_sha256=sha(split_path),
                      execution_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip())
        with report_path.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
            stream.write("\n")
    print(json.dumps(dict(pass_all=True, mode="verify" if args.verify else "register", cases=len(core["cases"]),
        selected=core["total_selected"], minimum_fold_count=min(min(v["fold_counts"]) for c in core["cases"].values() for v in c["views"]),
        verification_this_run=verification)))


if __name__ == "__main__":
    main()
