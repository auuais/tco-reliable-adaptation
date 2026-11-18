"""D1 role-scoped byte inventory and CPU array readers.

Preparation hashes bytes and reads JSON/source only. Loaders are for a controller
that has already verified the committed D1 execution registration, exact D0
replay and full parent seal. They never authorize execution themselves. Only the
two selected Heads/earlier states and explicit NPZ members can be decoded.
"""
import hashlib
from pathlib import Path
import zipfile

import profiled_d0_artifacts as d0


FRAMES, PAIRS, MODULES = d0.FRAMES, d0.PAIRS, d0.MODULES
RUNS = ("heads_earlier_P0", "heads_earlier_P3")
INPUTS = "results/lora_profiled_v1/inputs.json"
RGB_PLAN = "manifests/dense_focal_inputs.json"
RGB = "reports/dense_focal_pilot/inference/heads_base_rgb.npz"
PACKET = "results/mast3r_partner_shape/earlier.npz"
SPLITS = "results/lora_profiled_v1/splits.npz"
POSES = "results/mast3r_neural_transfer/pose_packets.npz"
PACKET_MEMBERS = tuple(f"heads_{frame}" for frame in FRAMES)
SPLIT_MEMBERS = tuple(f"heads_earlier_{frame}__{field}" for frame in FRAMES for field in ("rows", "xy", "fold"))
FACTOR_MEMBERS = tuple(module+suffix for module in MODULES for suffix in (".lora_A", ".lora_B"))
ROLES = ("rgb", "packet", "splits", "poses", "teacher", "adapter")
ROLE_MEMBERS = dict(rgb=["images"], packet=list(PACKET_MEMBERS), splits=list(SPLIT_MEMBERS),
                    poses=["heads_clean"], teacher=["depth", "confidence"], adapter=list(FACTOR_MEMBERS))
require, sha, read = d0.require, d0.sha, d0.read
_at, _descriptor, _verified_path, _unchanged = d0._at, d0._descriptor, d0._verified_path, d0._unchanged


def _validate_design(design):
    require(design.get("schema") == "profiled_lora_d1_design_v1" and
            design.get("stage") == "D1_single_paired_inference_and_calibration_response_probe_without_updates",
            "Wrong D1 design")
    scope, resources = design["scope"], design["resources"]
    require(scope["scene"] == "heads" and scope["adaptation_source"] == "earlier" and
            scope["frames"] == list(FRAMES) and scope["trained_state"] == RUNS[1] and
            scope["frozen_state"] == RUNS[0] and scope["directed_pairs"] == 20 and
            scope["support_folds"] == [0, 1] and scope["graph_weight"] == 0,
            "D1 scene/source/state/fold scope changed")
    require(resources["total_new_decoder_calls"] == 40 and resources["frozen_encoder_feature_computations"] == 5
            and resources["pristine_training_path_decoder_calls"] == resources["final_training_path_decoder_calls"] == 20
            and resources["new_FP32_decoder_calls"] == resources["network_backward_calls"] ==
            resources["optimizer_updates"] == resources["rental_usd"] == 0, "D1 call/update budget changed")
    require(design["trigger"]["exact_d0_replay_required_before_d1_execution"] is True,
            "D1 requires exact D0 replay")
    require(design["immutable_model"]["adapter"]["run_id"] == RUNS[1] and
            design["immutable_model"]["original_parameters_and_buffers_frozen"] is True,
            "D1 model state scope changed")


def _unique(rows, key, label):
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, "Duplicate " + label)
        result[value] = row
    return result


def _source_bindings(root, *maps):
    result = {}
    for values in maps:
        for name, digest in values.items():
            require(name not in result or result[name] == digest, "Conflicting parent source identity: " + name)
            require(sha(_at(root, name), text=True) == digest, "Changed bound parent source: " + name)
            result[name] = digest
    return result


def _role(root, name, file, digest):
    return dict(role=name, members=list(ROLE_MEMBERS[name]), **_descriptor(root, file, digest))


def build_inventory(root, design_file, weights_dir):
    """Build the exact allowed D1 inventory without importing tensor libraries.

    Revalidates selected parent metadata and all selected bytes. The controller
    separately performs the full 36-state parent-seal audit and exact D0 replay
    gate; this function does not rehash unrelated cases or decode their arrays.
    """
    root, weights_dir = Path(root).resolve(), Path(weights_dir).resolve()
    design_descriptor = _descriptor(root, design_file)
    design = read(_at(root, design_file)); _validate_design(design)
    trigger = design["trigger"]
    d0_execution = _descriptor(root, trigger["d0_execution"]["file"], trigger["d0_execution"]["sha256"])
    d0_report = _descriptor(root, trigger["d0_report"]["file"], trigger["d0_report"]["sha256"])
    previous, report = read(_at(root, d0_execution["file"])), read(_at(root, d0_report["file"]))
    require(previous.get("schema") == "profiled_d0_execution_v1" and report.get("schema") == "profiled_d0_report_v1"
            and report.get("status") == "complete" and
            report.get("registration") == {k: d0_execution[k] for k in ("file", "sha256")},
            "D0 report/registration linkage changed")
    inherited = previous["inventory"]
    require(inherited.get("schema") == "profiled_d0_artifact_inventory_v1", "Wrong D0 parent inventory")
    seal_desc = _descriptor(root, inherited["trigger"]["seal"]["file"], inherited["trigger"]["seal"]["sha256"])
    registry_desc = _descriptor(root, inherited["parent_registry"]["file"], inherited["parent_registry"]["sha256"])
    seal, registry = read(_at(root, seal_desc["file"])), read(_at(root, registry_desc["file"]))
    require(seal.get("pass_all") is True and seal["completed_states"] == 36 and
            seal["common_scientific_registry"] == registry_desc["file"] and
            seal["common_scientific_registry_sha256"] == registry_desc["sha256"], "M2 parent seal linkage changed")
    jobs, selected = _unique(registry["jobs"], "run_id", "M2 job"), _unique(seal["cases"], "run_id", "M2 sealed case")
    require(set(RUNS) <= set(jobs) and set(RUNS) <= set(selected), "Missing selected D1 parent states")
    source_bindings = _source_bindings(root, registry["source_sha256_lf"], previous["source_sha256_lf"])
    upstream = inherited["upstream_sources"]
    for name, item in upstream.items():
        require(sha(_at(root, name), text=True) == item["sha256_lf"], "Changed bound upstream writer: " + name)

    model = design["immutable_model"]
    weights = {}
    for key, basename in (("checkpoint", "model.safetensors"), ("config", "config.json")):
        expected = model[key]
        require(expected["file"] == basename, "Unexpected model/config file")
        weights[key] = _descriptor(weights_dir, basename, expected["sha256"])
        require(weights[key]["bytes"] == expected["bytes"] and weights[key] == inherited[key],
                "Model/config bytes differ from D0")
    require(weights["checkpoint"]["sha256"] == registry["checkpoint_sha256"], "Checkpoint differs from M2")
    config = read(_at(weights_dir, "config.json"))
    require(config.get("dec_depth") == 12 and config.get("dec_embed_dim") == 768,
            "Checkpoint decoder shape differs from adapter mapping")

    inputs_desc = _descriptor(root, INPUTS, registry["input_sha256"][INPUTS])
    inputs = read(_at(root, INPUTS))
    require(inputs["splits_binary_sha256"] == registry["input_sha256"][SPLITS], "Split identities disagree")
    rgb_plan = _descriptor(root, RGB_PLAN, inherited["rgb_plan"]["sha256"])
    meta = read(_at(root, RGB_PLAN))["inputs"]["heads_base"]
    require(meta["file"] == "heads_base_rgb.npz" and meta.get("scene") == "heads/seq-01"
            and meta.get("batch") == "base", "Wrong RGB case")
    roles = dict(
        rgb=_role(root, "rgb", RGB, meta["sha256"]),
        packet=_role(root, "packet", PACKET, inputs["registration"]["packet_sha256"]["earlier"]),
        splits=_role(root, "splits", SPLITS, inputs["splits_binary_sha256"]),
        poses=_role(root, "poses", POSES, inputs["registration"]["immutable_sha256"][POSES]),
    )
    require({k: roles["rgb"][k] for k in ("file", "sha256", "bytes")} == inherited["rgb_archive"],
            "RGB archive differs from D0")
    instances = [f'heads_base_{meta["sha256"][:12]}_{i}' for i in range(5)]
    hashes = [hashlib.md5(name.encode()).hexdigest() for name in instances]
    evidence, forward = {}, {}
    for run in RUNS:
        job, chosen = jobs[run], selected[run]["selected"]
        require(job["scene"] == "heads" and job["adaptation_source"] == "earlier" and
                job["arm"] == run.split("_")[-1] and job["input_packet"] == PACKET,
                "Selected parent job has wrong input role")
        completion_desc = _descriptor(root, chosen["report"], chosen["report_sha256"])
        complete = read(_at(root, chosen["report"]))
        raw = job["raw_attempt_parent"] + "/" + chosen["attempt"]
        public = job["public_attempt_parent"] + "/" + chosen["attempt"]
        require(chosen["status"] == "complete" and complete["run_id"] == run and complete["job"] == job
                and complete["attempt"] == chosen["attempt"] and complete["raw_attempt"] == raw
                and complete["raw_files"] == chosen["raw_files"], "Selected completion differs from parent seal")
        require(complete["checkpoint_sha256"] == weights["checkpoint"]["sha256"] and
                complete["original_state_unchanged"] is True and
                complete["original_state_sha256"] == chosen["original_state_sha256"] ==
                complete["pre_endpoint_original_state_sha256"] == complete["final_original_state_sha256"],
                "Original model state integrity differs")
        require(complete["input_sha256"][INPUTS] == inputs_desc["sha256"], "Selected adaptation provenance differs")
        capture = complete["endpoint"]["capture"]
        require(capture["input_sha256"] == meta["sha256"] and capture["prior_cache_files"] == 0
                and capture["inference_precision"] == "float32", "FP32 cache freshness or input differs")
        expected_names = {f"endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth" for i, j in PAIRS}
        require({name for name in chosen["raw_files"] if name.startswith("endpoint_cache/forward/")} == expected_names,
                "Missing/extra selected forward pair")
        forward[run] = [dict(role="fp32_first_camera_xyz", run_id=run, i=i, j=j,
                            **_descriptor(root, raw+f"/endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth",
                                          chosen["raw_files"][f"endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth"]))
                        for i, j in PAIRS]
        evidence[run] = dict(attempt=chosen["attempt"], raw_root=raw, public_root=public,
                            completion=completion_desc, original_state_sha256=complete["original_state_sha256"],
                            docker_image=complete["docker_image"], runtime_policy=complete["runtime_policy"],
                            adapter_initial_sha256=complete.get("adapter_initial_sha256"),
                            adapter_final_sha256=complete.get("adapter_final_sha256"))
        if run == RUNS[0]:
            require(chosen["adapter"].get("frozen_P0") is True and job["adapt_lora"] is False,
                    "P0 is not a pristine frozen model")
        else:
            require(job["adapt_lora"] is True and job["graph_weight"] == 0 and job["differentiate_solve"] is True
                    and complete["adapter_module_count"] == 144, "Wrong saved P3 mechanism")
            adapter = chosen["adapter"]
            require({k: adapter[k] for k in ("file", "sha256")} == {k: model["adapter"][k] for k in ("file", "sha256")}
                    and adapter["file"] == raw+"/adapters.npz"
                    and adapter["sha256"] == chosen["raw_files"]["adapters.npz"]
                    and adapter["state_sha256"] == complete["adapter_final_sha256"], "Wrong selected final adapter")
            roles["adapter"] = _role(root, "adapter", adapter["file"], adapter["sha256"])
            teacher_desc = _descriptor(root, public+"/teacher.json", complete["public_files"]["teacher.json"])
            teacher = read(_at(root, teacher_desc["file"]))
            require(teacher["decoder_calls"] == 20 and teacher["checkpoint_sha256"] == weights["checkpoint"]["sha256"]
                    and teacher["initial_adapter_state_sha256"] == complete["adapter_initial_sha256"]
                    and teacher["full_pair_pointmaps_archived"] is False, "Teacher provenance differs")
            teacher_file = raw+"/teacher_initial.npz"
            require(teacher["fused_native_file"] == teacher_file and
                    teacher["fused_native_sha256"] == chosen["raw_files"]["teacher_initial.npz"],
                    "Teacher tensor identity differs")
            roles["teacher"] = _role(root, "teacher", teacher_file, teacher["fused_native_sha256"])
    for key in ("original_state_sha256", "docker_image", "runtime_policy"):
        require(evidence[RUNS[0]][key] == evidence[RUNS[1]][key], "P0/P3 parent identity differs: " + key)
    result = dict(
        schema="profiled_d1_artifact_inventory_v1", design=design_descriptor,
        trigger=dict(d0_execution=d0_execution, d0_report=d0_report),
        parent_seal=seal_desc, parent_registry=registry_desc, parent_evidence=evidence,
        parent_source_sha256_lf=source_bindings, upstream_sources=upstream,
        checkpoint=weights["checkpoint"], config=weights["config"], config_hash_is_historical_M2_proof=False,
        adaptation_provenance=inputs_desc, rgb_plan=rgb_plan, rgb_archive_sha256=meta["sha256"],
        instances=instances, teacher_metadata=teacher_desc, roles=roles, forward=forward,
        scene="heads", adaptation_source="earlier", frames=list(FRAMES), runs=list(RUNS),
        module_names=list(MODULES), arrays_deserialized=False, execution_authorized=False,
        external_release_requirements=["Full parent-seal verification", "Exact D0 replay verified",
                                       "Committed executable D1 registration verified", "Teacher acceptance before final-state decode"],
        raw_tuple_note="Torch safely materializes the four-tensor container; only tuple[0] is returned or used",
    )
    _validate_roles(result)
    return result


def verify_inventory(root, weights_dir, inventory):
    _validate_roles(inventory)
    current = build_inventory(root, inventory["design"]["file"], weights_dir)
    require(current == inventory, "D1 input inventory changed; no tensor decode allowed")
    return current


def _validate_roles(inventory):
    require(isinstance(inventory, dict) and inventory.get("schema") == "profiled_d1_artifact_inventory_v1",
            "Wrong D1 inventory schema")
    require(inventory.get("scene") == "heads" and inventory.get("adaptation_source") == "earlier" and
            inventory.get("frames") == list(FRAMES) and inventory.get("runs") == list(RUNS)
            and inventory.get("module_names") == list(MODULES), "D1 role scope changed")
    require(set(inventory["roles"]) == set(ROLES) and set(inventory["forward"]) == set(RUNS),
            "Missing or additional D1 input role")
    roots = inventory["parent_evidence"]
    require(set(roots) == set(RUNS), "Additional selected model state")
    for run in RUNS:
        prefix = f"reports/lora_profiled_v1/cases/{run}/"
        require(roots[run]["raw_root"] == prefix+roots[run]["attempt"] and
                roots[run]["public_root"] == f"results/lora_profiled_v1/cases/{run}/"+roots[run]["attempt"]
                and "/" not in roots[run]["attempt"] and "\\" not in roots[run]["attempt"]
                and roots[run]["attempt"] not in ("", ".", ".."), "Unsafe selected attempt root")
    expected_files = dict(rgb=RGB, packet=PACKET, splits=SPLITS, poses=POSES,
                          teacher=roots[RUNS[1]]["raw_root"]+"/teacher_initial.npz",
                          adapter=roots[RUNS[1]]["raw_root"]+"/adapters.npz")
    for role in ROLES:
        value = inventory["roles"][role]
        require(value.get("role") == role and value.get("file") == expected_files[role]
                and value.get("members") == ROLE_MEMBERS[role], "D1 member/path role changed: " + role)
    digest = inventory["rgb_archive_sha256"]
    require(inventory["roles"]["rgb"]["sha256"] == digest and isinstance(digest, str) and len(digest) == 64,
            "RGB archive identity changed")
    names = [f'heads_base_{digest[:12]}_{i}' for i in range(5)]
    require(inventory["instances"] == names, "Forward instance names differ from complete RGB archive hash")
    hashes = [hashlib.md5(name.encode()).hexdigest() for name in names]
    for run in RUNS:
        pairs = inventory["forward"][run]
        require([(p.get("i"), p.get("j")) for p in pairs] == list(PAIRS), "D1 directed-pair inventory differs")
        for p in pairs:
            expected = roots[run]["raw_root"]+f'/endpoint_cache/forward/{hashes[p["i"]]}/{hashes[p["j"]]}.pth'
            require(p.get("role") == "fp32_first_camera_xyz" and p.get("run_id") == run and p.get("file") == expected,
                    "Wrong forward input role, pair or state")


def _role_descriptor(inventory, role):
    _validate_roles(inventory)
    require(role in ROLES, "Unauthorized input role")
    return inventory["roles"][role]


def _npz_read(root, descriptor, exact=False):
    """Return only the allowlisted members, with duplicate-container checks."""
    import numpy as np
    path = _verified_path(root, descriptor)
    expected = descriptor["members"]
    try:
        with zipfile.ZipFile(path) as zipped:
            names = [item.filename for item in zipped.infolist()]
            require(len(names) == len(set(names)), "Duplicate ZIP member, including an unselected member")
            require(all(name.endswith(".npy") and "/" not in name and "\\" not in name for name in names),
                    "Unsupported NPZ container member")
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files
            require(len(names) == len(set(names)), "Duplicate NPZ array member")
            require(set(expected) <= set(names) and (not exact or set(expected) == set(names)),
                    "Missing or unexpected NPZ member")
            return {name: archive[name].copy() for name in expected}
    finally:
        _unchanged(path, descriptor)


def _array(value, dtype, shape, label):
    import numpy as np
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and value.shape == tuple(shape),
            "Wrong dtype/shape: " + label)
    require(bool(np.isfinite(value).all()), "Nonfinite required array: " + label)
    return value


def load_rgb(root, inventory):
    values = _npz_read(root, _role_descriptor(inventory, "rgb"))
    images = _array(values["images"], "float32", (5, 3, 392, 518), "RGB images")
    require(bool(((images >= 0) & (images <= 1)).all()), "RGB values outside the original [0,1] range")
    return images


def load_adaptation(root, inventory):
    """Read five signed earlier packets and their rows/xy/fold members only."""
    import numpy as np
    packet = _npz_read(root, _role_descriptor(inventory, "packet"))
    splits = _npz_read(root, _role_descriptor(inventory, "splits"))
    result, source_folds, cell_folds = [], {}, {}
    for view, frame in enumerate(FRAMES):
        values = packet[f"heads_{frame}"]
        require(values.dtype == np.dtype("float64") and values.ndim == 2 and values.shape[1] == 7
                and 0 < len(values) <= 64 and np.isfinite(values).all(), "Wrong source-packet dtype/schema")
        require(np.array_equal(values[:, [0, 1, 3, 4, 5, 6]], np.rint(values[:, [0, 1, 3, 4, 5, 6]]))
                and (values[:, 2] > 0).all() and (values[:, 3] == frame-4).all(),
                "Wrong signed source frame, integer fields or measured axial depth")
        require(((values[:, :2] >= 0) & (values[:, :2] < [518, 392])).all()
                and ((values[:, 4:6] >= 0) & (values[:, 4:6] < [640, 480])).all()
                and (((values[:, 4]-40) % 80 == 0) & ((values[:, 5]-30) % 60 == 0)).all()
                and ((values[:, 6] > 0) & (values[:, 6] <= 10000)).all(), "Invalid original sensor coordinates/range")
        prefix = f"heads_earlier_{frame}"
        rows, xy, folds = (splits[prefix+"__"+field] for field in ("rows", "xy", "fold"))
        require(rows.dtype == np.dtype("int64") and rows.ndim == 1 and len(rows) >= 16
                and len(set(rows.tolist())) == len(rows) and ((rows >= 0) & (rows < len(values))).all(),
                "Invalid selected packet rows")
        _array(xy, "int64", (len(rows), 2), "adaptation xy")
        _array(folds, "int8", (len(rows),), "spatial folds")
        require(np.isin(folds, [0, 1]).all() and min(int((folds == k).sum()) for k in (0, 1)) >= 8,
                "Both original folds require at least eight observations")
        # Exact historical native remap/depth-order/dedup; no measurement is newly fitted.
        pixels = np.rint((values[:, :2]+.5)*[512/518, 384/392]-.5).astype(np.int64)
        inside = ((pixels >= 0) & (pixels < [512, 384])).all(axis=1)
        nearest = {}
        for row in np.argsort(values[:, 2]):
            if inside[row]: nearest.setdefault(tuple(pixels[row]), int(row))
        expected_rows = np.asarray(list(nearest.values()), dtype=np.int64)
        require(np.array_equal(rows, expected_rows) and np.array_equal(xy, pixels[rows]),
                "Selected rows or native half-pixel mapping differ from sealed preparation")
        for point, source, fold in zip(xy, values[rows, 3:6].astype(np.int64), folds):
            for table, key in ((source_folds, tuple(source)), (cell_folds, (view, *tuple(point//8)))):
                require(key not in table or table[key] == int(fold), "Source ray/native anchor cell crosses folds")
                table[key] = int(fold)
        result.append(dict(frame=frame, rows=rows, xy=xy, depth=values[rows, 2].copy(), fold=folds))
    return result


def load_poses(root, inventory):
    import numpy as np
    values = _npz_read(root, _role_descriptor(inventory, "poses"))
    poses = _array(values["heads_clean"], "float64", (5, 4, 4), "heads_clean supplied w2c")
    require(np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            and (np.linalg.det(poses) != 0).all(), "Invalid supplied homogeneous camera matrices")
    return poses  # Preserve original approximately rigid values; never orthogonalize.


def load_teacher(root, inventory):
    values = _npz_read(root, _role_descriptor(inventory, "teacher"), exact=True)
    for key in ("depth", "confidence"):
        _array(values[key], "float32", (5, 384, 512), "sealed teacher " + key)
    require(bool((values["confidence"] > 0).all()), "Nonpositive fused teacher confidence")
    return values


def load_adapter(root, inventory):
    values = _npz_read(root, _role_descriptor(inventory, "adapter"), exact=True)
    for module in MODULES:
        width = 2304 if module.endswith("attn.qkv") else 768
        _array(values[module+".lora_A"], "float32", (4, 768), module+".lora_A")
        _array(values[module+".lora_B"], "float32", (width, 4), module+".lora_B")
    return values


def load_forward(root, inventory, run_id, i, j):
    """Safely load a selected four-tensor container; expose only full tuple[0] XYZ."""
    import torch
    _validate_roles(inventory)
    require(run_id in RUNS and type(i) is int and type(j) is int and (i, j) in PAIRS,
            "Unauthorized FP32 state/pair")
    descriptor = next(item for item in inventory["forward"][run_id] if (item["i"], item["j"]) == (i, j))
    path = _verified_path(root, descriptor)
    try:
        values = torch.load(path, map_location="cpu", weights_only=True)
        require(type(values) is tuple and len(values) == 4, "Expected exact raw forward four-tuple")
        for value, shape in zip(values, ((384, 512, 3), (384, 512), (384, 512, 3), (384, 512))):
            require(type(value) is torch.Tensor and value.device.type == "cpu" and value.dtype == torch.float32
                    and tuple(value.shape) == shape, "Raw forward tensor dtype/device/shape differs")
        require(bool(torch.isfinite(values[0]).all()), "Nonfinite required first-camera pointmap")
        return values[0].numpy().copy()
    finally:
        _unchanged(path, descriptor)
