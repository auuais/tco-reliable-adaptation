"""Strict saved-artifact boundary for the registered, CPU-only D0 diagnosis.

Inventory construction reads JSON/source and hashes opaque bytes. It does not
decode tensors. The separate loaders must only be called after the controller
has verified a committed executable registration. No model module is imported.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

from profiled_artifacts import resolve_logical
from seal_profiled_predictions import verify_seal


SEAL = "results/lora_profiled_v1/prediction_seal.json"
REPLAY = "results/lora_profiled_v1/independent_public_replay.json"
RGB_PLAN = "manifests/dense_focal_inputs.json"
SPLITS = "results/lora_profiled_v1/splits.npz"
FRAMES = (25, 225, 425, 625, 825)
PAIRS = tuple(itertools.permutations(range(5), 2))
ADAPTER_RUNS = tuple(f"{scene}_{side}_P{arm}" for scene in ("heads", "chess", "fire")
                     for side in ("earlier", "later") for arm in range(1, 6))
FUNCTIONAL_RUNS = tuple(f"heads_earlier_P{arm}" for arm in range(4)) + ("heads_later_P0",)
MODULES = tuple(f"{stack}.{block}.{projection}" for stack in ("dec_blocks", "dec_blocks2")
                for block in range(12) for projection in
                ("attn.qkv", "attn.proj", "cross_attn.projq", "cross_attn.projk",
                 "cross_attn.projv", "cross_attn.proj"))
XY_MEMBERS = tuple(f"heads_earlier_{frame}__xy" for frame in FRAMES)
GRID = dict(native_shape=[384, 512], x_start=8, y_start=8, stride=16, points_per_view=768)
QUANTILES = [0, .25, .5, .75, .9, .99, 1]
UPSTREAM = (("third_party/mast3r", "mast3r/cloud_opt/sparse_ga.py"),
            ("third_party/mast3r", "mast3r/utils/misc.py"),
            ("third_party/mast3r/dust3r/croco", "models/blocks.py"))
LAYOUT = dict(
    schema="profiled_d0_source_layout_v1", native_shape=[384, 512],
    cache_tuple=["X_ii", "C_ii", "X_ji", "C_ji"], self_depth="tuple[0][...,2]",
    xyz_shape=[384, 512, 3], confidence_shape=[384, 512], source_dtype="float32",
    other_branch="partner pixels in first-camera coordinates; not partner self depth",
    instance_name="scene_base_<complete RGB archive sha256 first 12 chars>_<view index>",
    cache_path="endpoint_cache/forward/md5(instance_i)/md5(instance_j).pth",
    factor_A_shape="[4,768]", factor_B_shape="[2304,4] for attn.qkv; [768,4] otherwise",
    base_key="MODULE.weight", effective_update="(1/4)*B@A",
    raw_p0_comparator="same RGB/model/runtime; forward cache saved before prior-dependent optimizer",
    arrays_deserialized=False,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _at(root, name):
    return resolve_logical(Path(root), name, must_exist=True)


def _descriptor(root, name, expected=None):
    path = _at(root, name)
    value = sha(path)
    require(expected is None or value == expected, "Changed registered bytes: " + name)
    return dict(file=name, sha256=value, bytes=path.stat().st_size)


def _verified_path(root, descriptor):
    require(isinstance(descriptor, dict) and isinstance(descriptor.get("sha256"), str)
            and len(descriptor["sha256"]) == 64, "Missing SHA256 descriptor")
    path = _at(root, descriptor.get("file"))
    require(sha(path) == descriptor["sha256"], "Changed artifact before decode: " + descriptor["file"])
    if "bytes" in descriptor:
        require(path.stat().st_size == descriptor["bytes"], "Artifact size differs")
    return path


def _unchanged(path, descriptor):
    require(sha(path) == descriptor["sha256"], "Artifact changed during decode: " + descriptor["file"])


def _upstream_sources(root, registry):
    result = {}
    for checkout, relative in UPSTREAM:
        revision = registry["upstream_revisions"][checkout]
        require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
                "Malformed pinned upstream revision")
        directory = resolve_logical(Path(root), checkout)
        require(directory.is_dir(), "Pinned upstream checkout directory is missing")
        blob = subprocess.check_output(
            ["git", "-c", "safe.directory=" + str(directory), "-C", str(directory),
             "show", revision + ":" + relative])
        name = checkout + "/" + relative
        path = _at(root, name)
        require(path.read_bytes().replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"),
                "Writer differs from pinned upstream Git blob: " + name)
        result[name] = dict(sha256_lf=sha(path, text=True), revision=revision,
                            git_blob_path=relative)
    return result


def _validate_design(design):
    require(design.get("schema") == "profiled_lora_diagnosis_design_v1" and
            design.get("stage") == "D0_saved_artifact_diagnosis_without_updates", "Wrong D0 design")
    require(design["adapter_norm_scope"]["run_ids"] == list(ADAPTER_RUNS) and
            design["adapter_norm_scope"]["expected_adapters"] == 30 and
            design["adapter_norm_scope"]["expected_modules_per_adapter"] == 144 and
            design["adapter_norm_scope"]["expected_factor_arrays_per_adapter"] == 288,
            "D0 adapter scope differs")
    scope = design["functional_scope"]
    require(scope["run_ids"] == list(FUNCTIONAL_RUNS[:-1]) and scope["same_state_forward_comparator"] ==
            FUNCTIONAL_RUNS[-1] and scope["scene"] == "heads" and scope["adaptation_source"] == "earlier"
            and scope["frames"] == list(FRAMES) and scope["directed_pairs"] == 20, "D0 functional scope differs")
    require(scope.get("grid") == GRID and scope.get("quantiles") == QUANTILES,
            "D0 fixed grid or quantiles differ from the implemented mathematics")
    require(design["immutable_model"]["rank"] == 4 and design["immutable_model"]["alpha"] == 1,
            "D0 factor scaling differs")
    resource = design["resources"]
    require(resource["rental_usd"] == resource["model_forward_calls"] == resource["optimizer_updates"] == 0
            and resource["gpu_requested"] is False and resource["max_cpu_wall_minutes"] == 30
            and resource["max_concurrent_jobs"] == 1, "D0 resource scope differs")


def _verify_evaluator_sources(root, seal):
    """Verify the frozen byte-auditor dependency closure without scorer execution.

This does not impose the historical scorer's Python/NumPy runtime: D0 invokes
only its standard-library sealer and filename helper, not score arithmetic.
"""
    identity = seal["evaluation_execution"]
    descriptor = _descriptor(root, identity["file"], identity["sha256"])
    evaluation = read(_at(root, identity["file"]))
    require(evaluation.get("schema") == "profiled_evaluation_execution_v1",
            "Wrong parent evaluation registration schema")
    sources = evaluation.get("source_sha256_lf")
    require(isinstance(sources, dict) and
            {"scripts/seal_profiled_predictions.py", "scripts/profiled_artifacts.py"} <= set(sources),
            "Frozen evaluation registration omits byte-auditor dependencies")
    for name, expected in sources.items():
        require(sha(_at(root, name), text=True) == expected,
                "Frozen evaluation source changed: " + name)
    return descriptor, sources


def build_inventory(root, design_file, weights_dir):
    """Bind the complete allowed input inventory using JSON and opaque hashes only.

Replays the existing global-seal *byte* audit, not the scorer. May read/hash
archives but never deserializes any array or imports NumPy/Torch/safetensors.
This inventory itself does not authorize execution.
"""
    root, weights_dir = Path(root).resolve(), Path(weights_dir).resolve()
    design_descriptor = _descriptor(root, design_file)
    design = read(_at(root, design_file))
    _validate_design(design)
    trigger = design["trigger"]
    require(trigger["result"] == "no_eligible_M2_nominee", "D0 requires negative M2 trigger")
    seal_descriptor = _descriptor(root, SEAL, trigger["prediction_seal_sha256"])
    score_descriptor = _descriptor(root, trigger["score_file"], trigger["score_sha256"])
    replay_descriptor = _descriptor(root, REPLAY, trigger["independent_replay_sha256"])
    seal = read(_at(root, SEAL))
    evaluation_descriptor, evaluation_sources = _verify_evaluator_sources(root, seal)
    verified = verify_seal(root, SEAL, seal["evaluation_execution"])
    require(verified == seal and seal.get("pass_all") is True and seal["completed_states"] == 36,
            "Parent prediction seal did not verify")
    score, replay = read(_at(root, score_descriptor["file"])), read(_at(root, REPLAY))
    require(score.get("nominee", "missing") is None and replay.get("pass_all") is True
            and replay.get("summary", {}).get("nominee", "missing") is None,
            "Missing independently confirmed negative nomination")
    require(score["prediction_seal"] == SEAL and score["prediction_seal_sha256"] == seal_descriptor["sha256"]
            and replay["linkage"]["prediction_seal"] == {k: seal_descriptor[k] for k in ("file", "sha256")}
            and replay["linkage"]["report"] == {k: score_descriptor[k] for k in ("file", "sha256")},
            "Negative trigger linkage differs")
    registry_name = seal["common_scientific_registry"]
    registry_descriptor = _descriptor(root, registry_name, seal["common_scientific_registry_sha256"])
    registry = read(_at(root, registry_name))
    jobs = {row["run_id"]: row for row in registry["jobs"]}
    selected = {row["run_id"]: row["selected"] for row in seal["cases"]}
    require(len(jobs) == len(selected) == 36, "Duplicate or missing M2 cases")
    upstream = _upstream_sources(root, registry)
    checkpoint = _descriptor(weights_dir, "model.safetensors", design["immutable_model"]["checkpoint_sha256"])
    require(checkpoint["sha256"] == registry["checkpoint_sha256"], "Base checkpoint differs from M2")
    config = _descriptor(weights_dir, "config.json", design["immutable_model"]["prospective_current_config_sha256"])
    config_value = read(_at(weights_dir, "config.json"))
    require(config_value.get("dec_depth") == 12 and config_value.get("dec_embed_dim") == 768,
            "Current model configuration disagrees with fixed decoder module mapping")

    evidence, adapters, functional = {}, [], []
    for run_id in dict.fromkeys(ADAPTER_RUNS + FUNCTIONAL_RUNS):
        chosen, job = selected[run_id], jobs[run_id]
        completion = read(_at(root, chosen["report"]))
        require(chosen["status"] == "complete" and completion["job"] == job
                and completion["run_id"] == run_id and completion["attempt"] == chosen["attempt"],
                "Selected completion identity differs")
        raw = job["raw_attempt_parent"] + "/" + chosen["attempt"]
        require(completion["raw_attempt"] == raw and completion["raw_files"] == chosen["raw_files"],
                "Selected raw inventory differs from completion")
        evidence[run_id] = dict(attempt=chosen["attempt"],
            completion=_descriptor(root, chosen["report"], chosen["report_sha256"]),
            original_state_sha256=chosen["original_state_sha256"],
            checkpoint_sha256=completion["checkpoint_sha256"], docker_image=completion["docker_image"],
            runtime_policy=completion["runtime_policy"], input_sha256=completion["input_sha256"])
        if run_id in ADAPTER_RUNS:
            item = chosen["adapter"]
            require(item["file"] == raw + "/adapters.npz" and
                    item["sha256"] == chosen["raw_files"]["adapters.npz"], "Adapter path differs from seal")
            adapters.append(dict(run_id=run_id, **_descriptor(root, item["file"], item["sha256"]),
                                 modules=144, factor_arrays=288, rank=4, alpha=1))

    plan_descriptor = _descriptor(root, RGB_PLAN)
    rgb_meta = read(_at(root, RGB_PLAN))["inputs"]["heads_base"]
    rgb_name = "reports/dense_focal_pilot/inference/" + rgb_meta["file"]
    rgb_descriptor = _descriptor(root, rgb_name, rgb_meta["sha256"])
    names = [f'heads_base_{rgb_meta["sha256"][:12]}_{i}' for i in range(5)]
    hashes = [hashlib.md5(name.encode("utf-8")).hexdigest() for name in names]
    for run_id in FUNCTIONAL_RUNS:
        chosen = selected[run_id]
        raw = jobs[run_id]["raw_attempt_parent"] + "/" + chosen["attempt"]
        completion = read(_at(root, chosen["report"]))
        capture = completion["endpoint"]["capture"]
        require(capture["input_sha256"] == rgb_meta["sha256"] and capture["prior_cache_files"] == 0
                and capture["inference_precision"] == "float32", "Raw forward RGB/precision/freshness differs")
        expected = {f"endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth" for i, j in PAIRS}
        actual = {name for name in chosen["raw_files"] if name.startswith("endpoint_cache/forward/")}
        require(actual == expected, "Missing, extra or incorrectly named raw forward pair")
        pairs = []
        for i, j in PAIRS:
            name = f"endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth"
            pairs.append(dict(i=i, j=j, **_descriptor(root, raw + "/" + name, chosen["raw_files"][name])))
        functional.append(dict(run_id=run_id, pairs=pairs, instances=names,
                               role="same_state_comparator" if run_id == FUNCTIONAL_RUNS[-1] else "primary_comparison"))
    first, second = evidence[FUNCTIONAL_RUNS[0]], evidence[FUNCTIONAL_RUNS[-1]]
    for field in ("original_state_sha256", "checkpoint_sha256", "docker_image", "runtime_policy", "input_sha256"):
        require(first[field] == second[field], "P0 raw same-state comparator differs: " + field)
    for run_id in (FUNCTIONAL_RUNS[0], FUNCTIONAL_RUNS[-1]):
        require(selected[run_id]["adapter"].get("frozen_P0") is True, "P0 comparator contains an adapter")
    coordinates = dict(**_descriptor(root, SPLITS, registry["input_sha256"][SPLITS]),
                       members=list(XY_MEMBERS), frames=list(FRAMES), scene="heads", source="earlier")
    return dict(schema="profiled_d0_artifact_inventory_v1", design=design_descriptor,
        trigger=dict(seal=seal_descriptor, scores=score_descriptor, replay=replay_descriptor),
        parent_registry=registry_descriptor, parent_evidence=evidence,
        evaluation_registry=evaluation_descriptor, evaluation_source_sha256_lf=evaluation_sources,
        parent_source_sha256_lf=registry["source_sha256_lf"], upstream_sources=upstream,
        checkpoint=checkpoint, config=config, config_hash_is_historical_M2_proof=False,
        rgb_plan=plan_descriptor, rgb_archive=rgb_descriptor,
        layout=LAYOUT.copy(), module_names=list(MODULES), adapters=adapters,
        functional=functional, coordinates=coordinates, arrays_deserialized=False,
        executable_registration_required=True, execution_authorized=False)


def verify_inventory(root, weights_dir, inventory):
    """Rebuild using bytes/JSON only; exact equality includes membership/order."""
    require(isinstance(inventory, dict) and inventory.get("schema") == "profiled_d0_artifact_inventory_v1",
            "Wrong D0 inventory schema")
    current = build_inventory(root, inventory["design"]["file"], weights_dir)
    require(current == inventory, "D0 inventory changed; do not decode tensors")
    return current


def _archive_members(archive):
    names = archive.files
    require(len(names) == len(set(names)), "Duplicate NPZ members")
    return set(names)


def load_adapter(root, descriptor):
    """Decode one registered adapter, preserving nonfinite factors for diagnostics."""
    import numpy as np
    path = _verified_path(root, descriptor)
    require(descriptor.get("modules") == 144 and descriptor.get("factor_arrays") == 288
            and descriptor.get("rank") == 4 and descriptor.get("alpha") == 1,
            "Adapter descriptor has wrong factor contract")
    expected = {module + suffix for module in MODULES for suffix in (".lora_A", ".lora_B")}
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        require(_archive_members(archive) == expected, "Missing or extra adapter factors")
        for module in MODULES:
            a, b = archive[module + ".lora_A"], archive[module + ".lora_B"]
            width = 2304 if module.endswith("attn.qkv") else 768
            require(a.shape == (4, 768) and b.shape == (width, 4), "Incorrect factor orientation/shape: " + module)
            require(a.dtype == b.dtype == np.dtype("float32"), "Expected recorded FP32 adapter factors")
            result[module] = dict(A=a.astype(np.float64), B=b.astype(np.float64), base_key=module + ".weight")
    _unchanged(path, descriptor)
    return result


def load_forward(root, descriptor):
    """Read the four-tuple safely on CPU; return only first-camera axial depth."""
    import numpy as np
    import torch
    path = _verified_path(root, descriptor)
    require((descriptor.get("i"), descriptor.get("j")) in PAIRS, "Invalid directed pair descriptor")
    values = torch.load(path, map_location="cpu", weights_only=True)
    require(type(values) is tuple and len(values) == 4, "Expected exact forward four-tuple")
    for value, shape in zip(values, ((384, 512, 3), (384, 512), (384, 512, 3), (384, 512))):
        require(type(value) is torch.Tensor and value.device.type == "cpu" and
                value.dtype == torch.float32 and tuple(value.shape) == shape,
                "Raw forward dtype/device/shape differs from writer contract")
    result = values[0][..., 2].numpy().astype(np.float64, copy=True)
    _unchanged(path, descriptor)
    return result


def load_xy(root, descriptor):
    """Decode exactly five Heads/earlier coordinate members; no depth/pose packet."""
    import numpy as np
    require(descriptor.get("members") == list(XY_MEMBERS) and descriptor.get("frames") == list(FRAMES)
            and descriptor.get("scene") == "heads" and descriptor.get("source") == "earlier",
            "Wrong adaptation-coordinate allowlist")
    path = _verified_path(root, descriptor)
    result = []
    with np.load(path, allow_pickle=False) as archive:
        names = _archive_members(archive)
        require(set(XY_MEMBERS) <= names, "Missing authorized coordinate member")
        for member in XY_MEMBERS:
            value = archive[member]
            require(value.ndim == 2 and value.shape[1] == 2 and value.shape[0] >= 16
                    and np.issubdtype(value.dtype, np.integer), "Expected integer [N,2] adaptation xy")
            require(bool(((value >= 0) & (value < [512, 384])).all()), "Adaptation coordinates outside raster")
            result.append(value.astype(np.int64, copy=True))
    _unchanged(path, descriptor)
    return result


def load_base_weight_norms(weights_dir, checkpoint_descriptor, module_shapes):
    """Return base Frobenius norms, streaming one FP32 safetensors matrix at a time.

Call once, then reuse the norms across all 30 adapters. module_shapes maps the
authorized 144 module names to [out_features,in_features]. The file is hashed
before and after this complete stream, not once per matrix.
"""
    import numpy as np
    from safetensors import safe_open
    require(set(module_shapes) == set(MODULES), "Base-weight norm request must cover exactly 144 adapted matrices")
    for module, shape in module_shapes.items():
        require(list(shape) == [2304 if module.endswith("attn.qkv") else 768, 768],
                "Base-weight expected shape differs")
    path = _verified_path(weights_dir, checkpoint_descriptor)
    require(checkpoint_descriptor["file"] == "model.safetensors", "Unexpected checkpoint file")
    result = {}
    with safe_open(str(path), framework="numpy", device="cpu") as archive:
        keys = set(archive.keys())
        require({module + ".weight" for module in MODULES} <= keys, "Missing original adapted matrix key")
        for module in MODULES:
            value = archive.get_tensor(module + ".weight")
            require(value.dtype == np.dtype("float32") and list(value.shape) == list(module_shapes[module]),
                    "Frozen checkpoint matrix dtype/shape differs: " + module)
            require(bool(np.isfinite(value).all()), "Frozen checkpoint matrix is nonfinite")
            # Float64 scalar reduction; do not materialize the entire checkpoint.
            norm2 = np.einsum("ij,ij->", value, value, dtype=np.float64, optimize=False)
            require(math.isfinite(float(norm2)) and norm2 >= 0, "Invalid frozen-weight norm")
            result[module] = math.sqrt(float(norm2))
            del value
    _unchanged(path, checkpoint_descriptor)
    return result
