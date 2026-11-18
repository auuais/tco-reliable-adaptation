"""Byte-only audit of one completed M3 model attempt, never a score release.

This verifies registered provenance, saved metadata consistency, exact file
inventories and byte hashes without decoding NPZ/PTH/tensor files. Reported
tensor-state digests are cross-checked, not recomputed from tensor contents.
Actual container identity, successful exit, non-overlap and process freshness
across attempts remain obligations of the external controller/global sealer.
The prospective registry binds checkpoint/configuration identities; live host
weight files are not required to audit an archived attempt.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

from profiled_artifacts import logical_inventory, resolve_logical
from profiled_m3_model import _job, INPUT_PLAN, INPUT_ATTEMPT, RGB_MANIFEST


SELF = "scripts/profiled_m3_attempt_audit.py"
BASELINE_GRAPH = "results/lora_profiled_v1/baseline_graph.json"
MATCH_FOLDER = "corres_conf=desc_conf_subsample=8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def sha(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _digest(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA256: " + label)


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer: " + label)
    return value


def _number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            "Invalid finite nonnegative/positive number: " + label)
    return value


def _vector(value, label, length=5, integers=False, minimum=0, positive=False):
    require(isinstance(value, list) and len(value) == length, "Invalid vector: " + label)
    for item in value:
        _integer(item, label, minimum) if integers else _number(item, label, positive)
    return value


def _time(value):
    require(isinstance(value, str), "Missing explicit UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, "Timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def _file(root, name):
    return resolve_logical(root, name, must_exist=True)


def _inventory(directory, exclude=()):
    return {name: sha(path) for name, path in logical_inventory(directory).items() if name not in exclude}


def _identity(record, registry, registry_file, registry_sha, job, attempt, raw_name, public_name):
    core = registry["model_core"]
    require(record.get("schema") == "profiled_m3_model_attempt_v1" and record.get("job") == job and
            record.get("job_id") == job["job_id"] and record.get("attempt") == attempt,
            "Attempt/job identity differs")
    require(record.get("registry_file") == registry_file and record.get("registry_sha256") == registry_sha,
            "Attempt uses another execution registry")
    for key in ("source_sha256_lf", "artifact_sha256", "docker_image", "checkpoint_sha256",
                "checkpoint_config_sha256", "runtime_identity"):
        require(record.get(key) == core[key], "Attempt binding differs: " + key)
    case = core["cases"][job["case_id"]]
    require(record.get("input_sha256") == case["input_sha256"] and
            record.get("case_input_digest") == case["case_input_digest"], "Selected fitting input identities differ")
    require(record.get("input_plan") == INPUT_PLAN and record.get("input_attempt") == INPUT_ATTEMPT and
            record.get("runtime_policy") == core["runtime_identity"]["policy"] and record.get("seed") == 42,
            "Input preparation or deterministic arithmetic contract differs")
    require(record.get("raw_attempt") == raw_name and record.get("public_attempt") == public_name,
            "Attempt output path differs from its canonical M3 scope")
    require(isinstance(record.get("execution_commit"), str) and
            re.fullmatch(r"[0-9a-f]{40}", record["execution_commit"]) is not None, "Missing actual execution commit")
    for key in ("held_source_development_arrays_loaded", "target_depth_arrays_loaded",
                "accuracy_scoring_performed", "stage_release_authorized"):
        require(record.get(key) is False, "Attempt exceeds its unscored model-execution scope: " + key)
    require(type(record.get("rental_compute_usd")) in (int, float) and record["rental_compute_usd"] == 0,
            "This model protocol does not authorize rental compute")
    _integer(record.get("disk_free_bytes_before"), "free space", 1024 ** 3)


def _process(started, complete):
    first, last = started["process"], complete["process"]
    require(isinstance(first, dict) and isinstance(last, dict), "Missing process identity")
    keys = {"pid", "hostname", "process_run_uuid", "model_initialization_uuid", "model_constructions"}
    require(set(first) == keys and set(last) == keys | {"model_constructed_at_utc"}, "Unexpected process identity fields")
    _integer(first["pid"], "PID", 1)
    require(isinstance(first["hostname"], str) and bool(first["hostname"].strip()), "Missing host identity")
    for key in ("process_run_uuid", "model_initialization_uuid"):
        require(isinstance(first[key], str) and re.fullmatch(r"[0-9a-f]{32}", first[key]) is not None,
                "Invalid fresh process/model UUID")
    require(first["process_run_uuid"] != first["model_initialization_uuid"], "Process and model UUIDs must be distinct")
    require(type(first["model_constructions"]) is int and first["model_constructions"] == 0 and
            type(last["model_constructions"]) is int and last["model_constructions"] == 1,
            "Expected zero models at start and exactly one fresh model construction")
    require(all(first[key] == last[key] for key in keys - {"model_constructions"}), "Process identity changed during attempt")
    require(_time(started["started_at_utc"]) <= _time(last["model_constructed_at_utc"]) <=
            _time(complete["completed_at_utc"]), "Invalid start/model/finish chronology")
    return last


def _steps(record, public, job):
    trained = job["arm"] != "P0"
    expected = 8 if trained else 0
    steps = record.get("steps")
    require(isinstance(steps, list) and len(steps) == expected and
            type(record.get("optimizer_updates")) is int and record["optimizer_updates"] == expected,
            "Incomplete optimizer/step inventory")
    previous = record.get("adapter_initial_sha256")
    for index, row in enumerate(steps):
        require(_read(public / f"step_{index:02d}.json") == row, "Step file differs from completion trace")
        for key, value in (("step", index), ("optimizer_steps", 1), ("optimizer_updates_cumulative", index + 1),
                           ("graph_free_decoder_calls", 20), ("differentiable_decoder_calls", 20)):
            require(type(row.get(key)) is int and row[key] == value, "Step schedule/count differs: " + key)
        require(row.get("support_fold") is None if job["arm"] == "P1" else
                type(row.get("support_fold")) is int and row["support_fold"] == index % 2, "Support fold schedule differs")
        if job["arm"] == "P1":
            require(row.get("scales_before_update") is None, "P1 does not use the profiled five-scale objective")
        else:
            _vector(row.get("scales_before_update"), "profiled scales", positive=True)
        for key in ("loss_before_update", "seconds", "peak_allocated_mib", "peak_reserved_mib"):
            _number(row.get(key), key)
        for key in ("adapter_update_l2", "unclipped_gradient_norm"):
            _number(row.get(key), key, positive=True)
        require(row.get("gradient_clip_norm") == 1., "Gradient clipping differs")
        gradient = row.get("gradient", {})
        _number(gradient.get("norm"), "gradient norm", positive=True)
        count = _integer(gradient.get("parameters_with_gradient"), "parameters with gradient", 1)
        active = _integer(gradient.get("parameters_with_nonzero_gradient"), "active parameter count", 1)
        require(active <= count <= 288, "Gradient counts exceed the registered 144 two-factor modules")
        names, missing = gradient.get("active_names"), gradient.get("missing_names")
        require(isinstance(names, list) and isinstance(missing, list) and len(names) == active and len(missing) == 288 - count,
                "Gradient names/counts differ")
        require(all(isinstance(name, str) and name.endswith((".lora_A", ".lora_B")) for name in names + missing),
                "Non-LoRA gradient identity")
        require(len(names) == len(set(names)) and len(missing) == len(set(missing)) and not set(names) & set(missing),
                "Duplicate/conflicting gradient names")
        _digest(row.get("adapter_state_sha256"), "updated adapter state")
        require(row["adapter_state_sha256"] != previous, "Positive update lacks changed adapter identity")
        previous = row["adapter_state_sha256"]
    expected_counts = dict(teacher_decoder_calls=20 if trained else 0,
                           training_graph_free_decoder_calls=160 if trained else 0,
                           training_differentiable_decoder_calls=160 if trained else 0, endpoint_decoder_calls=20)
    for key, value in expected_counts.items():
        require(type(record.get(key)) is int and record[key] == value, "Total decoder count differs: " + key)
    if trained:
        _digest(record.get("adapter_initial_sha256"), "initial adapter")
        _digest(record.get("adapter_final_sha256"), "final adapter")
        require(record["adapter_final_sha256"] == previous and record.get("adapter_module_count") == 144 and
                record.get("trainable_parameters") == 1032192, "Adapter inventory or final identity differs")
    else:
        require(record.get("adapter_initial_sha256") is None and record.get("adapter_final_sha256") is None and
                record.get("adapter_module_count") == record.get("trainable_parameters") == 0, "P0 has an adapter")
    return steps


def _native_graph(root, raw, public, endpoint, rgb):
    graph_path = public / "endpoint/native_graph.json"
    require(endpoint.get("graph_file") == graph_path.relative_to(root).as_posix() and
            endpoint.get("graph_sha256") == sha(graph_path), "Native graph record/bytes differ")
    graph = _read(graph_path)
    names = [f"{rgb['key']}_{rgb['sha256'][:12]}_{index}" for index in range(5)]
    ids = [hashlib.md5(name.encode()).hexdigest() for name in names]
    require(graph.get("names") == names and graph.get("directed_forward_files") == graph.get("directed_match_files") == 20,
            "Native graph image/pair inventory differs")
    cache = raw / "endpoint_cache"
    cache_files = _inventory(cache)
    require(graph.get("cache_files") == cache_files, "Fresh cache bytes/inventory differ from graph receipt")
    forward = {f"forward/{first}/{second}.pth" for first, second in itertools.permutations(ids, 2)}
    matches = {f"{MATCH_FOLDER}/{first}-{second}.pth" for first, second in itertools.permutations(ids, 2)}
    canonical = graph.get("canonical_files")
    require(isinstance(canonical, list) and len(canonical) == 5, "Exactly five canonical cache files are required")
    canonical_names = set()
    for index, item in enumerate(canonical):
        path = _file(root, item["file"])
        require(path.parent == cache / "canon_views" and path.name.startswith(ids[index] + "_subsample=8_") and
                path.suffix == ".pth" and item.get("sha256") == sha(path), "Canonical file identity/bytes differ")
        # Keep Linux logical names in this set. Windows exposes ':' as U+F03A,
        # so a physical relative filename is not the registered logical name.
        cache_scope = cache.relative_to(root).as_posix() + "/"
        require(item["file"].startswith(cache_scope), "Canonical file escapes this cache scope")
        canonical_names.add(item["file"][len(cache_scope):])
    require(len(canonical_names) == 5 and set(cache_files) == forward | matches | canonical_names,
            "Native cache must contain exactly 20 forward, 20 match and five canonical files")
    edges = graph.get("edges")
    expected_edges = list(itertools.combinations(range(5), 2))
    require(isinstance(edges, list) and [(edge.get("i"), edge.get("j")) for edge in edges] == expected_edges,
            "Native unordered edge inventory differs")
    for edge in edges:
        i, j = edge["i"], edge["j"]
        _integer(edge.get("count"), "native correspondence count")
        _digest(edge.get("pixel_pairs_sha256_int64_le"), "native pixel-pair tensor identity")
        for prefix, first, second in (("", i, j), ("reverse_", j, i)):
            name = (cache / MATCH_FOLDER / f"{ids[first]}-{ids[second]}.pth").relative_to(root).as_posix()
            require(edge.get(prefix + "file") == name and edge.get(prefix + "sha256") == sha(_file(root, name)),
                    "Native edge points to another attempt or changed bytes")
    require(graph.get("canonical_ratio_used_as_depth_or_confidence") is False and
            graph.get("fused_confidence_source") == "third tensor of ((canon, canon2, cconf), focal) in this fresh native canonical cache",
            "Native confidence convention differs")
    return cache_files


def _endpoints(root, raw, public, record, registry, job):
    endpoint = record.get("endpoint", {})
    rgb = _read(_file(root, RGB_MANIFEST))["inputs"][job["scene"] + "_base"]
    rgb = dict(rgb, key=job["scene"] + "_base")
    for kind in ("native", "final"):
        name = (raw / "endpoint" / (kind + ".npz")).relative_to(root).as_posix()
        require(endpoint.get(kind + "_file") == name and endpoint.get(kind + "_sha256") == sha(_file(root, name)),
                "Native/final endpoint identity or bytes differ")
    capture = endpoint.get("capture", {})
    require(_read(public / "endpoint/native.json") == capture, "Native capture file differs")
    require(capture.get("input_sha256") == rgb["sha256"] and capture.get("output_sha256") == endpoint["native_sha256"],
            "Native capture input/output binding differs")
    require(capture.get("prior_cache_files") == 0 and capture.get("pairs") == 20 and
            capture.get("coarse_iterations") == capture.get("fine_iterations") == 300 and
            capture.get("native_size") == [384, 512] and capture.get("inference_precision") == "float32" and
            capture.get("device") == registry["model_core"]["runtime_identity"]["gpu"], "Fresh native endpoint protocol differs")
    _number(capture.get("seconds"), "native capture seconds")
    _number(capture.get("peak_allocated_mib"), "native capture memory")
    invalid = _vector(capture.get("nonpositive_depth_pixels"), "native nonpositive counts", integers=True)
    require(all(count <= 384 * 512 for count in invalid), "Native invalid count exceeds raster")
    counts = _vector(endpoint.get("sample_counts"), "native fitting counts", integers=True, minimum=16)
    bridge = _read(_file(root, registry["prerequisites"]["case_bridge"]["file"]))
    case_rows = [row for row in bridge["cases"] if
                 tuple(row[key] for key in ("scene", "side", "grid", "pose_condition")) ==
                 tuple(job[key] for key in ("scene", "side", "grid", "pose"))]
    require(len(case_rows) == 1 and counts == case_rows[0]["counts"],
            "Endpoint fitting counts differ from the registered CPU case bridge")
    require(endpoint.get("original_prior_rows_preserved") is True and endpoint.get("final_compute") == "CPU float64",
            "Common endpoint prior/final arithmetic differs")
    trace = _read(public / "endpoint/native_trace.json")
    require(isinstance(trace, list) and len(trace) == 300 and all(type(event.get("iteration")) is int and event["iteration"] == index and
            event.get("finite_gradient") is True and type(event.get("received_gradient")) in (int, float) and
            event["received_gradient"] == 1. for index, event in enumerate(trace)),
            "Native prior hook lacks all 300 fine unit-gradient iterations")
    final = endpoint.get("final", {})
    require(_read(public / "endpoint/final.json") == final, "Final refinement file differs")
    require(final.get("sample_counts") == counts and final.get("support_counts") == counts and
            final.get("gauge_ratio_count") == sum(counts) and final.get("sample_exclusions") == 0 and
            final.get("nonpositive_fit_predictions") == [0] * 5 and final.get("nonpositive_dense_pixels") == invalid,
            "Final fit changed allowed observation coverage or validity accounting")
    _number(final.get("g0"), "global metric gauge", positive=True)
    _vector(final.get("scales"), "relative depth scales", positive=True)
    require(final.get("metric_scales") == [final["g0"]] * 5 and final.get("positive_scales") is True and
            final.get("graph_weight") == .1 and final.get("fit_is_accuracy_metric") is False and
            final.get("gauge_estimator") == "pooled positive measured/native ratios; midpoint median" and
            final.get("endpoint") == "native KDP followed by all-allowed-observation five-scale refinement",
            "Final metric gauge/refinement convention differs")
    _vector(final.get("sensor_normalizers"), "final sensor normalizers", positive=True)
    _number(final.get("ridge"), "final ridge", positive=True)
    cache_files = _native_graph(root, raw, public, endpoint, rgb)
    return endpoint, counts, cache_files


def _teacher(root, raw, public, record, counts, registry, job):
    teacher = _read(public / "teacher.json")
    require(teacher.get("decoder_calls") == 20 and teacher.get("checkpoint_sha256") == record["checkpoint_sha256"] and
            teacher.get("initial_adapter_state_sha256") == record["adapter_initial_sha256"], "Teacher initialization differs")
    name = (raw / "teacher_initial.npz").relative_to(root).as_posix()
    require(teacher.get("fused_native_file") == name and teacher.get("fused_native_sha256") == sha(_file(root, name)),
            "Teacher fused raster artifact differs")
    require(BASELINE_GRAPH in registry["model_core"]["artifact_sha256"] and teacher.get("graph_audit") ==
            _read(_file(root, BASELINE_GRAPH))["scenes"][job["scene"]], "Teacher matching graph differs from sealed pretrained audit")
    require(teacher.get("confidence_and_graph_frozen") is True and teacher.get("full_pair_pointmaps_archived") is False,
            "Teacher confidence/graph contract differs")
    states = teacher.get("states")
    require(isinstance(states, list) and len(states) == 2, "Teacher needs both support/query folds")
    for fold, state in enumerate(states):
        require(state.get("support_fold") == fold, "Teacher fold ordering differs")
        _vector(state.get("gains"), "teacher gains", positive=True)
        _vector(state.get("sensor_normalizers"), "teacher sensor normalizers", positive=True)
        _number(state.get("ridge"), "teacher ridge", positive=True)
        support = _vector(state.get("support_counts"), "support counts", integers=True, minimum=8)
        query = _vector(state.get("query_counts"), "query counts", integers=True, minimum=8)
        require([first + second for first, second in zip(support, query)] == counts, "Support/query lost allowed measurements")
    require(states[0]["support_counts"] == states[1]["query_counts"] and
            states[1]["support_counts"] == states[0]["query_counts"], "Alternating folds do not partition the same packet")
    gauges = teacher.get("legacy_pair_gauges", {})
    require(set(gauges) == {f"{i}_{j}" for i, j in itertools.permutations(range(5), 2)}, "Legacy gauge pair inventory differs")
    for value in gauges.values():
        _number(value, "legacy pair gauge", positive=True)


def audit_m3_complete(root, registry_file, job_id, attempt):
    """Audit one complete model state; return a canonical byte-evidence receipt."""
    from profiled_m3_registry import verify_execution

    root = Path(root).resolve()
    require(isinstance(job_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", job_id) is not None and
            isinstance(attempt, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", attempt) is not None,
            "Invalid job/attempt path identity")
    registry_path = _file(root, registry_file)
    strict_registry = _read(registry_path)
    verified = verify_execution(root, registry_file, weights=None)
    registry, registry_hash = verified["registry"], verified["registry_sha256"]
    require(registry == strict_registry and registry_hash == sha(registry_path), "Registry verification bytes disagree")
    jobs = [job for job in registry["jobs"] if job["job_id"] == job_id]
    require(len(jobs) == 1, "Attempt job is not uniquely registered")
    job = jobs[0]
    _job(job)
    suffix = job_id + "/" + attempt
    raw_name, public_name = ("reports/lora_profiled_m3_v1/cases/" + suffix,
                             "results/lora_profiled_m3_v1/cases/" + suffix)
    raw, public = resolve_logical(root, raw_name), resolve_logical(root, public_name)
    require(raw.is_dir() and public.is_dir(), "Partial attempt directory pair")
    require((public / "complete.json").is_file() and not (public / "failure.json").exists(),
            "Partial attempt or conflicting failure/completion")
    started, complete = _read(public / "started.json"), _read(public / "complete.json")
    for record in (started, complete):
        _identity(record, registry, registry_file, registry_hash, job, attempt, raw_name, public_name)
    require(all(complete.get(key) == value for key, value in started.items() if key != "process"),
            "Completion changed immutable start metadata")
    require(complete.get("status") == "model_outputs_complete_without_scoring" and complete.get("pass_all") is True and
            complete.get("accuracy_improvement_established") is False and complete.get("original_state_unchanged") is True,
            "Not an unscored successful model completion")
    process = _process(started, complete)
    original = registry["model_core"]["original_state_sha256"]
    require(complete.get("original_state_sha256") == complete.get("pre_endpoint_original_state_sha256") ==
            complete.get("final_original_state_sha256") == original, "Frozen original state digests differ")
    raw_files, public_files = _inventory(raw), _inventory(public, exclude=("complete.json",))
    require(complete.get("raw_files") == raw_files and complete.get("public_files") == public_files,
            "Raw/public bytes or file inventories changed after completion")
    steps = _steps(complete, public, job)
    endpoint, counts, cache_files = _endpoints(root, raw, public, complete, registry, job)
    expected_raw = {"endpoint/native.npz", "endpoint/final.npz"} | {"endpoint_cache/" + name for name in cache_files}
    expected_public = {"started.json", "endpoint/native.json", "endpoint/final.json", "endpoint/native_graph.json", "endpoint/native_trace.json"}
    if job["arm"] != "P0":
        _teacher(root, raw, public, complete, counts, registry, job)
        expected_raw |= {"teacher_initial.npz", "adapters.npz"}
        expected_public |= {"teacher.json"} | {f"step_{index:02d}.json" for index in range(8)}
    require(set(raw_files) == expected_raw and set(public_files) == expected_public, "Unexpected/partial model artifact coverage")
    for key in ("seconds", "training_and_teacher_seconds", "endpoint_seconds", "preparation_peak_allocated_mib",
                "preparation_peak_reserved_mib", "endpoint_peak_reserved_mib", "peak_allocated_mib", "peak_reserved_mib"):
        _number(complete.get(key), key)
    require(complete["training_and_teacher_seconds"] + complete["endpoint_seconds"] <= complete["seconds"],
            "Phase durations exceed total measured runtime")
    require(sum(row["seconds"] for row in steps) <= complete["training_and_teacher_seconds"] + 1e-9,
            "Step durations exceed training duration")
    require(complete["peak_allocated_mib"] >= max([complete["preparation_peak_allocated_mib"], endpoint["capture"]["peak_allocated_mib"]]
            + [row["peak_allocated_mib"] for row in steps]), "Total allocated peak omits a recorded phase")
    require(complete["peak_reserved_mib"] == max([complete["preparation_peak_reserved_mib"], complete["endpoint_peak_reserved_mib"]]
            + [row["peak_reserved_mib"] for row in steps]), "Total reserved peak differs from recorded phases")
    receipt = dict(schema="profiled_m3_model_completion_byte_audit_v1", registry_file=registry_file,
        registry_sha256=registry_hash, job_id=job_id, attempt=attempt, job=job,
        complete_file=public_name + "/complete.json", complete_sha256=sha(public / "complete.json"),
        started_file=public_name + "/started.json", started_sha256=sha(public / "started.json"),
        raw_files=raw_files, public_files=dict(public_files, **{"complete.json": sha(public / "complete.json")}),
        native_file=endpoint["native_file"], native_sha256=endpoint["native_sha256"],
        final_file=endpoint["final_file"], final_sha256=endpoint["final_sha256"],
        input_sha256=complete["input_sha256"], case_input_digest=complete["case_input_digest"],
        checkpoint_sha256=complete["checkpoint_sha256"], checkpoint_config_sha256=complete["checkpoint_config_sha256"],
        original_state_sha256=original, adapter_initial_sha256=complete["adapter_initial_sha256"],
        adapter_final_sha256=complete["adapter_final_sha256"], process=process,
        started_at_utc=started["started_at_utc"], completed_at_utc=complete["completed_at_utc"],
        optimizer_updates=complete["optimizer_updates"], endpoint_decoder_calls=complete["endpoint_decoder_calls"],
        seconds=complete["seconds"], pass_all=True, arrays_deserialized=False,
        tensor_state_digests_independently_recomputed=False, live_weight_files_verified=False,
        actual_container_identity_verified=False, global_process_freshness_verified=False,
        stage_release_authorized=False, accuracy_scoring_performed=False,
        auditor_source_sha256_lf=sha(Path(__file__), text=True))
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt
