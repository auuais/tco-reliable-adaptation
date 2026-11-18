"""Fresh pristine/zero-LoRA native KDP and final endpoint equivalence check.

This is an engineering bridge, not an accuracy experiment. It reads only Heads
earlier adaptation observations. Each variant receives a separate empty cache,
300/300 native iterations, identical priors, and the common final calibration.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np
import torch

from profiled_final_calibration import refine_native


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/lora_profiled_v1"
RAW = ROOT / "reports/lora_profiled_v1"
FRAMES = (25, 225, 425, 625, 825)
MATCH_FOLDER = "corres_conf=desc_conf_subsample=8"


def dump_new(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def native_prior_samples(case):
    """Load original allowed rows, checking exact native pixel/value equivalence."""
    from prepare_profiled_inputs import native_selection, sha

    packet_path = ROOT / f"results/mast3r_partner_shape/{case['side']}.npz"
    split_path = PUBLIC / "splits.npz"
    if sha(packet_path) != case["packet_sha256"] or sha(split_path) != case["splits_sha256"]:
        raise ValueError("Permitted prior packet or sealed native selection changed")
    selected = []
    with np.load(packet_path, allow_pickle=False) as packet, np.load(split_path, allow_pickle=False) as split:
        for view, frame in enumerate(FRAMES):
            key = f"{case['scene']}_{case['side']}_{frame}__rows"
            values = packet[f"{case['scene']}_{frame}"][split[key]].copy()
            rows, xy, outside, duplicates = native_selection(values)
            original_xy = case["samples"][view]["xy"].detach().cpu().numpy()
            original_depth = case["samples"][view]["depth"].detach().cpu().numpy()
            selected_ids = xy[:, 1] * 512 + xy[:, 0]
            expected_ids = original_xy[:, 1] * 512 + original_xy[:, 0]
            actual_order, expected_order = np.argsort(selected_ids), np.argsort(expected_ids)
            if outside or duplicates or len(rows) != len(original_xy):
                raise ValueError("Original allowed packet changed native fitting coverage")
            np.testing.assert_array_equal(xy[actual_order], original_xy[expected_order])
            np.testing.assert_array_equal(values[rows, 2][actual_order], original_depth[expected_order])
            selected.append(values)
    return selected


def load_fresh_graph(cache, names):
    """Read native correspondences and fused point confidence, never canon2.

The canonical cache's third tensor is exactly the fused point confidence written
    by pinned canonical_view. Its second tensor is the depth ratio canon2 and is
    deliberately not used. All paths must belong to this fresh endpoint cache.
    """
    from prepare_profiled_inputs import sha

    ids = [hashlib.md5(name.encode()).hexdigest() for name in names]
    expected_matches = {f"{first}-{second}.pth" for first, second in itertools.permutations(ids, 2)}
    actual_matches = {path.name for path in (cache / MATCH_FOLDER).glob("*.pth")}
    if actual_matches != expected_matches:
        raise ValueError("Fresh native graph does not contain exactly 20 directed match files")
    expected_forward = {f"{first}/{second}.pth" for first, second in itertools.permutations(ids, 2)}
    actual_forward = {path.relative_to(cache / "forward").as_posix() for path in (cache / "forward").rglob("*.pth")}
    if actual_forward != expected_forward:
        raise ValueError("Fresh native inference does not contain exactly 20 forward files")
    confidence, canonical_files = [], []
    for name in ids:
        paths = list((cache / "canon_views").glob(name + "_subsample=8_*.pth"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one fresh native canonical cache per view")
        (canonical, ratio, fused), focal = torch.load(paths[0], map_location="cpu", weights_only=True)
        if canonical.shape != (384, 512, 3) or ratio.shape != (384, 512) or fused.shape != (384, 512):
            raise ValueError("Unexpected fresh native canonical cache schema")
        if not torch.isfinite(fused).all() or not (fused > 0).all():
            raise ValueError("Invalid native fused point confidence")
        confidence.append(fused.detach().clone())
        canonical_files.append(dict(file=paths[0].relative_to(ROOT).as_posix(), sha256=sha(paths[0])))
    matches, edge_records = [], []
    for i, j in itertools.combinations(range(5), 2):
        first = cache / MATCH_FOLDER / f"{ids[i]}-{ids[j]}.pth"
        reverse = cache / MATCH_FOLDER / f"{ids[j]}-{ids[i]}.pth"
        score, (xi, xj, conf) = torch.load(first, map_location="cpu", weights_only=True)
        back_score, (back_j, back_i, back_conf) = torch.load(reverse, map_location="cpu", weights_only=True)
        if score != back_score or not torch.equal(xi, back_i) or not torch.equal(xj, back_j) or not torch.equal(conf, back_conf):
            raise ValueError("Fresh native graph reverse correspondence copies differ")
        if xi.dtype != torch.int64 or xj.dtype != torch.int64 or xi.ndim != 2 or xi.shape != xj.shape or xi.shape[1] != 2:
            raise ValueError("Fresh native correspondences have incompatible coordinates")
        if int(score[2]) != len(xi) or conf.shape != (len(xi),):
            raise ValueError("Fresh native match count differs")
        joined = torch.cat([xi, xj], dim=1)
        if len(torch.unique(joined, dim=0)) != len(joined):
            raise ValueError("Duplicate fresh native pixel correspondences")
        for xy in (xi, xj):
            if not ((xy >= 0) & (xy < torch.tensor([512, 384]))).all():
                raise ValueError("Fresh native match lies outside the raster")
        numeric_sha = hashlib.sha256(joined.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
        matches.append(dict(i=i, j=j, xy_i=xi.clone(), xy_j=xj.clone()))
        edge_records.append(dict(i=i, j=j, count=len(xi), pixel_pairs_sha256_int64_le=numeric_sha,
                                 file=first.relative_to(ROOT).as_posix(), sha256=sha(first),
                                 reverse_file=reverse.relative_to(ROOT).as_posix(), reverse_sha256=sha(reverse)))
    return matches, torch.stack(confidence), dict(
        names=names, canonical_files=canonical_files, edges=edge_records,
        directed_forward_files=len(actual_forward), directed_match_files=len(actual_matches),
        fused_confidence_source="third tensor of ((canon, canon2, cconf), focal) in this fresh native canonical cache",
        canonical_ratio_used_as_depth_or_confidence=False,
        cache_files={path.relative_to(cache).as_posix(): sha(path) for path in sorted(cache.rglob("*.pth"))},
    )


def capture_endpoint(model, case, cache, raw, public):
    """Capture native KDP plus final refinement for one adapter state."""
    import mast3r.cloud_opt.sparse_ga as alignment
    from mast3r_alignment_hook import optimizer_hook
    from mast3r_calibrated_alignment import CalibratedPriors
    from mast3r_network_capture import capture, views_for
    from prepare_dense_focal_pilot import write_arrays
    from prepare_profiled_inputs import sha

    if cache.exists() or raw.exists() or public.exists():
        raise FileExistsError("Each endpoint needs unused cache, raw and public directories")
    original_samples = native_prior_samples(case)
    poses = case["w2c"].detach().cpu().numpy()
    intrinsic = case["K"].detach().cpu().numpy()
    if intrinsic.shape != (3, 3):
        raise ValueError("Registered KDP bridge expects the shared supplied 3x3 intrinsic")

    def factory(arguments):
        return CalibratedPriors(alignment, arguments, original_samples, poses, intrinsic, .01)

    with optimizer_hook(alignment, factory) as callbacks:
        metadata = capture(model, case["scene"] + "_base", cache, raw, public, "native")
    if metadata["prior_cache_files"] != 0 or metadata["coarse_iterations"] != 300 or metadata["fine_iterations"] != 300:
        raise ValueError("Endpoint did not use the registered fresh 300/300 native alignment")
    if len(callbacks) != 1 or len(callbacks[0].trace) != 300:
        raise ValueError("Native KDP prior hook did not execute all 300 fine iterations")
    if not all(event.get("finite_gradient") and event.get("received_gradient") == 1. for event in callbacks[0].trace):
        raise ValueError("Native KDP priors did not receive finite unit loss gradients")
    counts = [len(sample["xy"]) for sample in case["samples"]]
    if callbacks[0].counts != counts:
        raise ValueError("Native KDP fitting sample counts differ from the sealed selection")
    dump_new(public / "native_trace.json", callbacks[0].trace)
    _, names, _ = views_for(case["scene"] + "_base")
    matches, confidence, graph = load_fresh_graph(cache, names)
    dump_new(public / "native_graph.json", graph)
    with np.load(raw / "native.npz", allow_pickle=False) as archive:
        native = {name: archive[name].copy() for name in archive.files}
    if set(native) != {"depth", "intrinsics", "c2w"}:
        raise ValueError("Unexpected native output fields")
    # The tiny final solve is deliberately CPU float64 in this endpoint path.
    final_depth, metric_cameras, scales, diagnostics = refine_native(
        torch.from_numpy(native["depth"]), torch.from_numpy(native["intrinsics"]),
        torch.from_numpy(native["c2w"]), case["samples"], matches, confidence)
    final = dict(depth=final_depth.numpy(), intrinsics=native["intrinsics"].copy(),
                 c2w=metric_cameras.numpy(), scales=scales.numpy(),
                 g0=np.asarray(diagnostics["g0"], dtype=np.float64))
    write_arrays(raw / "final.npz", final)
    dump_new(public / "final.json", diagnostics)
    evidence = dict(capture=metadata, final=diagnostics, sample_counts=counts,
                    original_prior_rows_preserved=True, final_compute="CPU float64",
                    native_file=(raw / "native.npz").relative_to(ROOT).as_posix(),
                    native_sha256=sha(raw / "native.npz"),
                    final_file=(raw / "final.npz").relative_to(ROOT).as_posix(),
                    final_sha256=sha(raw / "final.npz"),
                    graph_file=(public / "native_graph.json").relative_to(ROOT).as_posix(),
                    graph_sha256=sha(public / "native_graph.json"))
    return native, final, matches, confidence, evidence


def differences(first, second):
    if set(first) != set(second):
        raise ValueError("Compared endpoint field inventories differ")
    return {name: dict(exact=bool(np.array_equal(first[name], second[name])),
                       max_absolute_difference=float(np.max(np.abs(first[name].astype(np.float64) - second[name].astype(np.float64)))))
            for name in first}


def run(weights, attempt, manifest, image):
    from profiled_runtime import configure_runtime, verify_runtime
    arithmetic = configure_runtime()
    from mast3r.model import AsymmetricMASt3R
    from mast3r_network_priors import digest, install
    from profiled_network import input_hashes, load_case, sha, source_hashes
    from register_profiled_execution import verify_registry

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", attempt):
        raise ValueError("Attempt must be a short alphanumeric/hyphen/underscore identifier")
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    verify_registry(manifest, "engineering")
    registration = json.loads(manifest.read_text(encoding="utf-8"))
    environment = json.loads((PUBLIC / "environment.json").read_text(encoding="utf-8"))
    checkpoint = sha(weights / "model.safetensors")
    if image != registration["docker_image"] or image != environment["docker_image"]:
        raise ValueError("Host-inspected Docker image differs from registered engineering runtime")
    if checkpoint != registration["checkpoint_sha256"] or checkpoint != environment["checkpoint_sha256"]:
        raise ValueError("Model checkpoint differs from registered engineering runtime")
    if torch.__version__ != environment["torch"] or torch.version.cuda != environment["cuda"]:
        raise ValueError("Current PyTorch/CUDA runtime differs from registered environment")
    public = PUBLIC / "endpoint_attempts" / attempt
    raw = RAW / "endpoint_attempts" / attempt
    if public.exists() or raw.exists() or (PUBLIC / "endpoint_bridge.json").exists():
        raise FileExistsError("Preserve existing endpoint attempts and published bridge evidence")
    public.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    sources = source_hashes(extra=(
        "scripts/check_profiled_endpoints.py", "scripts/profiled_final_calibration.py",
        "scripts/mast3r_calibrated_alignment.py", "scripts/mast3r_alignment_hook.py",
        "scripts/mast3r_joint_prior_loss.py", "scripts/prepare_profiled_inputs.py",
        "third_party/mast3r/mast3r/cloud_opt/sparse_ga.py",
        "third_party/mast3r/dust3r/dust3r/image_pairs.py",
    ))
    execution_commit = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    runtime = dict(execution_commit=execution_commit, docker_image=image, runtime_policy=arithmetic,
                   checkpoint_sha256=checkpoint, torch=torch.__version__, cuda=torch.version.cuda,
                   source_sha256_lf=sources, input_sha256=input_hashes(),
                   engineering_manifest=manifest.relative_to(ROOT).as_posix(),
                   engineering_manifest_sha256=sha(manifest))
    start = time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        np.random.seed(42)
        case = load_case("heads", "earlier", device="cpu")
        model = AsymmetricMASt3R.from_pretrained(str(weights), local_files_only=True).eval().cuda().requires_grad_(False)
        original_state = list(model.state_dict().items())
        before = digest(original_state)
        pristine = capture_endpoint(model, case, raw / "pristine_cache", raw / "pristine", public / "pristine")
        after_pristine = digest(original_state)
        if before != after_pristine:
            raise ValueError("Original pretrained state changed during pristine native capture")
        torch.manual_seed(42)
        modules = install(model, ROOT / "third_party/TCO/peft/lora.py")
        if len(modules) != 144:
            raise ValueError("Unexpected number of zero-initialized LoRA modules")
        adapter_before = digest([(name, value) for name, value in model.named_parameters() if value.requires_grad])
        zero = capture_endpoint(model, case, raw / "zero_cache", raw / "zero", public / "zero")
        adapter_after = digest([(name, value) for name, value in model.named_parameters() if value.requires_grad])
        after = digest(original_state)
        native_difference = differences(pristine[0], zero[0])
        final_difference = differences(pristine[1], zero[1])
        graph_equal = all(
            left["i"] == right["i"] and left["j"] == right["j"] and
            torch.equal(left["xy_i"], right["xy_i"]) and torch.equal(left["xy_j"], right["xy_j"])
            for left, right in zip(pristine[2], zero[2])) and len(pristine[2]) == len(zero[2])
        confidence_equal = torch.equal(pristine[3], zero[3])
        native_pass = all(item["exact"] for item in native_difference.values()) and graph_equal and confidence_equal
        final_pass = all(item["exact"] for item in final_difference.values())
        original_unchanged = before == after_pristine == after
        adapters_unchanged = adapter_before == adapter_after
        # Guard against concurrent edits while a lengthy local bridge was running.
        for name, expected in sources.items():
            if sha(ROOT / name, text=True) != expected:
                raise ValueError(f"Source changed during endpoint bridge: {name}")
        if input_hashes() != runtime["input_sha256"]:
            raise ValueError("Registered inputs changed during endpoint bridge")
        verify_runtime()
        report = dict(
            schema="profiled_endpoint_bridge_v1", attempt=attempt, scene="heads", adaptation_source="earlier",
            scope="Pristine versus zero-LoRA endpoint equivalence; no held-source or target-depth accuracy read",
            **runtime, device=torch.cuda.get_device_name(), allow_tf32=True, seed=42,
            inference_precision="float32", final_precision_device="float64 CPU",
            coarse_iterations=300, fine_iterations=300, pose_weight=.01, final_graph_weight=.1,
            adapter_module_count=len(modules), original_state_sha256=before, final_original_state_sha256=after,
            original_state_unchanged=original_unchanged, adapter_state_unchanged=adapters_unchanged,
            native_endpoint_pass=native_pass, final_endpoint_pass=final_pass,
            fresh_matching_exact=graph_equal, fresh_fused_confidence_exact=confidence_equal,
            native_difference=native_difference, final_difference=final_difference,
            captures=dict(pristine=pristine[4], zero_lora=zero[4]),
            seconds=time.perf_counter() - start, rental_compute_usd=0,
            pass_all=native_pass and final_pass and original_unchanged and adapters_unchanged,
        )
        dump_new(public / "comparison.json", report)
        if not report["pass_all"]:
            raise ValueError("Fresh native or final zero-LoRA endpoint equivalence failed; inspect preserved comparison")
        dump_new(PUBLIC / "endpoint_bridge.json", report)
        print(json.dumps({name: report[name] for name in ("pass_all", "native_endpoint_pass", "final_endpoint_pass", "seconds")}), flush=True)
    except Exception as error:
        dump_new(public / "failure.json", dict(
            **runtime, attempt=attempt, error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), seconds=time.perf_counter() - start, pass_all=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("/weights"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/lora_profiled_engineering_v1.json"))
    parser.add_argument("--image", required=True, help="Docker image ID inspected by the launching host")
    arguments = parser.parse_args()
    run(arguments.weights, arguments.attempt, arguments.manifest, arguments.image)
