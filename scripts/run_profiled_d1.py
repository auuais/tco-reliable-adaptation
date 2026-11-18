"""Register, run and replay the single bounded D1 MASt3R probe."""
import argparse
import ast
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import traceback
import unittest

from run_profiled_d0 import (ROOT, IMAGE, committed_identity, now, read,
                            runtime_identity as cpu_runtime, sha, write_new)
from profiled_d1_analysis import STATES, PAIRS, analyze_matrices, depth_statistics, key

DESIGN = "manifests/lora_profiled_d1_design_v1.json"
REGISTRY = "manifests/lora_profiled_d1_execution_v1.json"
INVENTORY = "results/lora_profiled_d1_v1/inputs_preparation01.json"
D0_REPLAY = "results/lora_profiled_d0_v1/verification02.json"
OUTPUT = "results/lora_profiled_d1_v1/attempts"
PRIVATE = "reports/lora_profiled_d1_v1/attempts"
SCRIPTS = ("run_profiled_d1", "run_profiled_d1_local", "profiled_d1_capture",
           "profiled_d1_bridge", "profiled_d1_artifacts", "profiled_d1_math",
           "mast3r_network_priors", "mast3r_network_capture", "profiled_network",
           "profiled_runtime", "run_mast3r_neural_transfer")
GPU_SECONDS, CPU_SECONDS = 3600, 1800
ALLOCATED_MIB, RESERVED_MIB = 8192, 9216
META_TOL = {"rtol": 1e-12, "atol": 1e-12}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(name):
    return dict(file=name, sha256=sha(ROOT / name))


def git(root, *args):
    command = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    return subprocess.check_output(command + list(args), text=True)


def source_closure():
    """Bind transitive local imports and all tracked upstream Python sources."""
    directory = ROOT / "scripts"
    pending = set(SCRIPTS)
    pending.update(path.stem for path in directory.glob("test_profiled_d1_*.py"))
    sources = {}
    while pending:
        name = pending.pop()
        path = directory / (name + ".py")
        rel = path.relative_to(ROOT).as_posix()
        if rel in sources:
            continue
        sources[rel] = sha(path, text=True)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                imports = [node.module.split(".")[0]]
            else:
                continue
            pending.update(n for n in imports if (directory / (n + ".py")).is_file())
    upstream = {}
    environment = read(ROOT / "results/lora_profiled_v1/environment.json")
    for rel, revision in environment["upstream_revisions"].items():
        repo = ROOT / rel
        require(git(repo, "rev-parse", "HEAD").strip() == revision,
                "Upstream source revision differs: " + rel)
        for filename in git(repo, "ls-files", "-z", "--", "*.py").split("\0"):
            if filename:
                upstream[f"{rel}/{filename}"] = sha(repo / filename, text=True)
    lora = "third_party/TCO/peft/lora.py"
    upstream[lora] = sha(ROOT / lora, text=True)
    return dict(sorted(sources.items())), dict(sorted(upstream.items()))


def d0_gate(inventory):
    replay = read(ROOT / D0_REPLAY)
    require(replay["schema"] == "profiled_d0_replay_v1" and replay["pass_all"] is True,
            "D0 exact replay did not pass")
    for name, field in (("d0_execution", "registration"), ("d0_report", "original_report")):
        require(replay[field] == {k: inventory["trigger"][name][k] for k in ("file", "sha256")},
                "D0 replay does not bind the selected evidence")
    return descriptor(D0_REPLAY)


def registration_payload(weights, *, full_parent):
    from profiled_d0_artifacts import verify_inventory as verify_parent
    from profiled_d1_artifacts import verify_inventory
    from profiled_d1_math import tolerance_spec
    inventory = read(ROOT / INVENTORY)
    replay = d0_gate(inventory)
    sources, upstream = source_closure()
    inherited = inventory["parent_source_sha256_lf"]
    require(all(sha(ROOT / name, text=True) == digest for name, digest in inherited.items()),
            "A bound M2/D0 source changed")
    verify_inventory(ROOT, weights, inventory)
    if full_parent:
        parent = read(ROOT / inventory["trigger"]["d0_execution"]["file"])
        verify_parent(ROOT, weights, parent["inventory"])
    return dict(schema="profiled_d1_execution_v1", design=descriptor(DESIGN),
                input_inventory=descriptor(INVENTORY), d0_replay=replay,
                source_sha256_lf=sources, upstream_sha256_lf=upstream,
                runtime_evidence=descriptor("results/lora_profiled_v1/environment.json"),
                image=IMAGE, teacher_raster_policy="exact_float32_equality",
                teacher_metadata_tolerance=META_TOL, math_tolerances=tolerance_spec(),
                limits=dict(model_seconds=GPU_SECONDS, cpu_seconds=CPU_SECONDS,
                            host_memory_bytes=8*1024**3, cuda_allocated_mib=ALLOCATED_MIB,
                            cuda_reserved_mib=RESERVED_MIB, concurrent_jobs=1),
                call_inventory=dict(encoder=5, pristine_decoder=20, final_decoder=20,
                                    new_fp32_decoder=0, optimizer=0, backward=0),
                output_roots=[OUTPUT, PRIVATE], rental_usd=0,
                commands={mode: f"python scripts/run_profiled_d1_local.py {mode} --attempt d1_01"
                          for mode in ("model", "analyze", "replay")},
                committed_registration_before_array_decode=True)


def prepare(weights):
    runtime = cpu_runtime()
    require(not (ROOT / REGISTRY).exists(), "Use a new registration version for any amendment")
    payload = registration_payload(weights, full_parent=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "scripts"), "test_profiled_d1_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    require(result.wasSuccessful() and not result.skipped and result.testsRun >= 84,
            "D1 deployment fixtures must pass without skips")
    sources, upstream = source_closure()
    require(sources == payload["source_sha256_lf"] and upstream == payload["upstream_sha256_lf"],
            "Source bytes changed during deployment checks")
    payload["synthetic_acceptance"] = dict(runtime=runtime, tests_run=result.testsRun,
                                           pass_all=True, arrays_deserialized=False)
    write_new(ROOT / REGISTRY, payload)
    print("D1 executable registration prepared. Commit before production array access.", flush=True)


def checked_registration(weights, *, full_parent):
    registered = read(ROOT / REGISTRY)
    sources, upstream = source_closure()
    require(sources == registered["source_sha256_lf"] and upstream == registered["upstream_sha256_lf"],
            "D1 executable source differs from registration")
    head = committed_identity(ROOT, REGISTRY, sources)
    payload = registration_payload(weights, full_parent=full_parent)
    proof = registered["synthetic_acceptance"]
    from run_profiled_d0 import EXPECTED_RUNTIME
    require(proof["pass_all"] is True and proof["runtime"] == EXPECTED_RUNTIME
            and proof["tests_run"] >= 84 and proof["arrays_deserialized"] is False,
            "Wrong synthetic deployment evidence")
    require(payload == {k: v for k, v in registered.items() if k != "synthetic_acceptance"},
            "D1 execution differs from registration")
    return registered, head


def check_gpu():
    import torch
    import numpy as np
    from profiled_runtime import configure_runtime
    env = read(ROOT / "results/lora_profiled_v1/environment.json")
    require(platform.system() == "Linux" and os.environ.get("PROFILED_D0_IMAGE_ID") == IMAGE
            and platform.python_version() == env["python"] and np.__version__ == "2.1.2"
            and torch.__version__ == env["torch"] and torch.version.cuda == env["cuda"],
            "Local device runtime differs from registration")
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == env["gpu"]
            and torch.cuda.get_device_properties(0).total_memory == env["total_vram_bytes"],
            "Only the registered local RTX 3080 is eligible")
    policy = configure_runtime()
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                torch_threads=torch.get_num_threads(), arithmetic=policy)


def save_arrays(path, values):
    import numpy as np
    with path.open("xb") as stream:
        np.savez(stream, **values)


def load_sealed_arrays(path, digest):
    import numpy as np
    require(sha(path) == digest, "Sealed evidence was modified")
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate sealed array members")
        result = {name: archive[name] for name in archive.files}
    require(sha(path) == digest, "Sealed evidence changed during decode")
    return result


def exact_raster(expected, actual):
    import numpy as np
    require(expected.dtype == actual.dtype == np.float32 and expected.shape == actual.shape,
            "Teacher raster shape or dtype differs")
    delta = np.abs(expected.astype(np.float64) - actual)
    return dict(equal=bool(np.array_equal(expected, actual)),
                differing_pixels=int(np.count_nonzero(delta)), max_abs_difference=float(delta.max()))


def record_failure(public, name, error):
    write_new(public / f"{name}_failure.json", dict(pass_all=False, finished_at_utc=now(),
              error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))


def resource_guard(start):
    import torch
    from profiled_runtime import verify_runtime
    verify_runtime()
    require(time.monotonic()-start < GPU_SECONDS, "D1 model wall-time cap exceeded")
    require(torch.cuda.max_memory_allocated()/2**20 <= ALLOCATED_MIB, "D1 allocated VRAM cap exceeded")
    require(torch.cuda.max_memory_reserved()/2**20 <= RESERVED_MIB, "D1 reserved VRAM cap exceeded")


def model_stage(weights, attempt, public, private):
    from run_profiled_lora import scientific_gpu_lock
    with scientific_gpu_lock(dict(run_id="heads_earlier_D1", attempt=attempt)):
        public.mkdir(parents=True, exist_ok=False)
        try:
            private.mkdir(parents=True, exist_ok=False)
            write_new(public / "model_started.json", dict(attempt=attempt, started_at_utc=now(),
                      execution=descriptor(REGISTRY), rental_usd=0))
            run_model(weights, public, private)
        except BaseException as error:
            record_failure(public, "model", error)
            raise


def run_model(weights, public, private):
    start = time.monotonic()
    registered, head = checked_registration(weights, full_parent=True)
    runtime = check_gpu()
    import numpy as np
    import torch
    import profiled_d1_artifacts as artifacts
    from profiled_d1_bridge import prepare_sensor_folds, verify_teacher_metadata, assemble_predictive_matrices
    from profiled_d1_capture import make_views, capture_training_paths, fuse_common
    from run_mast3r_neural_transfer import intrinsic

    inventory = read(ROOT / INVENTORY)
    counter = 0

    def progress(stage, details):
        nonlocal counter
        counter += 1
        write_new(public / "events" / f"{counter:03d}.json",
                  dict(stage=stage, entered_at_utc=now(), details=details))

    progress("release_verified", {"commit": head, "runtime": runtime})
    rgb = artifacts.load_rgb(ROOT, inventory)
    views = make_views(rgb, inventory["rgb_archive_sha256"])
    permitted = artifacts.load_adaptation(ROOT, inventory)
    poses = artifacts.load_poses(ROOT, inventory)
    case = dict(K=torch.as_tensor(intrinsic(), device="cuda", dtype=torch.float64),
                c2w=torch.as_tensor(np.linalg.inv(poses), device="cuda", dtype=torch.float64),
                samples=[dict(xy=r["xy"], depth=r["depth"]) for r in permitted],
                folds=[r["fold"] for r in permitted])
    teacher = artifacts.load_teacher(ROOT, inventory)
    metadata_descriptor = inventory["teacher_metadata"]
    teacher_metadata = read(ROOT / metadata_descriptor["file"])
    require(descriptor(metadata_descriptor["file"])["sha256"] == metadata_descriptor["sha256"],
            "Historical teacher metadata changed")
    prepared = None

    def teacher_gate(depth, confidence):
        nonlocal prepared
        comparisons = {name: exact_raster(teacher[name], value.cpu().numpy())
                       for name, value in (("depth", depth), ("confidence", confidence))}
        receipt = dict(rasters=comparisons, metadata=None, pass_all=False,
                       historical_per_partner_confidence_available=False)
        try:
            require(all(row["equal"] for row in comparisons.values()), "Exact fused teacher mismatch")
            prepared = prepare_sensor_folds(depth, confidence, case)
            receipt["current_metadata"] = prepared["metadata"]
            receipt["metadata"] = verify_teacher_metadata(prepared["metadata"], teacher_metadata["states"],
                                                          **registered["teacher_metadata_tolerance"])
            receipt["pass_all"] = True
        except Exception as error:
            receipt["error_type"], receipt["error"] = type(error).__name__, str(error)
        write_new(public / "teacher_gate.json", receipt)
        return receipt

    def load_adapter():
        progress("final_adapter_access", {})
        return artifacts.load_adapter(ROOT, inventory)

    resource_guard(start)
    capture = capture_training_paths(weights, ROOT / "third_party/TCO/peft/lora.py",
                                     views, load_adapter, teacher_gate, progress)
    progress("fp32_cache_access", {})
    depths = {f"xy_{i}": row["xy"] for i, row in enumerate(permitted)}
    fused = {s: capture[f"depth{k}"] for k, s in enumerate(("B0", "B1"))}
    for k, state in enumerate(("B0", "B1")):
        points = capture.pop(f"points{k}")
        for i, j in PAIRS:
            depths[f"{state}_{i}_{j}"] = points[i, j][0][..., 2].contiguous().cpu().numpy()
        del points
    confidence = capture["confidence"]
    for state, run in zip(("F0", "F1"), artifacts.RUNS):
        points = {}
        for i, j in PAIRS:
            resource_guard(start)
            xyz = artifacts.load_forward(ROOT, inventory, run, i, j)
            depths[f"{state}_{i}_{j}"] = xyz[..., 2].copy()
            points[i, j] = (torch.from_numpy(xyz).cuda(), None)
        fused[state], _ = fuse_common(points, confidence)
        del points

    matrices = {}
    for state in STATES:
        resource_guard(start)
        depths[f"{state}_fused"] = fused[state].cpu().numpy()
        for fold in (0, 1):
            values = assemble_predictive_matrices(fused[state], prepared["states"][fold],
                                                  prepared["queries"][fold])
            for field, tensor in values.items():
                matrices[key(fold, state, field)] = tensor.cpu().numpy()
    matrix_file, depth_file = private / "matrices.npz", private / "depths.npz"
    save_arrays(matrix_file, matrices)
    save_arrays(depth_file, depths)
    artifacts.verify_inventory(ROOT, weights, inventory)
    resource_guard(start)
    model_seal = dict(schema="profiled_d1_model_seal_v1", pass_all=True, execution=descriptor(REGISTRY),
                      execution_commit=head, runtime=runtime, capture=capture["metadata"],
                      teacher=descriptor((public / "teacher_gate.json").relative_to(ROOT).as_posix()),
                      arrays=[descriptor(p.relative_to(ROOT).as_posix()) for p in (matrix_file, depth_file)],
                      elapsed_seconds=time.monotonic()-start,
                      cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20, rental_usd=0)
    write_new(public / "model_seal.json", model_seal)
    print(json.dumps(model_seal), flush=True)


def cpu_stage(public, private, mode):
    runtime = cpu_runtime()
    start = time.monotonic()
    committed_identity(ROOT, REGISTRY, read(ROOT / REGISTRY)["source_sha256_lf"])
    seal_path = public / "model_seal.json"
    seal = read(seal_path)
    expected_paths = [private / name for name in ("matrices.npz", "depths.npz")]
    require(seal["pass_all"] is True and seal["execution"] == descriptor(REGISTRY)
            and [row["file"] for row in seal["arrays"]] ==
            [path.relative_to(ROOT).as_posix() for path in expected_paths],
            "Model seal or output membership differs")
    require(descriptor(seal["teacher"]["file"]) == seal["teacher"]
            and read(ROOT / seal["teacher"]["file"])["pass_all"] is True, "Teacher evidence changed")
    report_path = public / "report.json"
    request = public / f"{mode}_started.json"
    prior_time = 0.0
    if mode == "replay":
        prior_time = read(report_path)["seconds"]
    require(0 <= prior_time < CPU_SECONDS, "Combined CPU analysis allowance exhausted")
    signal.alarm(max(1, int(CPU_SECONDS-prior_time)))
    write_new(request, dict(started_at_utc=now(), model_seal_sha256=sha(seal_path), runtime=runtime))
    try:
        matrix, depths = (load_sealed_arrays(path, row["sha256"])
                          for path, row in zip(expected_paths, seal["arrays"]))
        response = analyze_matrices(matrix)
        functional = depth_statistics(depths)
        geometry_valid = all(row[role]["geometrically_valid"] for block in functional
                             for row in block["records"] for role in ("grid", "adaptation"))
        scientific = dict(responses=response, depth_statistics=functional,
                          all_selected_geometry_valid=geometry_valid,
                          pass_all=geometry_valid and all(r["response"]["pass_all"] for r in response),
                          accuracy_improvement_established=False, rental_usd=0)
        if mode == "analyze":
            write_new(report_path, dict(schema="profiled_d1_report_v1", scientific=scientific,
                      model_seal_sha256=sha(seal_path), runtime=runtime, seconds=time.monotonic()-start))
        else:
            original = read(report_path)
            exact = (original["scientific"] == scientific and original["runtime"] == runtime
                     and original["model_seal_sha256"] == sha(seal_path))
            write_new(public / "replay.json", dict(schema="profiled_d1_cpu_replay_v1",
                      exact_scientific_payload=exact, pass_all=exact and scientific["pass_all"],
                      report_sha256=sha(report_path), model_seal_sha256=sha(seal_path),
                      seconds=time.monotonic()-start, cumulative_cpu_wall_seconds=prior_time+time.monotonic()-start,
                      model_forward_calls=0, runtime=runtime, rental_usd=0))
            require(exact, "Independent process replay differs")
        require(scientific["pass_all"], "D1 arithmetic or selected-geometry acceptance failed")
        print(f"D1 {mode} passed", flush=True)
    except BaseException as error:
        record_failure(public, mode, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "model", "analyze", "replay"))
    parser.add_argument("--weights", type=Path, default=Path("/weights"))
    parser.add_argument("--attempt")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.weights)
        return
    require(args.attempt is not None and re.fullmatch(r"d1_[0-9]{2}", args.attempt), "Invalid D1 attempt ID")
    public, private = ROOT / OUTPUT / args.attempt, ROOT / PRIVATE / args.attempt

    def timeout(_signal, _frame):
        raise TimeoutError("D1 stage exceeded the registered wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(GPU_SECONDS if args.mode == "model" else CPU_SECONDS)
    if args.mode == "model":
        model_stage(args.weights, args.attempt, public, private)
    else:
        cpu_stage(public, private, args.mode)


if __name__ == "__main__":
    main()
