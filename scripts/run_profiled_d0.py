"""Prospectively registered CPU-only diagnosis of sealed LoRA/forward artifacts.

No model is constructed and no optimizer or evaluation depth is available here.
Preparation/registration verification only hash files; real tensors are decoded
solely by run/verify after the exact executable registration is in Git HEAD.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50"
DESIGN = "manifests/lora_profiled_diagnosis_design_v1.json"
REGISTRY = "manifests/lora_profiled_diagnosis_execution_v1.json"
OUTPUT = "results/lora_profiled_d0_v1"
SOURCES = (
    "scripts/profiled_d0_math.py", "scripts/test_profiled_d0_math.py",
    "scripts/profiled_d0_artifacts.py", "scripts/test_profiled_d0_artifacts.py",
    "scripts/run_profiled_d0.py", "scripts/test_run_profiled_d0.py",
    "scripts/run_profiled_d0_local.py", "scripts/test_run_profiled_d0_local.py",
    "scripts/seal_profiled_predictions.py",
)
EXPECTED_RUNTIME = dict(python="3.11.13", numpy="2.1.2", torch="2.7.1+cu128",
                        safetensors="0.5.3", system="Linux", image=IMAGE,
                        torch_threads=1, torch_interop_threads=1, cuda_available=False)
INDEPENDENT_RTOL = 1e-11
INDEPENDENT_ATOL = 1e-13


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def path_at(root, relative):
    from profiled_artifacts import resolve_logical
    return resolve_logical(root, relative)


def sha(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def source_hashes(root):
    return {name: sha(path_at(root, name), text=True) for name in SOURCES}


def runtime_identity():
    import numpy as np
    import torch
    import safetensors
    require(platform.system() == "Linux", "Production D0 uses the pinned Linux CPU image")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "D0 must explicitly hide CUDA")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", "D0 requires one CPU arithmetic thread: " + name)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    result = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                  safetensors=safetensors.__version__, system=platform.system(),
                  image=os.environ.get("PROFILED_D0_IMAGE_ID"),
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  cuda_available=torch.cuda.is_available())
    require(result == EXPECTED_RUNTIME, "D0 runtime differs from registered CPU environment")
    return result


def committed_identity(root, registry_name, sources):
    """Require the registration and new executable sources to exist unchanged in HEAD."""
    root = Path(root).resolve()
    command = ["git", "-c", "safe.directory=" + str(root), "-C", str(root)]
    head = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    for name in (registry_name,) + tuple(sources):
        actual = path_at(root, name).read_bytes()
        recorded = subprocess.check_output(command + ["show", "HEAD:" + name])
        if name in sources:
            actual, recorded = actual.replace(b"\r\n", b"\n"), recorded.replace(b"\r\n", b"\n")
        require(actual == recorded, "D0 registration/source must be committed before decode: " + name)
    return head


def prepare(root, weights, registry_name, checks_name):
    from profiled_d0_artifacts import build_inventory
    root = Path(root).resolve()
    destination = path_at(root, registry_name)
    require(not destination.exists(), "Preserve existing execution registration")
    runtime = runtime_identity()
    sources = source_hashes(root)
    checks = read(path_at(root, checks_name))
    require(checks.get("pass_all") is True and checks.get("tests_run", 0) > 0,
            "Passed synthetic preparation required")
    require(checks.get("source_sha256_lf") == sources, "Synthetic tests cover different source bytes")
    require(checks.get("runtime") == runtime, "Synthetic tests ran in a different runtime")
    inventory = build_inventory(root, DESIGN, weights)
    registration = dict(schema="profiled_d0_execution_v1", prepared_at_utc=now(),
        design=dict(file=DESIGN, sha256=sha(path_at(root, DESIGN))),
        source_sha256_lf=sources, runtime=runtime, inventory=inventory,
        synthetic_checks=dict(file=checks_name, sha256=sha(path_at(root, checks_name))),
        resources=dict(max_wall_seconds=1800, max_concurrent_jobs=1, gpu=False,
                       model_forward_calls=0, optimizer_updates=0, rental_usd=0),
        outputs=dict(root=OUTPUT, fresh_attempt_required=True),
        independent_check=dict(rtol=INDEPENDENT_RTOL, atol=INDEPENDENT_ATOL,
            weight="First registered adapter, lexicographically first module: direct BA Frobenius versus Gram",
            function="Heads earlier P1 versus P0, first directed pair, fixed grid: least-squares scalar versus dot projection"),
        units="Raw MASt3R self-camera axial model gauge; not meters and not an accuracy metric",
        interpretation="No diagnosis can release M3/rental or replace the failed screen; secants are not Jacobians",
        commands=dict(
            verify_registration=f"python scripts/run_profiled_d0.py verify-registration --registry {registry_name} --weights /weights",
            run=f"python scripts/run_profiled_d0.py run --registry {registry_name} --weights /weights --attempt FRESH_ID",
            verify=f"python scripts/run_profiled_d0.py verify --registry {registry_name} --weights /weights --attempt EXISTING_ID --output FRESH_RECEIPT"),
        committed_registration_required_before_decode=True,
        registration_preparation_decoded_tensors=False)
    write_new(destination, registration)
    return registration


def verify_registration(root, weights, registry_name, *, require_commit=False):
    from profiled_d0_artifacts import verify_inventory
    root = Path(root).resolve()
    registration = read(path_at(root, registry_name))
    require(registration.get("schema") == "profiled_d0_execution_v1", "Wrong D0 registration schema")
    require(registration.get("design") == dict(file=DESIGN, sha256=sha(path_at(root, DESIGN))), "Design changed")
    require(registration.get("source_sha256_lf") == source_hashes(root), "D0 sources changed")
    require(registration.get("runtime") == runtime_identity(), "D0 runtime changed")
    require(registration.get("resources") == dict(max_wall_seconds=1800, max_concurrent_jobs=1, gpu=False,
        model_forward_calls=0, optimizer_updates=0, rental_usd=0), "D0 resource contract changed")
    require(registration.get("outputs") == dict(root=OUTPUT, fresh_attempt_required=True), "Output contract changed")
    check = registration["synthetic_checks"]
    require(sha(path_at(root, check["file"])) == check["sha256"], "Synthetic acceptance changed")
    payload = read(path_at(root, check["file"]))
    require(payload.get("pass_all") is True and payload.get("source_sha256_lf") == registration["source_sha256_lf"]
            and payload.get("runtime") == registration["runtime"], "Invalid synthetic acceptance")
    require(registration.get("committed_registration_required_before_decode") is True
            and registration.get("registration_preparation_decoded_tensors") is False, "Wrong access policy")
    head = committed_identity(root, registry_name, SOURCES) if require_commit else None
    verify_inventory(root, weights, registration["inventory"])
    return registration, head


def _independent_weight_check(factors, measured):
    import numpy as np
    direct = float(np.linalg.norm((factors["B"] @ factors["A"]) / 4.0, ord="fro"))
    require(np.isclose(direct, measured, rtol=INDEPENDENT_RTOL, atol=INDEPENDENT_ATOL),
            "Independent materialized BA norm differs")
    return dict(direct_frobenius=direct, gram_frobenius=measured, pass_all=True)


def _independent_function_check(base, candidate, xy, measured):
    import numpy as np
    x, y = xy[:, 0], xy[:, 1]
    b, c = base[y, x], candidate[y, x]
    if not np.isfinite(b).all() or not np.isfinite(c).all() or float(b @ b) == 0:
        return dict(status="undefined_without_filtering", pass_all=None)
    coefficient = float(np.linalg.lstsq(b[:, None], c-b, rcond=None)[0][0])
    require(np.isclose(coefficient, measured, rtol=INDEPENDENT_RTOL, atol=INDEPENDENT_ATOL),
            "Independent least-squares scalar differs")
    return dict(lstsq_scalar=coefficient, dot_scalar=measured, pass_all=True)


def compute(root, weights, inventory, deadline, progress=None, emit=None):
    """Only registered loaders may touch arrays. No broad dataset/model helpers."""
    import numpy as np
    from profiled_d0_artifacts import load_adapter, load_forward, load_xy, load_base_weight_norms
    from profiled_d0_math import effective_weight_stats, aggregate_weight_stats, compare_depths, aggregate_equal_pairs, grid_coordinates
    def tick(stage):
        require(time.monotonic() < deadline, "D0 wall-time bound reached")
        if progress is not None:
            progress(stage)
    weights_out = []
    base_norms = None
    expected_modules = None
    independent = {}
    for index, descriptor in enumerate(inventory["adapters"]):
        tick("adapter:" + descriptor["run_id"])
        factors = load_adapter(root, descriptor)
        modules = sorted(factors)
        if base_norms is None:
            expected_modules = modules
            shapes = {name: (factors[name]["B"].shape[0], factors[name]["A"].shape[1]) for name in modules}
            base_norms = load_base_weight_norms(weights, inventory["checkpoint"], shapes)
        require(modules == expected_modules, "Adapter module membership changed")
        layer_stats = []
        for name in modules:
            record = effective_weight_stats(factors[name]["A"], factors[name]["B"], base_norms[name], rank=4, alpha=1)
            record["module"] = name
            record["base_key"] = factors[name]["base_key"]
            layer_stats.append(record)
        if index == 0:
            independent["weight"] = dict(run_id=descriptor["run_id"], module=modules[0],
                **_independent_weight_check(factors[modules[0]], layer_stats[0]["update_frobenius"]))
        record = dict(run_id=descriptor["run_id"], layers=layer_stats,
                      aggregate=aggregate_weight_stats(layer_stats))
        weights_out.append(record)
        if emit is not None:
            emit(f"adapter_{index:02d}.json", record)
        del factors
    tick("functional_coordinates")
    allowed = load_xy(root, inventory["coordinates"])
    grid = grid_coordinates()
    functional = {entry["run_id"]: entry for entry in inventory["functional"]}
    base_id = "heads_earlier_P0"
    cases = ["heads_earlier_P1", "heads_earlier_P2", "heads_earlier_P3", "heads_later_P0"]
    pairs = {key: {(p["i"], p["j"]): p for p in value["pairs"]} for key, value in functional.items()}
    expected_pairs = sorted(pairs[base_id])
    require(len(expected_pairs) == 20 and all(sorted(pairs[key]) == expected_pairs for key in cases),
            "Functional pair inventory differs")
    rows = {key: [] for key in cases}
    for pair_number, pair in enumerate(expected_pairs):
        tick("functional_pair:" + str(pair))
        base = load_forward(root, pairs[base_id][pair])
        for key in cases:
            candidate = load_forward(root, pairs[key][pair])
            result = dict(i=pair[0], j=pair[1], fixed_grid=compare_depths(base, candidate, grid),
                          adaptation_coordinates=compare_depths(base, candidate, allowed[pair[0]]))
            rows[key].append(result)
            if pair_number == 0 and key == "heads_earlier_P1":
                independent["function"] = dict(run_id=key, pair=list(pair),
                    **_independent_function_check(base, candidate, grid, result["fixed_grid"]["scalar_coefficient"]))
            del candidate
        if emit is not None:
            emit(f"pair_{pair_number:02d}.json", dict(pair=list(pair),
                comparisons={key: rows[key][-1] for key in cases}))
        del base
    output = dict(adapter_statistics=weights_out, functional_statistics={}, independent_checks=independent,
                  units="Uncalibrated self-camera axial model gauge, not metric depth",
                  numerical_tolerances=dict(independent_rtol=INDEPENDENT_RTOL, independent_atol=INDEPENDENT_ATOL))
    for key in cases:
        output["functional_statistics"][key] = dict(comparator=base_id, pairs=rows[key],
            equal_pair_fixed_grid=aggregate_equal_pairs([r["fixed_grid"] for r in rows[key]]),
            equal_pair_adaptation_coordinates=aggregate_equal_pairs([r["adaptation_coordinates"] for r in rows[key]]),
            meaning="same-state raw forward comparator only" if key == "heads_later_P0" else "final-minus-frozen raw output secant")
    tick("all_statistics_complete")
    return output


def attempt_path(root, attempt):
    require(isinstance(attempt, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", attempt), "Invalid fresh attempt ID")
    return path_at(root, OUTPUT + "/attempts/" + attempt)


def _alarm(signum, frame):
    raise TimeoutError("D0 1800-second wall bound reached")


def acquire_lock(root):
    """An OS lock, not a stale JSON flag, enforces one D0 process per workspace."""
    import fcntl
    path = path_at(root, "reports/lora_profiled_d0_v1/active.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(dict(pid=os.getpid(), container=os.environ.get("HOSTNAME"), started_at_utc=now())))
        stream.flush()
        return stream
    except BaseException:
        stream.close()
        raise


def execute(root, weights, registry_name, attempt, *, verification_output=None):
    import resource
    root = Path(root).resolve()
    destination = attempt_path(root, attempt)
    if verification_output is None:
        require(not destination.exists(), "Preserve existing D0 attempt")
    else:
        require(destination.is_dir(), "Verification requires an existing complete attempt")
        require(not path_at(root, verification_output).exists(), "Preserve existing verification receipt")
    started = time.monotonic()
    cpu_started = time.process_time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(1800)
    registration = None
    lock = None
    owned_attempt = False
    owned_verification = False
    try:
        lock = acquire_lock(root)
        registration, head = verify_registration(root, weights, registry_name, require_commit=True)
        identity = dict(file=registry_name, sha256=sha(path_at(root, registry_name)))
        if verification_output is None:
            destination.mkdir(parents=True, exist_ok=False)
            owned_attempt = True
            write_new(destination / "started.json", dict(started_at_utc=now(), registration=identity,
                execution_commit=head, runtime=registration["runtime"], pid=os.getpid(), container=os.environ.get("HOSTNAME")))
        else:
            write_new(path_at(root, verification_output + ".started.json"), dict(started_at_utc=now(),
                registration=identity, execution_commit=head, pid=os.getpid(), container=os.environ.get("HOSTNAME")))
            owned_verification = True
        def progress(stage):
            print(json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-started)), flush=True)
        def partial(name, data):
            write_new(destination / "partial" / name, dict(schema="profiled_d0_partial_v1",
                registration=identity, completion_status="partial_only", data=data))
        scientific = compute(root, weights, registration["inventory"], started+1800, progress,
                             partial if owned_attempt else None)
        verify_registration(root, weights, registry_name, require_commit=True)
        resources = dict(wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu_started,
            peak_host_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
            gpu_requested=False, model_forward_calls=0, optimizer_updates=0, rental_usd=0)
        if verification_output is None:
            report = dict(schema="profiled_d0_report_v1", status="complete", finished_at_utc=now(),
                registration=identity, execution_commit=head, runtime=registration["runtime"],
                resources=resources, scientific=scientific, diagnostic_only=True,
                calibration_absorption_established=False, accuracy_improvement_established=False,
                redesigns_consumed=0, rental_authorized=False)
            write_new(destination / "report.json", report)
        else:
            original = read(destination / "report.json")
            require(original.get("status") == "complete" and original.get("registration") == identity, "Original report identity differs")
            require(original.get("scientific") == scientific, "D0 numerical replay differs")
            report = dict(schema="profiled_d0_replay_v1", pass_all=True, verified_at_utc=now(),
                registration=identity, original_report=dict(file=(destination/"report.json").relative_to(root).as_posix(),
                    sha256=sha(destination/"report.json")), resources=resources,
                scope="Exact same-runtime diagnostic arithmetic replay; not new inference or physical accuracy verification")
            write_new(path_at(root, verification_output), report)
        return report
    except BaseException as exc:
        if owned_attempt and not (destination/"failure.json").exists():
            write_new(destination/"failure.json", dict(status="failed", at_utc=now(),
                type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                wall_seconds=time.monotonic()-started, rental_usd=0, model_forward_calls=0))
        if owned_verification and not path_at(root, verification_output).exists():
            write_new(path_at(root, verification_output), dict(status="failed", pass_all=False, at_utc=now(),
                type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                wall_seconds=time.monotonic()-started, rental_usd=0, model_forward_calls=0))
        raise
    finally:
        signal.alarm(0)
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify-registration", "run", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--checks")
    parser.add_argument("--attempt")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "prepare":
        require(args.checks is not None, "prepare requires --checks")
        prepare(args.root, args.weights, args.registry, args.checks)
        result = dict(registration_prepared=True, tensors_decoded=False)
    elif args.mode == "verify-registration":
        verify_registration(args.root, args.weights, args.registry)
        result = dict(registration_verified=True, tensors_decoded=False)
    else:
        require(args.attempt is not None, "run/verify requires --attempt")
        require(args.mode != "verify" or args.output is not None, "verify requires fresh --output")
        report = execute(args.root, args.weights, args.registry, args.attempt,
                         verification_output=args.output if args.mode == "verify" else None)
        result = dict(status=report.get("status", "verified"), rental_usd=0, model_forward_calls=0)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
