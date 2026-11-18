"""Serial, local-only Docker launcher for the sealed 36-state LoRA inventory.

Examples (PowerShell, repository root):
  python scripts/run_profiled_batch.py --manifest manifests/lora_profiled_execution_v2.json --attempt-prefix screen01 --dry-run
  python scripts/run_profiled_batch.py --manifest manifests/lora_profiled_execution_v2.json --attempt-prefix screen01 --execute --limit 2
  python scripts/run_profiled_batch.py --manifest manifests/lora_profiled_execution_v2.json --attempt-prefix screen01 --execute

The last command verifies and skips the first two completed states, then runs
the remaining states. A partial or failed attempt stops the queue. Investigate
it before explicitly choosing a new prefix; this script never overwrites or
chooses an attempt by time, score, or apparent quality. It never scores results,
starts a rental, deletes a container, or reads account/usage credentials.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

from profiled_artifacts import logical_inventory, resolve_logical


ROOT = Path(__file__).resolve().parents[1]
HOST_REPO = "C:/SVNProjects/Research/tco-research"
HOST_WEIGHTS = "E:/ResearchData/TCO/checkpoints/MASt3R/hf"
LAUNCHER = "scripts/run_profiled_batch.py"
RUNNER = "scripts/run_profiled_lora.py"
LABEL = "org.tco.profiled.batch"
GIB = 1024**3
# Bind every engine operation to this machine, even if another terminal changes
# the active Docker context while a long queue is running.
DOCKER = ["docker", "--host", "npipe:////./pipe/dockerDesktopLinuxEngine"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {name}")
    return path


def relative_file(parent, name):
    return resolve_logical(parent, name, must_exist=True)


def command(args, timeout=60):
    completed = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {args!r}\n{completed.stderr.strip()}")
    return completed.stdout.strip()


def check_identity(prefix):
    # Leave room for the deterministic container name and keep paths simple.
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", prefix):
        raise ValueError("Attempt prefix must be 1-40 lowercase letters/digits/hyphens/underscores")


def paths_for(root, job, prefix):
    return (inside(root, job["raw_attempt_parent"]) / prefix,
            inside(root, job["public_attempt_parent"]) / prefix)


def container_name(job, prefix):
    return f"tco-profiled-{prefix}-{job['run_id'].lower().replace('_', '-')}"


def docker_argv(job, prefix, manifest_relative, image):
    """No shell interpolation, remote Docker context, network, or mutable tag."""
    return [*DOCKER, "run", "--name", container_name(job, prefix), "--pull", "never",
            "--gpus", "all", "--network", "none", "--label", f"{LABEL}={prefix}",
            "--label", f"{LABEL}.run-id={job['run_id']}",
            "--env", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "--env", "HF_HUB_OFFLINE=1",
            "--mount", f"type=bind,source={HOST_REPO},target=/workspace",
            "--mount", f"type=bind,source={HOST_WEIGHTS},target=/weights,readonly",
            "--workdir", "/workspace", image, "python", RUNNER,
            "--run-id", job["run_id"], "--manifest", manifest_relative,
            "--attempt", prefix, "--weights", "/weights", "--image", image]


def inspect_container(name):
    value = command([*DOCKER, "container", "inspect", name, "--format", "{{json .}}"])
    return json.loads(value)


def inspect_optional(name):
    value = subprocess.run([*DOCKER, "container", "inspect", name, "--format", "{{json .}}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
    if value.returncode:
        if "No such object" in value.stderr or "No such container" in value.stderr:
            return None
        raise RuntimeError(f"Cannot inspect own container: {value.stderr.strip()}")
    return json.loads(value.stdout)


def live_gpu_containers():
    names = command([*DOCKER, "ps", "--format", "{{.Names}}"])
    conflicts = []
    for name in names.splitlines():
        item = inspect_container(name)
        # Stop neither other studies nor unrelated GPU work. Refuse to start.
        requests = item.get("HostConfig", {}).get("DeviceRequests") or []
        has_gpu = any("gpu" in capability for request in requests
                      for capability in request.get("Capabilities", []))
        if name.startswith("tco-profiled-") or has_gpu:
            conflicts.append(dict(name=name, id=item["Id"], gpu_requested=has_gpu))
    return conflicts


def host_checks(image, minimum_free_gib):
    if os.name != "nt" or ROOT.resolve() != Path(HOST_REPO).resolve():
        raise ValueError(f"This launcher is restricted to the local Windows workspace {HOST_REPO}")
    if not Path(HOST_WEIGHTS, "model.safetensors").is_file():
        raise FileNotFoundError("The pinned local E: checkpoint mount is unavailable")
    # Prevent DOCKER_HOST/contexts from silently turning local work into rental work.
    context = command(["docker", "context", "show"])
    context_record = json.loads(command(["docker", "context", "inspect", context]))[0]
    endpoint = context_record.get("Endpoints", {}).get("docker", {}).get("Host", "")
    override = os.environ.get("DOCKER_HOST")
    if not endpoint.startswith("npipe://") or (override and not override.startswith("npipe://")):
        raise ValueError("Only a Windows local named-pipe Docker engine is allowed")
    image_record = json.loads(command([*DOCKER, "image", "inspect", image]))[0]
    if image_record.get("Id") != image:
        raise ValueError("Inspected Docker image differs from the pinned immutable image ID")
    free = shutil.disk_usage(ROOT).free
    if free < minimum_free_gib*GIB:
        raise OSError(f"Only {free/GIB:.2f} GiB free in the repository volume; need {minimum_free_gib:g} GiB")
    weights_free = shutil.disk_usage(Path(HOST_WEIGHTS)).free
    if weights_free < GIB:
        raise OSError("Less than 1 GiB remains on the E: checkpoint volume")
    # Logs may contain full diagnostics and must stay in ignored local storage.
    command(["git", "-C", str(ROOT), "check-ignore", "reports/lora_profiled_v1/batches/.probe"])
    return dict(docker_context=context, docker_endpoint=endpoint, docker_image=image,
                repository_free_bytes=free, checkpoint_volume_free_bytes=weights_free,
                image_inspected=True, local_engine_only=True, pinned_engine=DOCKER[2])


def verify_completed(root, manifest, registry, job, prefix):
    """Return None for an unstarted attempt; reject every partial/failed attempt.

    A completion seal is usable only if every declared raw/public file is still
    present with its exact hash, there are no undeclared files, and its complete
    scientific identity matches this queue. This never reads accuracy values.
    """
    raw, public = paths_for(root, job, prefix)
    if not raw.exists() and not public.exists():
        return None
    path = public / "complete.json"
    if not raw.is_dir() or not public.is_dir() or not path.is_file() or (public / "failure.json").exists():
        raise ValueError(f"Partial/failed attempt {job['run_id']}/{prefix}; preserve it, investigate, then use an explicit new prefix")
    report = read(path)
    required = dict(run_id=job["run_id"], attempt=prefix, job=job, pass_all=True,
                    status="predictions_sealed_without_accuracy_scoring", original_state_unchanged=True,
                    accuracy_improvement_established=False, accuracy_scoring_performed=False,
                    optimizer_updates=job["optimizer_steps"], docker_image=registry["docker_image"],
                    checkpoint_sha256=registry["checkpoint_sha256"],
                    manifest=manifest.relative_to(root).as_posix(), manifest_sha256=sha(manifest),
                    source_sha256_lf=registry["source_sha256_lf"], artifact_sha256=registry["artifact_sha256"])
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"Completion identity differs: {job['run_id']}/{prefix}: {field}")
    for name, expected in registry["input_sha256"].items():
        if report.get("input_sha256", {}).get(name) != expected:
            raise ValueError(f"Completion input differs: {name}")
    original = report.get("original_state_sha256")
    if not original or report.get("pre_endpoint_original_state_sha256") != original or report.get("final_original_state_sha256") != original:
        raise ValueError("Completion does not establish an unchanged original model")
    if registry.get("runtime_policy") is not None and report.get("runtime_policy") != registry["runtime_policy"]:
        raise ValueError("Completion deterministic arithmetic policy differs")
    for base, key, extra in ((raw, "raw_files", set()), (public, "public_files", {"complete.json"})):
        inventory = report.get(key)
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError(f"Missing completion inventory {key}")
        observed = logical_inventory(base)
        if set(observed) != set(inventory) | extra:
            raise ValueError(f"Completion inventory differs for {base}")
        for name, expected in inventory.items():
            physical = resolve_logical(base, name, must_exist=True)
            if physical != observed[name] or sha(physical) != expected:
                raise ValueError(f"Completion file hash differs: {base / name}")
    return dict(run_id=job["run_id"], attempt=prefix, complete=path.relative_to(root).as_posix(),
                complete_sha256=sha(path), raw_files_verified=len(report["raw_files"]),
                public_files_verified=len(report["public_files"]))


def build_plan(manifest, registry, prefix):
    return dict(schema="profiled_local_queue_v1", attempt_prefix=prefix,
                manifest=manifest.relative_to(ROOT).as_posix(), manifest_sha256=sha(manifest),
                launcher=LAUNCHER, launcher_sha256_lf=sha(ROOT / LAUNCHER, text=True),
                docker_image=registry["docker_image"], checkpoint_sha256=registry["checkpoint_sha256"],
                host_repo=HOST_REPO, host_weights=HOST_WEIGHTS,
                source_sha256_lf=registry["source_sha256_lf"], artifact_sha256=registry["artifact_sha256"],
                input_sha256=registry["input_sha256"], rental_compute_usd=0,
                jobs=[dict(job=job, container=container_name(job, prefix),
                           argv=docker_argv(job, prefix, manifest.relative_to(ROOT).as_posix(), registry["docker_image"]))
                      for job in registry["jobs"]],
                selection="Manifest order only; --limit takes its prefix; skip only verified same-attempt completions",
                recovery="Preserve all partial/failed evidence; no automatic retry, overwrite, or container deletion")


def preflight(args):
    from register_profiled_execution import verify_registry
    check_identity(args.attempt_prefix)
    if args.limit is not None and not 1 <= args.limit <= 36:
        raise ValueError("--limit must be between 1 and 36")
    if not math.isfinite(args.max_wall_hours) or args.max_wall_hours <= 0:
        raise ValueError("--max-wall-hours must be finite and positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 1:
        raise ValueError("--min-free-gib must be finite and at least 1")
    manifest = inside(ROOT, args.manifest)
    verify_registry(manifest, "scientific")
    registry = read(manifest)
    if not registry.get("scientific_ready_at_registration") or len(registry["jobs"]) != 36:
        raise ValueError("Only a fully registered, bridge-validated 36-state scientific inventory may launch")
    if RUNNER not in registry["source_sha256_lf"]:
        raise ValueError("Scientific registry does not bind the one-case runner")
    plan = build_plan(manifest, registry, args.attempt_prefix)
    directory = ROOT / "results/lora_profiled_v1/batches" / args.attempt_prefix
    plan_path = directory / "plan.json"
    if directory.exists() and (not plan_path.is_file() or read(plan_path) != plan):
        raise ValueError("Existing queue plan differs or is incomplete; preserve it and use a new prefix")
    selection = registry["jobs"][:args.limit] if args.limit is not None else registry["jobs"]
    # Inspect the complete inventory for failure, even during a limited resume.
    completions = {job["run_id"]: verify_completed(ROOT, manifest, registry, job, args.attempt_prefix)
                   for job in registry["jobs"]}
    host = host_checks(registry["docker_image"], args.min_free_gib)
    host["checkpoint_sha256"] = sha(Path(HOST_WEIGHTS) / "model.safetensors")
    if host["checkpoint_sha256"] != registry["checkpoint_sha256"]:
        raise ValueError("The actual local checkpoint differs from the registered bytes")
    conflicts = live_gpu_containers()
    if conflicts:
        raise ValueError(f"Other GPU/research containers are running; no launch: {json.dumps(conflicts)}")
    for name in ("reports/lora_profiled_v1/scientific_gpu.lock", "reports/lora_profiled_v1/batch.lock"):
        if (ROOT / name).exists():
            raise ValueError(f"Existing local research lock {name}; inspect its owner; never delete a live lock")
    return manifest, registry, plan, selection, completions, host


@contextmanager
def batch_lock(prefix):
    path = ROOT / "reports/lora_profiled_v1/batch.lock"
    payload = dict(attempt_prefix=prefix, pid=os.getpid(), hostname=socket.gethostname(), created_at_utc=now())
    dump_new(path, payload)
    try:
        yield
    finally:
        if path.exists() and read(path) == payload:
            path.unlink()


def stop_owned(name, prefix, run_id):
    item = inspect_container(name)
    labels = item.get("Config", {}).get("Labels") or {}
    if labels.get(LABEL) != prefix or labels.get(f"{LABEL}.run-id") != run_id:
        raise ValueError("Refusing to stop a container without this queue's exact ownership labels")
    if item["State"].get("Running"):
        command([*DOCKER, "stop", "--time", "20", name], timeout=45)
    after = inspect_container(name)
    if after["State"].get("Running"):
        raise RuntimeError(f"Own container is still running after stop: {name}")
    return dict(name=name, running=False, container_exit_code=after["State"].get("ExitCode"),
                oom_killed=after["State"].get("OOMKilled"), stopped_at_utc=now())


def run_container(argv, log, name, prefix, run_id, deadline):
    """Attach to one fresh local container; timeout/interrupt stops only its label."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Batch wall-clock deadline reached before starting another case")
    # Existence is checked explicitly: no stale container is deleted or reused.
    names = command([*DOCKER, "ps", "-a", "--format", "{{.Names}}"])
    if name in names.splitlines():
        raise ValueError(f"Container name already exists: {name}; preserve it and investigate")
    exitcode = None
    stop = None
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop = stop_owned(name, prefix, run_id)
                    exitcode = process.wait(timeout=30)
                    return dict(status="timed_out", docker_exit_code=exitcode, stop=stop)
                try:
                    exitcode = process.wait(timeout=min(5., remaining))
                except subprocess.TimeoutExpired:
                    pass
            exitcode = process.returncode
        except BaseException as error:
            # A Ctrl+C must not leave attached local GPU work running unnoticed.
            try:
                if inspect_optional(name) is not None:
                    stop = stop_owned(name, prefix, run_id)
                if process.poll() is None:
                    exitcode = process.wait(timeout=30)
                else:
                    exitcode = process.returncode
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as stop_error:
                stop = dict(stop_failed=True, error=str(stop_error), container_state_unverified=True)
            return dict(status="launcher_interrupted", docker_exit_code=exitcode, stop=stop,
                        error_type=type(error).__name__, error=str(error))
    try:
        item = inspect_optional(name)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return dict(status="container_state_unverified", docker_exit_code=exitcode,
                    container_exit_code=None, error=str(error), stop=None)
    if item is None:
        return dict(status="container_not_created", docker_exit_code=exitcode,
                    container_exit_code=None, oom_killed=False, stop=None)
    state = item["State"]
    if state.get("Running"):
        stop = stop_owned(name, prefix, run_id)
        return dict(status="client_exited_before_container", docker_exit_code=exitcode,
                    container_exit_code=stop.get("container_exit_code"), stop=stop,
                    error="Docker client exited while its GPU container remained running; stopped it")
    return dict(status="exited", docker_exit_code=exitcode, container_exit_code=state.get("ExitCode"),
                oom_killed=state.get("OOMKilled"), stop=stop)


def execute(args, checked):
    from register_profiled_execution import verify_registry
    manifest, registry, plan, selection, completions, host = checked
    prefix = args.attempt_prefix
    queue_dir = ROOT / "results/lora_profiled_v1/batches" / prefix
    raw_dir = ROOT / "reports/lora_profiled_v1/batches" / prefix
    invocation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    public = queue_dir / "invocations" / invocation
    logs = raw_dir / "invocations" / invocation
    deadline = time.monotonic() + args.max_wall_hours*3600
    started = now()
    events = []
    terminal = dict(schema="profiled_local_queue_invocation_v1", invocation=invocation,
                    attempt_prefix=prefix, started_at_utc=started, host=host,
                    selected_run_ids=[job["run_id"] for job in selection], max_wall_hours=args.max_wall_hours,
                    limit=args.limit, rental_compute_usd=0, accuracy_scoring_performed=False)
    with batch_lock(prefix):
        plan_path = queue_dir / "plan.json"
        if not plan_path.exists():
            dump_new(plan_path, plan)
        elif read(plan_path) != plan:
            raise ValueError("Queue plan changed after preflight")
        logs.mkdir(parents=True, exist_ok=False)
        dump_new(public / "started.json", terminal)
        try:
            for job in selection:
                verify_registry(manifest, "scientific")
                if build_plan(manifest, registry, prefix) != plan:
                    raise ValueError("Launcher, scientific identity or queue plan changed during execution")
                completed = verify_completed(ROOT, manifest, registry, job, prefix)
                if completed is not None:
                    event = dict(status="verified_skip", verified_at_utc=now(), **completed)
                    events.append(event)
                    dump_new(public / f"{job['run_id']}.json", event)
                    print(f"Verified completed {job['run_id']}/{prefix}", flush=True)
                    continue
                host_checks(registry["docker_image"], args.min_free_gib)
                conflicts = live_gpu_containers()
                if conflicts:
                    raise ValueError(f"Another GPU/research job appeared: {json.dumps(conflicts)}")
                if (ROOT / "reports/lora_profiled_v1/scientific_gpu.lock").exists():
                    raise ValueError("One-case scientific GPU lock exists; inspect owner before recovery")
                entry = next(item for item in plan["jobs"] if item["job"]["run_id"] == job["run_id"])
                log = logs / f"{job['run_id']}.log"
                job_start = now()
                dump_new(public / f"{job['run_id']}.started.json",
                         dict(run_id=job["run_id"], started_at_utc=job_start, container=entry["container"],
                              argv=entry["argv"], log=log.relative_to(ROOT).as_posix()))
                print(f"Starting {job['run_id']}/{prefix}; log {log}", flush=True)
                result = run_container(entry["argv"], log, entry["container"], prefix, job["run_id"], deadline)
                event = dict(run_id=job["run_id"], started_at_utc=job_start, finished_at_utc=now(),
                             log=log.relative_to(ROOT).as_posix(), log_sha256=sha(log), **result)
                # Preserve the real process outcome even if subsequent checksum
                # verification fails before this case can become a valid seal.
                dump_new(public / f"{job['run_id']}.exit.json", event)
                if result["status"] != "exited" or result["docker_exit_code"] != 0 or result.get("container_exit_code") != 0 or result.get("oom_killed"):
                    dump_new(public / f"{job['run_id']}.json", event)
                    events.append(event)
                    raise RuntimeError(f"Case failed or timed out: {job['run_id']}; exit details preserved; queue stopped")
                completed = verify_completed(ROOT, manifest, registry, job, prefix)
                if completed is None:
                    raise ValueError("Container exited successfully without a sealed completed attempt")
                event.update(status="completed_and_verified", seal=completed)
                dump_new(public / f"{job['run_id']}.json", event)
                events.append(event)
                print(f"Completed and verified {job['run_id']}/{prefix}", flush=True)
            all_seals = {job["run_id"]: verify_completed(ROOT, manifest, registry, job, prefix)
                         for job in registry["jobs"]}
            terminal.update(status="success", selected_jobs_verified=len(selection),
                            all_36_predictions_sealed=all(value is not None for value in all_seals.values()),
                            completed_states=sum(value is not None for value in all_seals.values()), seals=all_seals)
        except BaseException as error:
            terminal.update(status="failed", error_type=type(error).__name__, error=str(error),
                            all_36_predictions_sealed=False,
                            recovery="Preserve this invocation, logs and all attempt files; investigate before explicitly selecting a new prefix")
            raise
        finally:
            terminal.update(finished_at_utc=now(), events=events)
            dump_new(public / "terminal.json", terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/lora_profiled_execution_v2.json"))
    parser.add_argument("--attempt-prefix", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="Validate all source/input/completion hashes and show argv; no writes or launch")
    action.add_argument("--execute", action="store_true", help="Explicitly run the serial local GPU queue")
    parser.add_argument("--limit", type=int, help="First N jobs in registered order; 2 validates the first P0 and neural states")
    parser.add_argument("--max-wall-hours", type=float, default=24., help="Per-invocation wall deadline, including running jobs (default 24); stops only own container")
    parser.add_argument("--min-free-gib", type=float, default=5., help="Repository volume reserve checked before each case (default 5 GiB)")
    args = parser.parse_args()
    try:
        checked = preflight(args)
        if args.dry_run:
            manifest, registry, plan, selection, completions, host = checked
            print(json.dumps(dict(dry_run=True, no_writes=True, gpu_work_performed=False,
                local_only=True, rental_compute_usd=0, host=host, manifest=plan["manifest"],
                selected_jobs=len(selection), registered_jobs=len(registry["jobs"]),
                verified_skips=[completions[job["run_id"]] for job in selection if completions[job["run_id"]] is not None],
                planned=[entry for entry in plan["jobs"] if entry["job"] in selection],
                max_wall_hours=args.max_wall_hours), indent=2))
        else:
            terminal = execute(args, checked)
            print(json.dumps(dict(status=terminal["status"], completed_states=terminal["completed_states"],
                                  all_36_predictions_sealed=terminal["all_36_predictions_sealed"],
                                  accuracy_scoring_performed=False, rental_compute_usd=0), indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Local queue refused/stopped: {error}\nNo later case was launched. Preserve evidence.\n")


if __name__ == "__main__":
    main()
