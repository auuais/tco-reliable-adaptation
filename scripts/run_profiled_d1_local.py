"""Local-only Docker launcher and owned-container cleanup for D1."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from run_profiled_d0 import ROOT, IMAGE, now, write_new
from run_profiled_d0_local import DOCKER, ENGINE, docker_json, ensure_stopped

WEIGHTS = Path("E:/ResearchData/TCO/checkpoints/MASt3R/hf")
LIMITS = {"prepare": 1850, "model": 3650, "analyze": 1850, "replay": 1850}
MEMORY = 8 * 1024**3
LABEL = "tco.profiled.d1.invocation"


def create_command(mode, name, label, attempt):
    argv = ["create", "--name", name, "--pull", "never", "--network", "none",
            "--memory", "8g", "--memory-swap", "8g",
            "--label", f"{LABEL}={label}", "--env", f"PROFILED_D0_IMAGE_ID={IMAGE}",
            "--env", "HF_HUB_OFFLINE=1", "--env", "CUBLAS_WORKSPACE_CONFIG=:4096:8"]
    if mode == "model":
        argv += ["--gpus", "all"]
    else:
        argv += ["--env", "CUDA_VISIBLE_DEVICES=", "--cpus", "2"]
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            argv += ["--env", var + "=1"]
    argv += ["--mount", f"type=bind,source={ROOT.as_posix()},target=/workspace",
             "--mount", f"type=bind,source={WEIGHTS.as_posix()},target=/weights,readonly",
             "--workdir", "/workspace", IMAGE, "python", "scripts/run_profiled_d1.py", mode]
    if attempt is not None:
        argv += ["--attempt", attempt]
    return DOCKER + argv


def inspect_matches(item, label, mode):
    config = item["HostConfig"]
    requests = config.get("DeviceRequests") or []
    gpu = any("gpu" in cap for request in requests for cap in request.get("Capabilities", []))
    return (item["Config"]["Labels"].get(LABEL) == label and item["Image"] == IMAGE
            and config["NetworkMode"] == "none" and config["Memory"] == MEMORY
            and config["MemorySwap"] == MEMORY and gpu == (mode == "model")
            and (gpu or not requests))


@contextmanager
def host_lock():
    import msvcrt
    path = ROOT / "reports/lora_profiled_d1_v1/launcher.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def check_local_engine():
    if not WEIGHTS.is_dir() or Path(ROOT).drive.upper() != "C:":
        raise RuntimeError("D1 uses the local Windows checkout and existing weights")
    if docker_json(["image", "inspect", IMAGE])[0]["Id"] != IMAGE:
        raise RuntimeError("Wrong local image")
    running = subprocess.check_output(DOCKER + ["ps", "-q"], text=True, timeout=30).splitlines()
    for container_id in running:
        item = docker_json(["inspect", container_id])[0]
        if item["Name"].startswith(("/tco-profiled-", "/tco-m3-d0-", "/tco-profiled-d1-")) or (
                item["HostConfig"].get("DeviceRequests")):
            raise RuntimeError("Existing research or GPU job owns the local device: " + item["Name"])


def launch(mode, name, attempt):
    check_local_engine()
    target = ROOT / "results/lora_profiled_d1_v1/launches" / name
    logs = ROOT / "reports/lora_profiled_d1_v1/launches" / name
    target.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    label = uuid.uuid4().hex
    argv = create_command(mode, name, label, attempt)
    limit = LIMITS[mode]
    start = time.monotonic()
    write_new(target / "started.json", dict(started_at_utc=now(), name=name, label=label,
              engine=ENGINE, argv=argv, outer_deadline_seconds=limit, rental_usd=0))
    owned = None
    state = None
    error = None
    cleanup = []
    exit_code = None
    try:
        owned = subprocess.check_output(argv, text=True, timeout=30).strip()
        item = docker_json(["inspect", owned])[0]
        if not inspect_matches(item, label, mode):
            raise RuntimeError("Container does not match the registered local resource limits")
        write_new(target / "ownership.json", dict(id=owned, label=label, image=item["Image"],
                  gpu_requested=mode == "model", device_requests=item["HostConfig"]["DeviceRequests"],
                  network=item["HostConfig"]["NetworkMode"], memory=item["HostConfig"]["Memory"]))
        with (logs / "stdout.log").open("xb") as stream:
            exit_code = subprocess.run(DOCKER + ["start", "--attach", owned], stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=limit).returncode
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        if owned is None:
            try:
                item = docker_json(["inspect", name])[0]
                if item["Config"]["Labels"].get(LABEL) == label:
                    owned = item["Id"]
            except Exception as e:
                cleanup.append(dict(type=type(e).__name__, message=str(e)))
    finally:
        if owned is not None:
            state, errors = ensure_stopped(owned)
            cleanup.extend(errors)
        observed = None if state is None else state["State"]
        stopped = observed is not None and observed["Running"] is False
        receipt = dict(finished_at_utc=now(), name=name, container_id=owned, state=observed,
                       verified_stopped=stopped, exit_code=exit_code, error=error,
                       cleanup_errors=cleanup, wall_seconds=time.monotonic()-start, rental_usd=0)
        write_new(target / "terminal.json", receipt)
    print(json.dumps(receipt), flush=True)
    if error is not None or exit_code != 0 or not stopped or observed["ExitCode"] != 0:
        raise RuntimeError("D1 stage failed; see " + str(logs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(LIMITS))
    parser.add_argument("--attempt")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.mode != "prepare" and not re.fullmatch(r"d1_[0-9]{2}", args.attempt or ""):
        parser.error("Provide a valid attempt ID")
    name = args.name or f"tco-profiled-d1-{args.mode}-{(args.attempt or '01').replace('_', '-')}"
    if not re.fullmatch(r"tco-profiled-d1-[a-z0-9-]{1,65}", name):
        parser.error("Provide a fresh D1 container name")
    with host_lock():
        launch(args.mode, name, args.attempt)


if __name__ == "__main__":
    main()
