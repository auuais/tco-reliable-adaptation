"""Run the six registered step-selection cases in serial offline Docker containers."""
import argparse
import subprocess
import time
from uuid import uuid4

from native_lora_step_inputs import RAW, ROOMS, cases, verify_registration
from prepare_profiled_inputs import read
from run_profiled_d0 import IMAGE, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import LABEL, check_local_engine, create_command, host_lock, inspect_matches


def launch(env, side, attempt):
    job = f"{env}_{side}"
    receipt_path = RAW / "launches" / job / attempt / "terminal.json"
    if receipt_path.exists():
        previous = read(receipt_path)
        if previous["exit_code"] == 0 and previous["stopped"] and not previous["error"]:
            return
        raise ValueError(f"Preserved failed attempt for {job}. Select a fresh technical attempt after diagnosis.")
    check_local_engine()
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    name = f"tco-profiled-step-{env}-{side}-{attempt}"
    label = uuid4().hex
    base = create_command("model", name, label, None)
    command = base[:base.index(IMAGE)+1] + [
        "python", "-u", "scripts/run_native_lora_step.py", env, side, "--attempt", attempt]
    write_new(receipt_path.with_name("start.json"), dict(container=name, argv=command, rental_compute_usd=0))
    container, exit_code, error = None, None, None
    start = time.monotonic()
    try:
        container = subprocess.check_output(command, text=True, timeout=30).strip()
        if not inspect_matches(docker_json(["inspect", container])[0], label, "model"):
            raise ValueError("Local image, GPU allocation, ownership or offline policy differs")
        with receipt_path.with_name("stdout.log").open("xb") as log:
            exit_code = subprocess.run(DOCKER + ["start", "--attach", container], stdout=log,
                                       stderr=subprocess.STDOUT, timeout=1800).returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if container is None:
            try:
                item = docker_json(["inspect", name])[0]
                if item["Config"]["Labels"].get(LABEL) == label:
                    container = item["Id"]
            except subprocess.SubprocessError:
                pass
    finally:
        final, cleanup = ensure_stopped(container) if container else (None, [])
        state = final["State"] if final else None
        stopped = bool(state and not state["Running"])
        record = dict(job=job, attempt=attempt, exit_code=exit_code, stopped=stopped,
                      state=state, error=error, cleanup=cleanup, wall_seconds=time.monotonic()-start,
                      rental_compute_usd=0)
        write_new(receipt_path, record)
    if not stopped or exit_code or error or state["ExitCode"] != 0:
        raise RuntimeError(f"Local native run failed: {receipt_path.parent}")
    print(job, attempt, round(record["wall_seconds"], 1), "seconds, stopped", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", choices=ROOMS)
    parser.add_argument("--attempt", choices=("attempt01", "attempt02"), default="attempt01")
    args = parser.parse_args()
    verify_registration(committed=True)
    with host_lock():
        for room, side in cases():
            if args.room is None or args.room == room:
                launch(room, side, args.attempt)
