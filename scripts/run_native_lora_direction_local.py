"""Launch only local RTX 3080 containers and retain terminal receipts."""
import argparse
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from native_lora_direction_inputs import ENVIRONMENTS, RAW, SIDES
from run_profiled_d0 import IMAGE, now, write_new
from run_profiled_d0_local import DOCKER, ENGINE, docker_json, ensure_stopped
from run_profiled_d1_local import LABEL, check_local_engine, create_command, host_lock, inspect_matches


def launch(tag, args, limit):
    suffix = uuid4().hex[:8]
    name = f"tco-profiled-native-direction-{tag}-{suffix}"
    folder = RAW / "launches" / name
    folder.mkdir(parents=True, exist_ok=False)
    base = create_command("model", name, suffix, None)
    command = base[:base.index(IMAGE)+1] + ["python", "-u", "scripts/run_native_lora_direction.py", *args]
    write_new(folder / "started.json", dict(tag=tag, command=command, engine=ENGINE,
              started_at=now(), rental_usd=0))
    container, code, error = None, None, None
    start = time.monotonic()
    try:
        container = subprocess.check_output(command, text=True, timeout=30).strip()
        details = docker_json(["inspect", container])[0]
        if not inspect_matches(details, suffix, "model"):
            raise ValueError("Wrong pinned image, ownership, GPU allocation or offline runtime")
        with (folder / "stdout.log").open("xb") as log:
            code = subprocess.run(DOCKER + ["start", "--attach", container], stdout=log,
                                  stderr=subprocess.STDOUT, timeout=limit).returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if container is None:
            try:
                candidate = docker_json(["inspect", name])[0]
                if candidate["Config"]["Labels"].get(LABEL) == suffix:
                    container = candidate["Id"]
            except subprocess.SubprocessError:
                pass
    finally:
        item, cleanup = ensure_stopped(container) if container else (None, [])
        state = item["State"] if item else None
        stopped = state is not None and not state["Running"]
        record = dict(tag=tag, container=container, image=IMAGE, terminal_state=state,
                      stopped=stopped, code=code, error=error, cleanup=cleanup,
                      seconds=time.monotonic()-start, rental_usd=0, finished_at=now())
        write_new(folder / "terminal.json", record)
        print(name, code, stopped, round(record["seconds"], 1), flush=True)
    if error or not stopped or code != 0 or state["ExitCode"] != 0:
        raise RuntimeError(f"Local run {name} did not complete. Logs: {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering", choices=("bridge01", "bridge02", "bridge03"))
    parser.add_argument("--room", choices=ENVIRONMENTS)
    args = parser.parse_args()
    with host_lock():
        if args.engineering:
            check_local_engine()
            launch(args.engineering, ["bridge", "--bridge-attempt", args.engineering], 600)
        else:
            from native_lora_direction_registration import verify_registration
            verify_registration(require_committed=True)
            for environment in (args.room,) if args.room else ENVIRONMENTS:
                for side in SIDES:
                    check_local_engine()
                    launch(f"{environment}-{side}", ["case", environment, side], 2100)


if __name__ == "__main__":
    main()
