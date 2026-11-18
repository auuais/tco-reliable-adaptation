"""Run the registered compensation bridge or one scientific job locally."""
import argparse
import json
import subprocess
import time
from uuid import uuid4

from profiled_compensation_protocol import PUBLIC, RAW, ROOT, require
from run_profiled_d0 import IMAGE, now, write_new
from run_profiled_d0_local import DOCKER, ENGINE, docker_json, ensure_stopped
from run_profiled_d1_local import LABEL, check_local_engine, create_command, host_lock, inspect_matches


def launch(name, inner, seconds, *, check_engine=True):
    if check_engine:
        check_local_engine()
    records = ROOT / PUBLIC / "launches" / name
    logs = ROOT / RAW / "launches" / name
    records.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    label = uuid4().hex
    base = create_command("model", name, label, None)
    argv = base[:base.index(IMAGE)+1] + ["python"] + inner
    write_new(records / "started.json", dict(name=name, label=label, argv=argv, engine=ENGINE,
                                            limit_seconds=seconds, started_at_utc=now(), rental_usd=0))
    owned = None
    error = None
    state = None
    exit_code = None
    cleanup = []
    start = time.monotonic()
    try:
        owned = subprocess.check_output(argv, timeout=30, text=True).strip()
        item = docker_json(["inspect", owned])[0]
        require(inspect_matches(item, label, "model"), "Local image, memory, network or ownership differs")
        write_new(records / "ownership.json", dict(id=owned, label=label, image=item["Image"],
                  memory_bytes=item["HostConfig"]["Memory"], device_requests=item["HostConfig"]["DeviceRequests"]))
        with (logs / "stdout.log").open("xb") as stream:
            exit_code = subprocess.run(DOCKER + ["start", "--attach", owned], stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=seconds).returncode
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        if owned is None:
            try:
                item = docker_json(["inspect", name])[0]
                if item["Config"]["Labels"].get(LABEL) == label:
                    owned = item["Id"]
            except Exception as inspect_error:
                cleanup.append(str(inspect_error))
    finally:
        if owned is not None:
            final, failures = ensure_stopped(owned)
            cleanup.extend(failures)
            state = None if final is None else final["State"]
        stopped = state is not None and state["Running"] is False
        receipt = dict(name=name, container_id=owned, state=state, verified_stopped=stopped,
                       launcher_exit_code=exit_code, error=error, cleanup_errors=cleanup,
                       wall_seconds=time.monotonic()-start, finished_at_utc=now(), rental_usd=0)
        write_new(records / "terminal.json", receipt)
    print(json.dumps(receipt), flush=True)
    if not stopped or error is not None or exit_code != 0 or state["ExitCode"] != 0:
        raise RuntimeError("Local compensation case failed. Logs: " + str(logs))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", required=True, choices=("bridge01", "bridge02", "bridge03"))
    args = p.parse_args()
    with host_lock():
        launch("tco-profiled-compensation-" + args.bridge,
               ["scripts/check_profiled_compensation_bridge.py", "--attempt", args.bridge], 1850)


if __name__ == "__main__":
    main()
