"""Screen the frozen TCO recipe and full reference on one ten-view window."""

import argparse
import subprocess
import time

from run_profiled_d0 import now, read, sha, write_new
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from tco_one_percent_protocol import ALLOCATOR, CONFIRM_MANIFEST, DATA, IMAGE, PRIVATE, PUBLIC, ROOT
from tco_one_percent_protocol import file_hashes, job_for, read_committed, verify


MANIFEST = ROOT / "manifests/tco_one_percent_view_stress_v1.json"
WORK = PRIVATE / "view_stress"


def register():
    verify()
    lock = read_committed(CONFIRM_MANIFEST)
    scene = "heads/seq-01"
    frames = list(range(0, 1000, 100))
    method, updates = lock["candidate"].rsplit("_", 1)
    jobs = [job_for(scene, arm, steps, 42, "confirmation", frames=frames,
                    name=f"stress_{arm}_{steps}") for arm, steps in (("four", 40), (method, int(updates)))]
    write_new(MANIFEST, dict(confirmation_sha256=sha(CONFIRM_MANIFEST, text=True),
                             launcher_sha256=sha(__file__, text=True), jobs=jobs,
                             width=518, timeout_seconds=360, rental_compute_usd=0,
                             input_source_sha256=file_hashes(DATA / "data", {scene: frames})))


def launch(plan, job):
    name = job["id"]
    folder = WORK / name
    terminal = folder / "terminal.json"
    if terminal.is_file():
        return read(terminal)
    folder.mkdir(parents=True, exist_ok=False)
    inline = (
        "import sys; from pathlib import Path; sys.path.insert(0,'scripts'); "
        "from run_tco_one_percent import run, check_case_inputs, DATA_ROOT; from run_profiled_d0 import read; "
        f"p=read(Path('/workspace/manifests/{MANIFEST.name}')); "
        f"j=next(q for q in p['jobs'] if q['id']=={name!r}); "
        "check_case_inputs(j,p['input_source_sha256'],DATA_ROOT); "
        f"run(j,Path('/workspace/reports/tco_one_percent_v1/view_stress/{name}/case'),p['width'])"
    )
    container_name = f"tco-one-{name.replace('_', '-')}"
    command = DOCKER + ["create", "--pull", "never", "--network", "none", "--name", container_name,
                        "--gpus", "all", "--shm-size", "8g", "--workdir", "/workspace",
                        "--mount", f"type=bind,source={ROOT.as_posix()},target=/workspace",
                        "--mount", f"type=bind,source={DATA.as_posix()},target=/data,readonly",
                        "--mount", "type=volume,source=tco-torch-extensions,target=/root/.cache/torch_extensions",
                        "--env", "MKL_NUM_THREADS=4", "--env", "OPENBLAS_NUM_THREADS=4", "--env", "HF_HUB_OFFLINE=1",
                        "--env", f"PYTORCH_CUDA_ALLOC_CONF={ALLOCATOR}",
                        IMAGE, "python", "-u", "-c", inline]
    write_new(folder / "start.json", dict(case=name, utc=now(), argv=command))
    start = time.monotonic()
    container, code, fault = None, None, None
    log = folder / "stdout.log"
    try:
        container = subprocess.check_output(command, text=True, timeout=45).strip()
        with log.open("xb") as stream:
            code = subprocess.run(DOCKER + ["start", "--attach", container], stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=plan["timeout_seconds"]).returncode
    except Exception as exc:
        fault = type(exc).__name__
    finally:
        final, stop_issues = ensure_stopped(container) if container else (None, [])
    stopped = bool(final and not final["State"]["Running"])
    if container and not stopped:
        raise RuntimeError("The stress-test container did not stop")
    summary = read(folder / "case/complete.json") if code == 0 and stopped and fault is None else None
    oom = bool(log.is_file() and "CUDA out of memory" in log.read_text(encoding="utf-8", errors="replace"))
    verdict = "completed" if summary else "oom" if oom else "timeout" if fault == "TimeoutExpired" else "error"
    output = dict(case=name, verdict=verdict, stopped=stopped, exit_code=code, error=fault, stop_issues=stop_issues,
                  wall_seconds=time.monotonic() - start, result=summary, log_sha256=sha(log) if log.is_file() else None,
                  rental_compute_usd=0)
    write_new(terminal, output)
    print(name, verdict, flush=True)
    return output


def run():
    verify()
    plan = read_committed(MANIFEST)
    if (plan["confirmation_sha256"] != sha(CONFIRM_MANIFEST, text=True)
            or plan["launcher_sha256"] != sha(__file__, text=True)
            or not (PUBLIC / "confirmation_decision.json").is_file()):
        raise ValueError("This secondary test follows the published five-view verdict")
    with host_lock():
        others = subprocess.check_output(DOCKER + ["ps", "-q"], text=True, timeout=20).splitlines()
        if others and any(item["HostConfig"].get("DeviceRequests") for item in docker_json(["inspect", *others])):
            raise RuntimeError("A Docker job is using the local GPU")
        rows = [launch(plan, item) for item in plan["jobs"]]
    write_new(PUBLIC / "view_count_stress.json", dict(registration_sha256=sha(MANIFEST, text=True),
                                                      runs=rows, rental_compute_usd=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("register", "run"))
    args = cli.parse_args()
    register() if args.action == "register" else run()
