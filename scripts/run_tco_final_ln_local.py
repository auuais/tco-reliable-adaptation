"""Launch registered runs serially on the local RTX 3080; reference packets stay off the model mount."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from run_tco_courtyard_camera_local import check_device_available
from tco_final_ln_protocol import ALLOCATOR, IMAGE, LOCK, MANIFEST, PREP, RAW, ROOT, STUDY, read, seal, sha


def utc():
    return datetime.now(timezone.utc).isoformat()


def mount(src, dst, writable=False):
    if not src.exists():
        raise FileNotFoundError(src)
    return ["--mount", f"type=bind,source={src.as_posix()},target={dst}" + ("" if writable else ",readonly")]


def docker_base(name, gpu=False):
    args = DOCKER + ["create","--name",name,"--pull","never","--network","none","--shm-size","8g",
                      "--label",f"tco.study={STUDY}", "--env","PYTHONDONTWRITEBYTECODE=1",
                      "--env","PYTHONUNBUFFERED=1", "--env","HF_HUB_OFFLINE=1","--env","TRANSFORMERS_OFFLINE=1",
                      "--env","OMP_NUM_THREADS=3","--env","MKL_NUM_THREADS=3","--env","OPENBLAS_NUM_THREADS=3",
                      "--workdir","/workspace"]
    if gpu:
        args += ["--gpus","all","--env",f"PYTORCH_CUDA_ALLOC_CONF={ALLOCATOR}",
                 "--mount","type=volume,source=tco-torch-extensions,target=/root/.cache/torch_extensions"]
    else:
        args += ["--cpus","3","--env","CUDA_VISIBLE_DEVICES="]
    for name in ("scripts","configs","third_party/TCO"):
        args += mount(ROOT / name, "/workspace/" + name)
    return args


def device():
    cmd = ["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu","--format=csv,noheader,nounits"]
    text = subprocess.check_output(cmd, text=True, timeout=10).strip()
    if not re.fullmatch(r"NVIDIA GeForce RTX 3080, [\d.]+, 10240, \d+, \d+", text):
        raise RuntimeError("Wrong physical device: " + text)
    return text


def attempts(identifier):
    return sorted((RAW / "cases" / identifier).glob("attempt[0-9][0-9]"))


def success(job):
    valid = []
    for directory in attempts(job["id"]):
        receipt = read(directory / "terminal.json")
        if receipt["status"] == "completed":
            if receipt["complete_sha256"] != sha(directory / "model/complete.json"):
                raise ValueError("Completed case checksum changed: " + job["id"])
            valid.append(directory)
    if len(valid) > 1:
        raise RuntimeError("Duplicate successful receipts")
    return valid[0] if valid else None


def classify(log, rc, modeldir):
    if rc == 0 and (modeldir / "complete.json").is_file():
        return "completed"
    text = log.read_text(errors="replace")
    if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
        return "oom"
    return "error"


def launch(job):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",job["id"]) or not re.fullmatch(r"[a-zA-Z0-9_-]+",job["slug"]):
        raise ValueError("Invalid identifier")
    previous = attempts(job["id"])
    if success(job):
        return "completed"
    if previous:
        last = read(previous[-1] / "terminal.json")
        if len(previous) >= 2 or last["status"] not in ("provisioning","interrupted","timeout"):
            return last["status"]
    check_device_available()
    telemetry = device()
    inp = PREP / "model" / (job["slug"] + ".npz")
    if sha(inp) != job["input_file_sha256"]:
        raise ValueError("Input file differs from locked stimulus")
    count = len(previous) + 1
    directory = RAW / "cases" / job["id"] / f"attempt{count:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    seal(directory / "job.json",job)
    name = "tco-ln-" + job["id"].lower().replace("_","-") + "-" + str(count)
    cmd = docker_base(name, True)
    cmd += mount(Path("E:/ResearchData/TCO/checkpoints/VGGT-1B"),"/data/checkpoints/VGGT-1B")
    cmd += mount(inp,"/inputs/room.npz") + mount(directory,"/case",True)
    cmd += [IMAGE,"python","-B","-u","scripts/run_tco_final_ln_model.py","--job","/case/job.json",
            "--inputs","/inputs/room.npz","--out","/case/model"]
    created = utc()
    repl = ROOT/"manifests"/(STUDY+"_replication.json")
    seal(directory / "start.json", dict(job=job["id"],name=name,started=created,device=telemetry,
         image=IMAGE,registration_sha256=sha(MANIFEST),primary_lock_sha256=sha(LOCK) if job["role"] in ("primary","anchor") else None,
         replication_lock_sha256=sha(repl) if job["role"] == "replication" else None,
         input_file_sha256=sha(inp),worker_sha256=sha(ROOT / "scripts/run_tco_final_ln_model.py"),deadline_seconds=1200,rental_compute_usd=0))
    cid,rc,error,stop_state = None,None,None,None
    status = "provisioning"
    log = directory / "stdout.log"
    started = time.monotonic()
    print(created, "START", job["id"],count, flush=True)
    try:
        cid = subprocess.check_output(cmd, text=True, timeout=40).strip()
        info, = docker_json(["inspect",cid])
        if (info["Image"] != IMAGE or info["HostConfig"]["NetworkMode"] != "none" or info["Config"]["Labels"].get("tco.study") != STUDY
                or not info["HostConfig"].get("DeviceRequests") or
                any(m["RW"] for m in info["Mounts"] if m["Destination"] not in ("/case","/root/.cache/torch_extensions"))):
            raise RuntimeError("Wrong container settings")
        seal(directory / "container.json", dict(id=cid,name=name))
        with log.open("xb") as stream:
            rc = subprocess.run(DOCKER+["start","--attach",cid], stdout=stream,stderr=subprocess.STDOUT,timeout=1200).returncode
        status = classify(log,rc,directory/"model")
    except subprocess.TimeoutExpired as exc:
        error = str(exc)
        status = "timeout" if cid else "provisioning"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        error = repr(exc)
    finally:
        if cid:
            stop_state,issues = ensure_stopped(cid)
            if stop_state is None or stop_state["State"]["Running"]:
                raise RuntimeError("Unverified owned-container stop: "+str(issues))
        result = directory / "model/complete.json"
        seal(directory / "terminal.json",dict(id=job["id"],status=status,attempt=count,utc=utc(),wall_seconds=time.monotonic()-started,
             exit_code=rc,error=error,stopped=bool(not cid or not stop_state["State"]["Running"]),rental_compute_usd=0,
             complete_sha256=sha(result) if result.is_file() else None))
        if cid:
            subprocess.run(DOCKER+["rm",cid],check=True,capture_output=True,timeout=25)
        print(utc(),status,job["id"], flush=True)
    return status


def select(stage):
    plan = read(MANIFEST)
    if stage == "engineering":
        return plan[stage]
    if not (RAW / "qualification_pass.json").is_file() or not read(RAW / "qualification_pass.json").get("passed"):
        raise ValueError("Qualification is pending")
    if stage in ("development","duplicates"):
        return plan[stage]
    if stage in ("primary","anchor","replication"):
        lock = read(ROOT/"manifests"/(STUDY+"_replication.json") if stage == "replication" else LOCK)
        for rel,digest in lock["pinned_files"].items():
            if sha(ROOT/rel) != digest:
                raise ValueError("Lock bytes changed: "+rel)
        return lock["jobs" if stage == "replication" else stage]
    if stage == "diagnostics":
        return read(ROOT/"manifests"/(STUDY+"_followup.json"))["diagnostics"]
    return read(ROOT/"manifests"/(STUDY+"_"+stage+".json"))["jobs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("engineering","development","duplicates","primary","anchor","followup","diagnostics","replication"))
    parser.add_argument("--only", action="append",default=[])
    parser.add_argument("--status",action="store_true")
    args = parser.parse_args()
    jobs = select(args.stage)
    if args.only:
        jobs = [j for j in jobs if j["id"] in args.only]
        if {j["id"] for j in jobs} != set(args.only):
            raise ValueError("Unregistered case")
    if args.status:
        for j in jobs:
            a = attempts(j["id"])
            end = a[-1]/"terminal.json" if a else None
            print(j["id"],read(end)["status"] if end and end.is_file() else "started" if a else "pending")
        return
    with host_lock():
        for j in jobs:
            status = launch(j)
            if status in ("provisioning","interrupted","timeout"):
                status = launch(j)
            if status == "error" or args.stage == "engineering" and status != "completed":
                raise RuntimeError("Inspect "+j["id"]+": "+status)


if __name__ == "__main__":
    main()
