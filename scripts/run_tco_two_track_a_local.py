"""Run only committed Track A jobs using the pinned local RTX 3080 and a reference-free container."""
import argparse
from pathlib import Path
import re
import subprocess
import time
from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from run_tco_courtyard_camera_local import check_device_available
from run_tco_final_ln_local import classify, device, docker_base, mount, utc
from tco_final_ln_protocol import ROOT, IMAGE, read, seal, sha

STUDY = "tco_two_track_v1"
RAW = ROOT / "reports" / STUDY
PREP = Path("E:/ResearchData/TCO/preparation") / STUDY
WORKER = ROOT / "scripts/run_tco_two_track_a.py"
POLICY = ROOT / "scripts/tco_two_track_hybrid.py"
MASTER = ROOT / "manifests" / (STUDY + ".json")


def committed(*paths):
    names = [p.resolve().relative_to(ROOT).as_posix() for p in paths]
    subprocess.run(["git","ls-files","--error-unmatch","--",*names], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["git","diff","--exit-code","HEAD","--",*names], cwd=ROOT, check=True, capture_output=True)


def completed(job):
    attempts = sorted((RAW/"cases"/job["id"]).glob("attempt[0-9][0-9]"))
    for attempt in attempts:
        rec = read(attempt/"terminal.json")
        if rec["status"] == "completed":
            if rec["complete_sha256"] != sha(attempt/"model/complete.json"):
                raise ValueError("Previously finished result changed")
            return attempt
    return None


def launch(job, ledger):
    identifier = job["id"]
    if any(re.fullmatch(r"[A-Za-z0-9_-]+", value) is None for value in (identifier, job["slug"])):
        raise ValueError("Nonportable case id")
    if completed(job):
        return "completed"
    previous = sorted((RAW/"cases"/identifier).glob("attempt[0-9][0-9]"))
    if previous:
        rec = read(previous[-1]/"terminal.json")
        if rec["status"] not in ("timeout","interrupted","provisioning") or len(previous)>=2:
            return rec["status"]
    check_device_available()
    gpu = device()
    packet = PREP/"model"/(job["slug"]+".npz")
    if sha(packet) != job["input_file_sha256"]:
        raise ValueError("Portable input differs from registered file")
    count = len(previous)+1
    case = RAW/"cases"/identifier/f"attempt{count:02d}"
    case.mkdir(parents=True,exist_ok=False)
    seal(case/"job.json",job)
    name = "tco-track-a-"+identifier.lower().replace("_","-")+"-"+str(count)
    cmd = docker_base(name,True)
    cmd[cmd.index("--label")+1] = "tco.study="+STUDY
    cmd += mount(Path("E:/ResearchData/TCO/checkpoints/VGGT-1B"),"/data/checkpoints/VGGT-1B")
    cmd += mount(packet,"/inputs/room.npz") + mount(case,"/case",True)
    init_args = []
    if job.get("initialization_sha256"):
        artifact = PREP/"initializations"/(job["initialization_sha256"]+".safetensors")
        if sha(artifact) != job["initialization_sha256"]:
            raise ValueError("Wrong training artifact")
        cmd += mount(artifact,"/inputs/init.safetensors")
        init_args = ["--initialization","/inputs/init.safetensors"]
    cmd += [IMAGE,"python","-B","-u","scripts/"+WORKER.name,"--job","/case/job.json","--inputs","/inputs/room.npz","--out","/case/model",*init_args]
    deadline = 3600 if job["steps"]>=120 else 1200
    seal(case/"start.json",dict(id=identifier,name=name,started=utc(),device=gpu,image=IMAGE,
         revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         registration_sha256=sha(MASTER),ledger_sha256=sha(ledger),worker_sha256=sha(WORKER),policy_sha256=sha(POLICY),
         input_file_sha256=sha(packet),deadline_seconds=deadline,rental_compute_usd=0))
    cid,code,error = None,None,None
    status = "provisioning"
    started = time.monotonic()
    print(utc(),"START",identifier,count,flush=True)
    try:
        cid = subprocess.check_output(cmd,text=True,timeout=40).strip()
        info, = docker_json(["inspect",cid])
        if (info["Image"]!=IMAGE or info["HostConfig"]["NetworkMode"]!="none"
                or info["Config"]["Labels"].get("tco.study")!=STUDY or not info["HostConfig"].get("DeviceRequests")
                or any(m["RW"] for m in info["Mounts"] if m["Destination"] not in ("/case","/root/.cache/torch_extensions"))):
            raise RuntimeError("Wrong local model sandbox")
        seal(case/"container.json",dict(id=cid,name=name))
        with (case/"stdout.log").open("xb") as log:
            code = subprocess.run(DOCKER+["start","--attach",cid],stdout=log,stderr=subprocess.STDOUT,timeout=deadline).returncode
        status = classify(case/"stdout.log",code,case/"model")
    except subprocess.TimeoutExpired as exc:
        status,error = ("timeout" if cid else "provisioning"),str(exc)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        error = repr(exc)
    finally:
        if cid:
            stopped,issues = ensure_stopped(cid)
            if not stopped or stopped["State"]["Running"]:
                raise RuntimeError("Could not stop owned worker "+str(issues))
        done = case/"model/complete.json"
        seal(case/"terminal.json",dict(id=identifier,status=status,error=error,attempt=count,utc=utc(),
             complete_sha256=sha(done) if done.is_file() else None,exit_code=code,stopped=True,
             wall_seconds=time.monotonic()-started,rental_compute_usd=0))
        if cid:
            subprocess.run(DOCKER+["rm",cid],check=True,timeout=25,capture_output=True)
        print(utc(),status,identifier,flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=Path,required=True)
    parser.add_argument("--only",action="append",default=[])
    parser.add_argument("--status",action="store_true")
    args = parser.parse_args()
    ledger = args.jobs.resolve()
    if ledger.parent != ROOT/"manifests" or not ledger.name.startswith(STUDY+"_"):
        raise ValueError("Expected registered Track A ledger")
    committed(ledger, MASTER, WORKER, POLICY, Path(__file__))
    jobs = read(ledger)["jobs"]
    if args.only:
        jobs = [j for j in jobs if j["id"] in args.only]
        if {j["id"] for j in jobs} != set(args.only):
            raise ValueError("Unknown registered case")
    if args.status:
        for j in jobs:
            end = sorted((RAW/"cases"/j["id"]).glob("attempt*/terminal.json"))
            print(j["id"],read(end[-1])["status"] if end else "pending")
        return
    if any(j["profile"]=="clean" for j in jobs) and not read(RAW/"qualification_pass.json")["passed"]:
        raise ValueError("Engineering qualification failed")
    with host_lock():
        for j in jobs:
            status = launch(j,ledger)
            if status in ("timeout","interrupted","provisioning"):
                status = launch(j,ledger)
            if status != "completed":
                raise RuntimeError("Inspect "+j["id"]+": "+status)


if __name__ == "__main__":
    main()
