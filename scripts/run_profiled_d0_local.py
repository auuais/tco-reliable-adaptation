"""Local Docker launcher for D0, with an independently enforced wall deadline."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

from run_profiled_d0 import ROOT, IMAGE, REGISTRY, write_new

ENGINE = "npipe:////./pipe/dockerDesktopLinuxEngine"
DOCKER = ["docker", "--host", ENGINE]
OUTER_SECONDS = 1850


def docker_json(arguments):
    return json.loads(subprocess.check_output(DOCKER+arguments, text=True, timeout=30))


def create_argv(root, weights, name, inner):
    if not re.fullmatch(r"tco-m3-d0-[a-z0-9-]{1,65}",name):
        raise ValueError("Use a fresh tco-m3-d0- container name")
    return DOCKER+["create", "--name",name,"--network","none","--cpus","2","--memory","6g",
        "--env","CUDA_VISIBLE_DEVICES=","--env","OMP_NUM_THREADS=1",
        "--env","OPENBLAS_NUM_THREADS=1","--env","MKL_NUM_THREADS=1","--env","NUMEXPR_NUM_THREADS=1",
        "--env","PROFILED_D0_IMAGE_ID="+IMAGE,
        "--mount","type=bind,source="+Path(root).resolve().as_posix()+",target=/workspace",
        "--mount","type=bind,source="+Path(weights).resolve().as_posix()+",target=/weights,readonly",
        "--workdir","/workspace",IMAGE]+inner


def ensure_stopped(container_id):
    """Inspect, stop, then kill if necessary; never claim an unobserved shutdown."""
    errors=[]
    state=None
    for action in (None,"stop","kill"):
        if action is not None:
            args=[action,"--time","5",container_id] if action=="stop" else [action,container_id]
            try:
                result=subprocess.run(DOCKER+args,timeout=20,check=False,capture_output=True)
                if result.returncode!=0:
                    errors.append(dict(action=action,returncode=result.returncode))
            except BaseException as exc:
                errors.append(dict(action=action,type=type(exc).__name__,message=str(exc)))
        try:
            state=docker_json(["inspect",container_id])[0]
            if state["State"]["Running"] is False:
                return state,errors
        except BaseException as exc:
            errors.append(dict(action="inspect",type=type(exc).__name__,message=str(exc)))
            state=None
    return state,errors


def launch(root, weights, name, inner):
    root=Path(root).resolve()
    archive=root/"reports/lora_profiled_d0_v1/launches"/name
    if archive.exists():
        raise FileExistsError("Preserve existing launcher evidence")
    image=docker_json(["image","inspect",IMAGE])[0]
    if image["Id"]!=IMAGE:
        raise ValueError("Local Docker image differs")
    argv=create_argv(root,weights,name,inner)
    archive.mkdir(parents=True,exist_ok=False)
    created=subprocess.check_output(argv,text=True,timeout=30).strip()
    start=time.monotonic()
    timed_out=False
    error=None
    client_exit=-1
    try:
        inspect=docker_json(["inspect",created])[0]
        if inspect["Image"]!=IMAGE or inspect["HostConfig"].get("DeviceRequests") not in (None,[]):
            raise ValueError("Wrong actual container image or GPU request")
        if inspect["HostConfig"]["NetworkMode"]!="none":
            raise ValueError("D0 container must have no network")
        initial=dict(schema="profiled_d0_local_launch_v1",started_at_utc=datetime.now(timezone.utc).isoformat(),
            engine=ENGINE,name=name,container_id=created,image_id=inspect["Image"],argv=argv,
            device_requests=inspect["HostConfig"].get("DeviceRequests"),
            network_mode=inspect["HostConfig"]["NetworkMode"],
            memory_limit_bytes=inspect["HostConfig"]["Memory"],
            nanocpus=inspect["HostConfig"].get("NanoCpus"),outer_deadline_seconds=OUTER_SECONDS,
            rental_usd=0,gpu_requested=False)
        write_new(archive/"started.json",initial)
        with (archive/"stdout.log").open("xb") as stream:
            result=subprocess.run(DOCKER+["start","--attach",created],stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=OUTER_SECONDS,check=False)
            client_exit=result.returncode
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        timed_out=isinstance(exc,subprocess.TimeoutExpired)
        client_exit=124 if timed_out else -1
    finally:
        terminal,cleanup_errors=ensure_stopped(created)
        receipt=dict(schema="profiled_d0_local_terminal_v1",finished_at_utc=datetime.now(timezone.utc).isoformat(),
            name=name,container_id=created,image_id=None if terminal is None else terminal["Image"],
            state=None if terminal is None else terminal["State"],error=error,cleanup_errors=cleanup_errors,
            client_exit_code=client_exit,timed_out=timed_out,wall_seconds=time.monotonic()-start,
            stopped_verified=terminal is not None and terminal["State"]["Running"] is False,rental_usd=0)
        write_new(archive/"terminal.json",receipt)
    print(json.dumps(receipt,indent=2))
    if error is not None or client_exit!=0 or not receipt["stopped_verified"] or terminal["State"]["ExitCode"]!=0:
        raise RuntimeError("D0 container did not complete successfully; retained logs: "+str(archive))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","verify-registration","run","verify"))
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--weights",type=Path,default=Path("E:/ResearchData/TCO/checkpoints/MASt3R/hf"))
    parser.add_argument("--name",required=True)
    parser.add_argument("--registry",default=REGISTRY)
    parser.add_argument("--checks")
    parser.add_argument("--attempt")
    parser.add_argument("--output")
    args=parser.parse_args()
    inner=["python","scripts/run_profiled_d0.py",args.mode,"--registry",args.registry,"--weights","/weights"]
    for option in ("checks","attempt","output"):
        value=getattr(args,option)
        if value is not None:
            inner += ["--"+option,value]
    launch(args.root,args.weights,args.name,inner)


if __name__=="__main__":
    main()
