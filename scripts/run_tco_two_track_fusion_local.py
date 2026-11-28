"""Prepare and run CPU-only Track B archived jobs on the physical workstation."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

from run_profiled_d0_local import DOCKER,docker_json,ensure_stopped
from run_tco_final_ln_local import mount
from run_tco_two_track_fusion import SEEDS,STUDY,require,signature
from tco_final_ln_protocol import IMAGE,ROOT,read,seal,sha


PUB=ROOT/"results"/STUDY
RAW=ROOT/"reports"/STUDY/"fusion"
PREP=Path("E:/ResearchData/TCO/preparation")/STUDY
OLD=ROOT/"results/tco_one_percent_v1"
ARCH=Path("E:/ResearchData/TCO/research_results/tco_one_percent_v1/snapshots/study-v1")
SRC=RAW/"sources"
REG=ROOT/"manifests"/(STUDY+".json")


def stage():
    registered=subprocess.check_output(["git","show","HEAD:"+REG.relative_to(ROOT).as_posix()],cwd=ROOT)
    require(registered.replace(b"\r\n",b"\n")==REG.read_bytes().replace(b"\r\n",b"\n"),"Uncommitted study registration")
    receipt=read(OLD/"archive_receipt.json")
    require(sha(ARCH/"manifest.json")==receipt["manifest_sha256"] and receipt["pass_all"],"Unverified 40-step archive manifest")
    indexed={r["file"]:r for r in read(ARCH/"manifest.json")["selection"]["files"]}
    plan=read(PUB/"development_inputs.json")
    baseline={r["id"]:r for r in read(OLD/"evidence.json")["rows"]}
    rooms=[]
    for orig in plan["rooms"]:
        room=dict(orig)
        room["archives"]=[]
        for seed in SEEDS:
            name=room["room"].replace("/","_").replace("-","")+f"_four_40_{seed}"
            rel=Path("reports/tco_one_percent_v1/cases")/name/"attempt01"
            for n in ("prediction.npz","complete.json","metrics.json"):
                key=(rel/n).as_posix()
                digest=indexed[key]["sha256"]
                backup=ARCH/"data"/key
                original=ROOT/key
                require(sha(backup)==digest,"Mirror verification failed")
                if not original.exists():
                    original.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(backup,original)
                require(sha(original)==digest,"Original/mirror mismatch")
            complete=read(ROOT/rel/"complete.json")
            base=baseline[name]
            require(base["output_sha256"]==complete["output_sha256"] and
                    base["metrics"]==complete["metrics"]==read(ROOT/rel/"metrics.json") and
                    list(base["frames"])==room["frames"]==[0,200,400,600,800] and base["seed"]==seed
                    and base["scene"]==room["room"] and base["arm"]=="four" and base["steps"]==40 and
                    base["input_sha256"]==room["input_sha256"] and base["reference_sha256"]==room["reference_sha256"],
                    "Archive/source or prepared authorization disagrees")
            src=ROOT/rel/"prediction.npz"
            with np.load(src,allow_pickle=False) as arrays:
                require(set(arrays.files)=={"canonical","pipeline","depth","extrinsics","intrinsics","depth_conf"},
                        "Unexpected archived prediction arrays")
                require(signature({k:arrays[k] for k in arrays.files})==base["output_sha256"],
                        "Archived named-array signature changed")
            digest=indexed[(rel/"prediction.npz").as_posix()]["sha256"]
            dest=SRC/(name+".npz")
            SRC.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                try:
                    os.link(src,dest)
                except OSError:
                    shutil.copy2(src,dest)
            require(sha(dest)==digest,"Staging digest changed")
            room["archives"].append(dict(seed=seed,file=dest.name,digest=digest,array_sha256=base["output_sha256"],
                                         receipt_sha256=sha(ROOT/rel/"complete.json")))
        rooms.append(room)
    seal(SRC/"inputs.json",dict(rooms=rooms,registration_sha256=sha(REG),archive_manifest_sha256=receipt["manifest_sha256"],
                               study=STUDY,worker_sha256=sha(ROOT/"scripts/run_tco_two_track_fusion.py"),
                               fusion_sha256=sha(ROOT/"scripts/tco_two_track_fusion.py")))
    subprocess.run([sys.executable,"-B",str(ROOT/"scripts/test_tco_two_track_fusion.py")],check=True)
    seal(PUB/"b_fusion_qualification.json",dict(registration_sha256=sha(REG),source_lock_sha256=sha(SRC/"inputs.json"),
         archives_verified=sum(len(r["archives"]) for r in rooms),second_disk_manifest_sha256=receipt["manifest_sha256"],
         synthetic_checks=["signed_similarity","unequal_intrinsics_and_hole_support","nearest_visible_z",
                           "two_source_log_midpoint_and_degenerate_gauge","triplicate_identity"],
         test_sha256=sha(ROOT/"scripts/test_tco_two_track_fusion.py"),fusion_sha256=sha(ROOT/"scripts/tco_two_track_fusion.py"),
         qualified=True,rental_compute_usd=0))
    print("Prepared and authenticated 9 independent predictions and 3 separate input packets",flush=True)


def launch(action):
    lock=read(SRC/"inputs.json")
    require(lock["worker_sha256"]==sha(ROOT/"scripts/run_tco_two_track_fusion.py") and
            lock["fusion_sha256"]==sha(ROOT/"scripts/tco_two_track_fusion.py") and
            read(PUB/"b_fusion_qualification.json")["qualified"],"Source lock or qualifying code changed")
    name="tco-two-track-b-"+action+"-"+uuid.uuid4().hex[:10]
    receipt=RAW/"jobs"/name
    receipt.mkdir(parents=True)
    points,scores=RAW/"points",RAW/"scores"
    points.mkdir(exist_ok=True)
    scores.mkdir(exist_ok=True)
    args=DOCKER+["create","--name",name,"--network","none","--pull","never","--cpus","3","--shm-size","4g",
                 "--label",f"tco.study={STUDY}","--env","CUDA_VISIBLE_DEVICES=","--env","OMP_NUM_THREADS=3",
                 "--env","OPENBLAS_NUM_THREADS=3","--env","MKL_NUM_THREADS=3","--env","PYTHONDONTWRITEBYTECODE=1",
                 "--env","HF_HUB_OFFLINE=1","--env","TRANSFORMERS_OFFLINE=1","--workdir","/workspace"]
    for directory in ("scripts","configs","third_party/TCO"):
        args+=mount(ROOT/directory,"/workspace/"+directory)
    args+=mount(SRC,"/sources")+mount(PREP/"model","/model")
    if action=="infer":
        args+=mount(points,"/outputs",True)
    else:
        args+=mount(points,"/points")+mount(PREP/"reference","/reference")+mount(scores,"/scores",True)
    args += [IMAGE,"python","-B","-u","scripts/run_tco_two_track_fusion.py",action]
    cid=None
    ret=None
    elapsed=time.monotonic()
    seal(receipt/"start.json",dict(action=action,utc=datetime.now(timezone.utc).isoformat(),image=IMAGE,
         registration_sha256=sha(REG),source_lock_sha256=sha(SRC/"inputs.json"),rental_compute_usd=0))
    try:
        cid=subprocess.check_output(args,text=True,timeout=45).strip()
        info,=docker_json(["inspect",cid])
        mounts={v["Destination"] for v in info["Mounts"]}
        require(info["Image"]==IMAGE and not info["HostConfig"].get("DeviceRequests") and
                info["HostConfig"]["NetworkMode"]=="none" and (("/reference" in mounts)==(action!="infer"))
                and "/data" not in mounts,"Unexpected model or score sandbox")
        with (receipt/"stdout.log").open("xb") as output:
            ret=subprocess.run(DOCKER+["start","--attach",cid],stdout=output,stderr=subprocess.STDOUT,timeout=5400).returncode
        require(ret==0,f"{action} failed. Inspect {receipt}/stdout.log")
    finally:
        stopped=False
        if cid:
            state,_=ensure_stopped(cid)
            stopped=bool(state and not state["State"]["Running"])
            if stopped:
                subprocess.run(DOCKER+["rm",cid],check=True,capture_output=True,timeout=30)
        seal(receipt/"terminal.json",dict(action=action,exit_code=ret,stopped=stopped,utc=datetime.now(timezone.utc).isoformat(),
             wall_seconds=time.monotonic()-elapsed,rental_compute_usd=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","infer","score","independent"))
    args=parser.parse_args()
    if args.stage=="prepare":
        stage()
    else:
        launch(args.stage)
