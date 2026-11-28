"""Audit the two zero-update camera readouts with a reference-free projection and a separate CPU scorer."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
import uuid

from score_tco_final_ln import check_against, require
from tco_final_ln_protocol import IMAGE, read, seal, sha
from tco_two_track_protocol import MANIFEST, PRIVATE, PUBLIC, ROOT, STUDY

PREP = Path("E:/ResearchData/TCO/preparation") / STUDY
OLD = ROOT/"results/tco_one_percent_v1"
ARCH = Path("E:/ResearchData/TCO/research_results/tco_one_percent_v1/snapshots/study-v1")
KEYS = ("canonical","supplied_canonical")


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_plan(stage):
    from score_tco_two_track_a import locate
    if stage!="retrospective":
        ledger=MANIFEST.with_name(STUDY+"_"+stage+".json")
        roster=read(ledger)
        require(roster["registration_sha256"]==sha(MANIFEST),"Wrong registration")
        jobs = [j for j in roster["jobs"] if j["arm"]=="ZERO"]
        require(len(jobs)==6 and len({j["room"] for j in jobs})==6,"One zero forward per fresh room required")
        cases={}
        for j in jobs:
            _,terms,done=locate(j)
            if done:
                cases[j["id"]]=done[0]/"model"
            j["source_status"]=terms[-1]["status"]
        return dict(stage=stage,jobs=jobs,roster_sha256=sha(ledger)), cases
    arch=read(OLD/"archive_receipt.json")
    require(arch["pass_all"] and sha(ARCH/"manifest.json")==arch["manifest_sha256"],"Invalid verified archive")
    entries={r["file"]:r["sha256"] for r in read(ARCH/"manifest.json")["selection"]["files"]}
    src=PUBLIC/"development_inputs.json"
    hist=read(OLD/"evidence.json")["rows"]
    jobs, files=[],{}
    for p in read(src)["rooms"]:
        original, = [r for r in hist if r["scene"]==p["room"] and r["arm"]=="zero" and r["seed"]==42 and r["steps"]==0]
        require(all(original[k]==p[k] for k in ("frames","input_sha256","reference_sha256")),"Old control packet differs")
        rel=Path("reports/tco_one_percent_v1/cases")/original["id"]/original["attempt"]
        hashes={}
        for name in ("prediction.npz","complete.json"):
            key=(rel/name).as_posix()
            hashes[name]=entries[key]
            require(sha(ROOT/key)==sha(ARCH/"data"/key)==entries[key],"Old zero archive differs from sealed mirror")
        job=dict(**p,id="historical_"+original["id"],arm="ZERO",steps=0,seed=42,source_status="completed",
                 source_id=original["id"],archive_array_sha256=original["output_sha256"],
                 archive_prediction_sha256=hashes["prediction.npz"],archive_receipt_sha256=hashes["complete.json"])
        jobs.append(job)
        files[job["id"]]=ROOT/rel/"prediction.npz"
    return dict(stage=stage,jobs=jobs,packet_sha256=sha(src),archive_manifest_sha256=arch["manifest_sha256"],
                historical_rows_sha256=sha(OLD/"evidence.json")),files


def build_old(plan):
    import numpy as np
    import torch
    from run_tco_final_ln_model import inputs
    from run_tco_frozen_prefix import signature
    from tco_one_percent_geometry import scene_scale
    from utils.geometry import unproject_depth_to_world_points_torch
    require(not torch.cuda.is_available() and not Path("/references").exists(),"CPU-only projection without a reference")
    for i,j in enumerate(plan["jobs"]):
        folder=Path("/outputs")/j["id"]
        if (folder/"complete.json").is_file():
            receipt=read(folder/"complete.json")
            require(receipt["archive_prediction_sha256"]==j["archive_prediction_sha256"]
                    and receipt["predictions_sha256"]==sha(folder/"prediction.npz"),"Invalid existing zero case")
            continue
        packet,input_id=inputs(Path(f"/rgb/{i}.npz"),j["input_file_sha256"],False)
        src=Path(f"/archive/{i}.npz")
        require(input_id==j["input_sha256"] and sha(src)==j["archive_prediction_sha256"],"Input changed")
        with np.load(src,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        require(signature(arrays)==j["archive_array_sha256"],"Archive tensor signature differs")
        projection_start_utc=utc()
        before=time.perf_counter()
        with torch.no_grad():
            depth=torch.from_numpy(arrays["depth"]).float()
            rig=torch.from_numpy(packet["extrs"]).float()
            K=torch.from_numpy(packet["intrs"]).float()
            s,fit=scene_scale(rig,torch.from_numpy(arrays["extrinsics"])[0].float())
            require(s is not None,"Unobservable prediction scale")
            if rig.shape[-2]==3:
                bottom=rig.new_zeros((5,1,4))
                bottom[:,:,3]=1
                rig=torch.cat((rig,bottom),dim=1)
            projected=unproject_depth_to_world_points_torch(depth*s,rig[None],K[None])[0].numpy()
        projection_seconds=time.perf_counter()-before
        projection_end_utc=utc()
        output=dict(canonical=arrays["canonical"],supplied_canonical=projected)
        require(projected.shape==output["canonical"].shape and projected.dtype==np.float32 and
                all(np.isfinite(a).all() for a in output.values()),"Invalid reprojected historical control")
        folder.mkdir(exist_ok=True,parents=True)
        tmp=folder/"prediction.partial.npz"
        np.savez_compressed(tmp,**output)
        tmp.replace(folder/"prediction.npz")
        seal(folder/"complete.json",dict(**j,completed_adam_updates=0,output_sha256=signature(output),
             predictions_sha256=sha(folder/"prediction.npz"),supplied_scale=fit,projection_cpu_seconds=projection_seconds,
             projection_start_utc=projection_start_utc,projection_end_utc=projection_end_utc,
             provenance="Derived from archived ordinary zero forward; projected on CPU without references",
             rental_compute_usd=0))


def score_inside(plan, independent):
    from score_tco_two_track_a import score
    first_room=min(j["room"] for j in plan["jobs"])
    for i,j in enumerate(plan["jobs"]):
        if j["source_status"]!="completed" or (independent and j["room"]!=first_room):
            continue
        for key in KEYS:
            root=Path("/scores")
            suffix=Path(j["id"])/(key+".json")
            target=root/("independent" if independent else "first")/suffix
            if not target.exists():
                score(j,Path(f"/cases/{i}"),Path("/prepared"),target,
                      root/"first"/suffix if independent else None,output_key=key)


def launch(stage,action):
    from run_profiled_d0_local import DOCKER,docker_json,ensure_stopped
    from run_profiled_d1_local import host_lock
    from run_tco_final_ln_local import docker_base,mount
    plan, cases=source_plan(stage)
    raw=PRIVATE/"zero_controls"/stage
    plan_path=raw/"plan.json"
    seal(plan_path,plan)
    out=raw/"outputs"
    scores=raw/"scores"
    scores.mkdir(parents=True,exist_ok=True)
    if action=="report":
        report(stage,plan,raw,cases)
        return
    name="tco-zero-"+uuid.uuid4().hex[:12]
    receipt=raw/name
    receipt.mkdir()
    cmd=docker_base(name)+mount(plan_path,"/plan.json")
    cmd[cmd.index("--label")+1]="tco.study="+STUDY
    if action=="build":
        require(stage=="retrospective","Fresh zero predictions come from the sealed model worker")
        out.mkdir(exist_ok=True)
        cmd+=mount(out,"/outputs",True)
        for i,j in enumerate(plan["jobs"]):
            cmd+=mount(cases[j["id"]],f"/archive/{i}.npz")+mount(PREP/"model"/(j["slug"]+".npz"),f"/rgb/{i}.npz")
    else:
        cmd+=mount(PREP/"model","/prepared/model")+mount(PREP/"reference","/prepared/reference")+mount(scores,"/scores",True)
        for i,j in enumerate(plan["jobs"]):
            if j["source_status"]=="completed":
                cmd+=mount(out/j["id"] if stage=="retrospective" else cases[j["id"]],f"/cases/{i}")
    cmd += [IMAGE,"python","-B","-u","scripts/tco_two_track_zero.py",action,stage,"--inside"]
    cid,code=None,None
    with host_lock():
        begin=time.monotonic()
        start_utc=utc()
        seal(receipt/"start.json",dict(action=action,stage=stage,start_utc=start_utc,plan_sha256=sha(plan_path),
                                      image=IMAGE,rental_compute_usd=0))
        try:
            cid=subprocess.check_output(cmd,text=True,timeout=45).strip()
            cfg,=docker_json(["inspect",cid])
            mounts={v["Destination"] for v in cfg["Mounts"]}
            require(cfg["Image"]==IMAGE and not cfg["HostConfig"].get("DeviceRequests")
                    and cfg["HostConfig"]["NetworkMode"]=="none" and ("/prepared/reference" in mounts)==(action!="build"),
                    "Wrong projection/scoring isolation")
            with (receipt/"stdout.log").open("xb") as stream:
                code=subprocess.run(DOCKER+["start","--attach",cid],stdout=stream,stderr=subprocess.STDOUT,timeout=3600).returncode
            require(code==0,"Inspect "+str(receipt/"stdout.log"))
        finally:
            if cid:
                stopped,_=ensure_stopped(cid)
                require(stopped and not stopped["State"]["Running"],"Zero control worker still active")
                subprocess.run(DOCKER+["rm",cid],check=True,timeout=30,capture_output=True)
            seal(receipt/"terminal.json",dict(action=action,plan_sha256=sha(plan_path),exit_code=code,
                 start_utc=start_utc,end_utc=utc(),elapsed_seconds=time.monotonic()-begin,image=IMAGE,rental_compute_usd=0))


def report(stage,plan,raw,cases):
    rows=[]
    first_room=min(j["room"] for j in plan["jobs"])
    for j in plan["jobs"]:
        for key in KEYS:
            row={k:j[k] for k in ("room","seed","input_sha256","reference_sha256","frames","dataset","steps")}
            row.update(arm="ZERO" if key=="canonical" else "ZERO_supplied",status=j["source_status"],id=j["id"]+"_"+key)
            if j["source_status"]=="completed":
                case=raw/"outputs"/j["id"] if stage=="retrospective" else cases[j["id"]]
                model=read(case/"complete.json")
                suffix=Path(j["id"])/(key+".json")
                record=raw/"scores/first"/suffix
                score=read(record)
                require(score["readout"]==key and score["model_receipt_sha256"]==sha(case/"complete.json") and
                        score["predictions_sha256"]==model["predictions_sha256"],"Incomplete zero-score binding")
                row.update({k:score[k] for k in ("accuracy_mm","completeness_mm","valid_points","scoring_seconds",
                           "model_receipt_sha256","predictions_sha256")})
                row["score_sha256"]=sha(record)
                if j["room"]==first_room:
                    other=raw/"scores/independent"/suffix
                    replay=read(other)
                    require(replay["readout"]==key and replay["first_score_sha256"]==sha(record),"Wrong independent readout")
                    row.update(independent_mm_difference=check_against(score,replay),rescore_sha256=sha(other))
                if stage=="retrospective":
                    row.update(archive_prediction_sha256=j["archive_prediction_sha256"],
                               provenance="historical zero forward, CPU derived prior projection",
                               projection_cpu_seconds=model["projection_cpu_seconds"] if key=="supplied_canonical" else 0)
                else:
                    supplied=key=="supplied_canonical"
                    raw_time=model["native_readout_seconds"]
                    prior_time=model["supplied_readout_seconds"]
                    offset=prior_time-raw_time if supplied else 0
                    require(0 <= raw_time < model["scene_seconds"] and prior_time > 0, "Missing timed readout")
                    row.update(scene_seconds=model["scene_seconds"]+offset,cold_seconds=model["cold_seconds"]+offset,
                               projection_gpu_seconds=prior_time if supplied else raw_time,
                               both_saved_readouts_seconds=model["scene_seconds"]+prior_time,
                               timing_basis="Shared model interval plus only the selected delivered readout",
                               peak_allocated_bytes=model["supplied_peak_allocated" if supplied else "peak_allocated"],
                               peak_reserved_bytes=model["supplied_peak_reserved" if supplied else "peak_reserved"])
            rows.append(row)
    seal(PUBLIC/("a_"+stage+"_zero_readouts.json"),dict(rows=rows,source_plan_sha256=sha(raw/"plan.json"),
         readouts_per_forward=2, independent_room=first_room,rental_compute_usd=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("build","first","independent","report"))
    p.add_argument("stage",choices=("retrospective","arkitscenes_fresh","hypersim_fresh"))
    p.add_argument("--inside",action="store_true")
    args=p.parse_args()
    if args.inside:
        plan=read("/plan.json")
        if args.action=="build":
            build_old(plan)
        else:
            score_inside(plan,args.action=="independent")
    else:
        launch(args.stage,args.action)
