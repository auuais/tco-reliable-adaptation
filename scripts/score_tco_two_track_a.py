"""Score isolated, registered five-view outputs and publish paired case-level evidence."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

from score_tco_final_ln import PROTOCOL, REFERENCE_FILES, check_against, require
from tco_two_track_protocol import MANIFEST, PRIVATE, PUBLIC, ROOT, STUDY, digest, read, seal

PREP = Path("E:/ResearchData/TCO/preparation") / STUDY


def score(job, case, prepared, target, against=None, *, output_key="canonical"):
    import numpy as np
    import torch
    from run_experiment import evaluate
    from run_tco_courtyard_camera import seed_model
    from run_tco_frozen_prefix import signature

    require(not torch.cuda.is_available(), "CPU-only scoring required")
    model = prepared / "model" / (job["slug"] + ".npz")
    ref = prepared / "reference" / (job["slug"] + ".npz")
    report, prediction = case / "complete.json", case / "prediction.npz"
    receipt = read(report)
    require(digest(model) == job["input_file_sha256"] and digest(ref) == job["reference_file_sha256"],
            "Input or reference file changed")
    require(receipt["predictions_sha256"] == digest(prediction) and
            all(receipt[k] == job[k] for k in ("id","arm","room","seed","steps","input_file_sha256","input_sha256"))
            and receipt["completed_adam_updates"] == job["steps"] and receipt["rental_compute_usd"] == 0,
            "Unmatched model case or update count")
    with np.load(model, allow_pickle=False) as packet, np.load(ref, allow_pickle=False) as truth:
        require(set(packet.files) == {"images","extrs","intrs"} and set(truth.files) == {"pointclouds","valid_mask"},
                "Invalid isolated packets")
        images = packet["images"]
        input_hash = signature(dict(rgb=images, poses=packet["extrs"], intrinsics=packet["intrs"]))
        points, mask = truth["pointclouds"], truth["valid_mask"]
        ref_hash = signature(dict(surface=points, mask=mask))
    require(input_hash == job["input_sha256"] and ref_hash == job["reference_sha256"], "Decoded input/reference changed")
    with np.load(prediction, allow_pickle=False) as outputs:
        arrays = {k:outputs[k] for k in outputs.files}
    require(receipt["output_sha256"] == signature(arrays), "Prediction content differs from receipt")
    require(output_key == "canonical" or (output_key == "supplied_canonical" and job["arm"] == "ZERO"
            and job["steps"] == 0), "Readout is not registered for this case")
    canonical = arrays[output_key]
    require(images.shape[:2] == (5,3) and canonical.shape == points.shape == (5,*images.shape[2:],3)
            and canonical.dtype == np.float32 and np.isfinite(canonical).all(), "Invalid shared readout")
    require(mask.shape == points.shape[:-1] and mask.dtype == np.bool_ and np.isfinite(points[mask]).all()
            and all(view.any() for view in mask), "Invalid evaluation support")
    seed_model(job["seed"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        t0 = time.monotonic()
        metrics = evaluate(canonical, dict(images=torch.from_numpy(images), pointclouds=points, valid_mask=mask), Path(scratch))
        secs = time.monotonic()-t0
    result = dict(id=job["id"], room=job["room"], arm=job["arm"], seed=job["seed"], scoring_cpu=True, readout=output_key,
                  input_sha256=input_hash, model_file_sha256=digest(model), reference_sha256=ref_hash,
                  reference_file_sha256=digest(ref), predictions_sha256=receipt["predictions_sha256"],
                  model_receipt_sha256=digest(report), valid_points=metrics["valid_points"], protocol=metrics["protocol"],
                  evaluator_sha256={f:digest(ROOT/f) for f in REFERENCE_FILES}, scoring_seconds=secs,
                  accuracy_mm=1000*metrics["accuracy_mean_m"], completeness_mm=1000*metrics["completeness_mean_m"])
    require(result["protocol"] == PROTOCOL and result["valid_points"] == int(mask.sum()) and
            all(result[k] > 0 and math.isfinite(result[k]) for k in ("accuracy_mm","completeness_mm")), "Invalid score")
    if against:
        require(read(against)["readout"] == output_key, "Independent rescore used a different readout")
        result["first_score_sha256"] = digest(against)
        result["independent_rescore_mm_difference"] = check_against(result, read(against))
    seal(target, result)
    print(job["id"], "CPU rescore" if against else "CPU score", flush=True)


def locate(job):
    paths = sorted((PRIVATE/"cases"/job["id"]).glob("attempt[0-9][0-9]"))
    require(paths and all((p/"terminal.json").exists() for p in paths), "Case still pending: " + job["id"])
    terms = [read(p/"terminal.json") for p in paths]
    success = [p for p,t in zip(paths,terms) if t["status"] == "completed"]
    require(len(success) <= 1 and (not success or success[0] == paths[-1]), "Invalid retry history")
    for p,t in zip(paths,terms):
        start = read(p/"start.json")
        require(t["id"] == job["id"] and t["rental_compute_usd"] == 0 and t["stopped"]
                and read(p/"job.json") == job and start["registration_sha256"] == digest(MANIFEST)
                and start["input_file_sha256"] == job["input_file_sha256"], "Attempt ledger mismatch")
        if t["status"] == "completed":
            require(t["complete_sha256"] == digest(p/"model"/"complete.json"), "Terminal receipt changed")
            receipt = read(p/"model"/"complete.json")
            require(t["exit_code"] == 0 and receipt["cap_mib"] == 8960
                    and receipt["runtime"]["gpu"] == "NVIDIA GeForce RTX 3080"
                    and 0 < receipt["peak_allocated"] <= receipt["cap_bytes"]
                    and receipt["completed_adam_updates"] == len(receipt["active_counts"]) == job["steps"]
                    and all(receipt[k] == v for k,v in job.items()), "A different model policy completed this case")
    return paths, terms, success


def rescore_ids(jobs):
    groups = {}
    for job in jobs:
        key = job["dataset"], job["arm"]
        groups.setdefault(key, []).append((job["room"],job["seed"],job["id"]))
    return {min(group)[2] for group in groups.values()}


def inside(stage, independent):
    planned = read(Path("/index")/(stage+".json"))["jobs"]
    for task in planned:
        job, rel = task["job"], task["case"]
        target = Path("/scores") / ("independent" if independent else "first") / job["id"] / "score.json"
        if not target.exists():
            against = Path("/scores/first")/job["id"]/"score.json" if independent else None
            score(job, Path("/evidence")/rel, Path("/prepared"), target, against)


def launch_scores(stage, selected, repeat):
    from run_tco_final_ln_local import docker_base, mount
    from tco_final_ln_protocol import IMAGE
    from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
    from run_profiled_d1_local import host_lock

    kind = "independent" if repeat else "first"
    scores = PRIVATE/"scores"
    if all((scores/kind/t["job"]["id"]/"score.json").is_file() for t in selected):
        return
    scores.mkdir(parents=True, exist_ok=True)
    folder = PRIVATE/"score_batches"
    index = stage + "_" + kind
    seal(folder/(index+".json"), dict(jobs=selected))
    name = "tco-aa-score-" + uuid.uuid4().hex[:14]
    execution = folder/name
    execution.mkdir()
    cmd = docker_base(name) + mount(folder,"/index") + mount(PRIVATE/"cases","/evidence") + mount(PREP,"/prepared") + mount(scores,"/scores",True)
    cmd[cmd.index("--label")+1] = "tco.study="+STUDY
    cmd += [IMAGE,"python","-B","-u","scripts/score_tco_two_track_a.py",index,"--inside"]
    if repeat:
        cmd.append("--independent")
    cid, code = None, None
    with host_lock():
        begin = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        seal(execution/"start.json",dict(started=started,image=IMAGE,stage=stage,kind=kind,
                                        index_sha256=digest(folder/(index+".json")),rental_compute_usd=0))
        try:
            cid = subprocess.check_output(cmd,text=True,timeout=60).strip()
            cfg, = docker_json(["inspect",cid])
            require(cfg["Image"] == IMAGE and not cfg["HostConfig"].get("DeviceRequests") and
                    cfg["HostConfig"]["NetworkMode"] == "none", "Scoring isolation differs")
            with (execution/"stdout.log").open("xb") as log:
                code = subprocess.run(DOCKER+["start","--attach",cid],stdout=log,stderr=subprocess.STDOUT,
                                      timeout=180+300*len(selected)).returncode
            require(code == 0,"CPU evaluator failed: "+str(execution))
        finally:
            if cid:
                state,_ = ensure_stopped(cid)
                require(state and not state["State"]["Running"], "Scorer still running")
                subprocess.run(DOCKER+["rm",cid],check=True,capture_output=True,timeout=30)
            seal(execution/"terminal.json",dict(image=IMAGE,started=started,utc=datetime.now(timezone.utc).isoformat(),
                                               elapsed_seconds=time.monotonic()-begin,exit_code=code,
                                               index_sha256=digest(folder/(index+".json")),rental_compute_usd=0))


def release(stage):
    plan = MANIFEST.with_name(STUDY+"_"+stage+".json")
    roster = read(plan)
    require(roster["registration_sha256"] == digest(MANIFEST), "Different registered protocol")
    jobs = roster["jobs"]
    locations = {j["id"]:locate(j) for j in jobs}
    repeat = rescore_ids(jobs)
    tasks = [dict(job=j,case=locations[j["id"]][2][0].joinpath("model").relative_to(PRIVATE/"cases").as_posix())
             for j in jobs if locations[j["id"]][2]]
    launch_scores(stage,tasks,False)
    launch_scores(stage,[t for t in tasks if t["job"]["id"] in repeat],True)
    rows = []
    orders = {}
    for job in jobs:
        uid = job["id"]
        paths,terms,success = locations[uid]
        row = {k:job[k] for k in ("id","room","seed","arm","steps","input_sha256","reference_sha256","dataset")}
        row.update(status=terms[-1]["status"],attempts=[dict(status=t["status"],terminal_sha256=digest(p/"terminal.json"))
                   for p,t in zip(paths,terms)])
        if success:
            metrics = PRIVATE/"scores/first"/uid/"score.json"
            s, rec = read(metrics), read(success[0]/"model"/"complete.json")
            key = job["room"],job["seed"]
            count = min(10,job["steps"])
            order = rec["view_permutations"][:count]
            # Ordinary time-matched controls can have fewer updates than the other arms.
            previous = orders.get(key, [])
            common = min(len(previous),count)
            require(len(order) == count and previous[:common] == order[:common], "Paired view ordering differs")
            if count > len(previous):
                orders[key] = order
            require(s["id"] == uid and s["model_receipt_sha256"] == digest(success[0]/"model"/"complete.json")
                    and s["predictions_sha256"] == rec["predictions_sha256"] and
                    all(s[k] == job[k] for k in ("room","arm","seed","input_sha256","reference_sha256")),
                    "Existing score is inconsistent")
            for k in ("accuracy_mm","completeness_mm","valid_points","predictions_sha256","model_receipt_sha256","scoring_seconds"):
                row[k] = s[k]
            row.update(score_sha256=digest(metrics),scene_seconds=rec["scene_seconds"],
                       peak_allocated_bytes=rec["peak_allocated"],peak_reserved_bytes=rec["peak_reserved"],
                       completed_adam_updates=rec["completed_adam_updates"],cold_seconds=rec["cold_seconds"],
                       placement_seconds=rec["placement_seconds"],allocation_retries=rec["allocation_retries"],
                       model_oom_count=rec["oom_count"],adam_state_counts=rec["adam_state_counts"],
                       cache=rec["cache"],first_ten_view_permutations=order)
            if uid in repeat:
                other = PRIVATE/"scores/independent"/uid/"score.json"
                audit = read(other)
                require(audit["first_score_sha256"] == digest(metrics),"Mismatched independent rescore")
                row.update(rescore_mm_difference=check_against(audit,s),rescore_sha256=digest(other))
        rows.append(row)
    out = PUBLIC/("a_"+stage+"_rows.json")
    seal(out,dict(rows=rows,registration_sha256=digest(MANIFEST),plan_sha256=digest(plan),rental_compute_usd=0))
    print(out)


def wait_for_roster(stage):
    """Wait outside the workstation lock while the registered GPU launcher owns it."""
    from run_profiled_d1_local import host_lock
    plan = MANIFEST.with_name(STUDY+"_"+stage+".json")
    jobs = read(plan)["jobs"]
    pending_retry = {"timeout","interrupted","provisioning"}
    while True:
        for job in jobs:
            attempts = sorted((PRIVATE/"cases"/job["id"]).glob("attempt[0-9][0-9]"))
            if not attempts or not (attempts[-1]/"terminal.json").exists():
                break
            terminal = read(attempts[-1]/"terminal.json")
            if terminal["status"] in pending_retry and terminal["attempt"] < 2:
                break
        else:
            try:
                with host_lock():
                    return
            except OSError:
                pass
        time.sleep(15)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage")
    p.add_argument("--inside",action="store_true")
    p.add_argument("--independent",action="store_true")
    p.add_argument("--wait",action="store_true",help="Wait for every registered attempt before starting CPU scoring")
    args = p.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]+",args.stage),"Invalid stage")
    if args.inside:
        inside(args.stage,args.independent)
    else:
        if args.wait:
            wait_for_roster(args.stage)
        release(args.stage)
