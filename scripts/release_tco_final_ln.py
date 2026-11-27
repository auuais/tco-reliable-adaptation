"""Score a completed stage offline, verify archived evidence and replay the LayerNorm comparison."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time
import uuid

from replay_tco_final_ln import analyze, require
from score_tco_final_ln import check_against, check_optimizer
from tco_final_ln_protocol import IMAGE, LOCK, MANIFEST, PREP, PUBLIC, RAW, ROOT, SEEDS, STUDY, read, seal, sha


def jobs_for(stage):
    from run_tco_final_ln_local import select
    if stage == "development":
        return select(stage) + select("duplicates")
    if stage == "primary":
        return select(stage) + select("anchor")
    if stage == "followup":
        branch = read(ROOT/"manifests"/(STUDY+"_followup.json"))
        return select(stage) + branch.get("diagnostics",[])
    return select(stage)


def ledger(jobs):
    from run_tco_final_ln_local import attempts
    records = {}
    for job in jobs:
        identifier = job["id"]
        paths = attempts(identifier)
        require(1 <= len(paths) <= 2, f"Await all registered slots before scoring: {identifier}")
        history, success = [], None
        for i, path in enumerate(paths,1):
            require((path/"terminal.json").is_file(), f"Missing terminal result: {identifier} attempt {i}")
            start, term, written = (read(path/f) for f in ("start.json","terminal.json","job.json"))
            require(written == job and term["id"] == identifier and term["attempt"] == i
                    and term["stopped"] and term["rental_compute_usd"] == start["rental_compute_usd"] == 0
                    and start["image"] == IMAGE and start["input_file_sha256"] == job["input_file_sha256"]
                    and start["deadline_seconds"] == 1200 and start["registration_sha256"] == sha(MANIFEST),
                    f"Invalid start, terminal or job receipt {identifier}/{i}")
            if job["role"] in ("primary","anchor"):
                require(start["primary_lock_sha256"] == sha(LOCK), f"Different input lock {identifier}")
            if job["role"] == "replication":
                from tco_final_ln_protocol import STUDY, ROOT
                require(start["replication_lock_sha256"] == sha(ROOT/"manifests"/(STUDY+"_replication.json")),
                        f"Different replication lock {identifier}")
            if i > 1:
                require(history[-1]["status"] in {"provisioning","interrupted","timeout"}, "Unregistered retry")
            history.append(dict(attempt=i, status=term["status"], exit_code=term["exit_code"],
                                wall_seconds=term["wall_seconds"], start_sha256=sha(path/"start.json"),
                                terminal_sha256=sha(path/"terminal.json"), stdout_sha256=(
                                    sha(path/"stdout.log") if (path/"stdout.log").exists() else None)))
            if term["status"] == "completed":
                require(term["exit_code"] == 0 and success is None, "Repeated or unsuccessful completed job")
                case = path/"model"
                result = read(case/"complete.json")
                require(term["complete_sha256"] == sha(case/"complete.json")
                        and all(result.get(k) == v for k,v in job.items())
                        and result["predictions_sha256"] == sha(case/"prediction.npz")
                        and result["input_sha256"] == job["input_sha256"]
                        and result["rental_compute_usd"] == 0 and result["cap_mib"] == 8960,
                        f"Changed model output/registration {identifier}")
                check_optimizer(result)
                success = dict(path=str(case.relative_to(RAW).as_posix()),
                               complete_sha256=term["complete_sha256"], predictions_sha256=result["predictions_sha256"])
        require(success is None or history[-1]["status"] == "completed", "Attempt after success")
        records[identifier] = dict(job=job, attempts=history, success=success)
    return records


def repeat_ids(records):
    """First lexicographic room, seed 42, every main arm in each environment."""
    rooms = defaultdict(list)
    for rec in records.values():
        j = rec["job"]
        if j["role"] in ("primary","development","replication","followup") and j["seed"] == SEEDS[0]:
            rooms[(j["environment"], j["arm"])].append((j["room"], j["id"]))
    return {min(options)[1] for options in rooms.values()}


def projected_jobs(records, independent):
    selected = repeat_ids(records) if independent else set(records)
    return [dict(job=v["job"],case=v["success"]["path"]) for key,v in records.items()
            if key in selected and v["success"] is not None]


def inside(index, independent):
    import score_tco_final_ln as scorer
    cases = read(index)
    destination = Path("/scores") / ("independent" if independent else "first")
    for item in cases:
        job, case = item["job"], Path("/evidence") / item["case"]
        uid = job["id"]
        target = destination/uid
        if (target/"score.json").is_file():
            print("Preserving sealed score",uid,flush=True)
            continue
        args = ["--case",str(case),"--model",f"/prepared/model/{job['slug']}.npz",
                "--reference",f"/prepared/reference/{job['slug']}.npz","--input-file-sha",job["input_file_sha256"],
                "--reference-file-sha",job["reference_file_sha256"],"--out",str(target)]
        if independent:
            args.extend(["--against",f"/scores/first/{uid}/score.json"])
        scorer.main(args)


def cpu_run(stage, records, independent):
    from run_tco_final_ln_local import docker_base, mount, utc
    from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
    tasks = projected_jobs(records,independent)
    if not tasks:
        return
    kind = "independent" if independent else "first"
    scores = RAW / "scores"
    if all((scores/kind/i["job"]["id"]/"score.json").exists() for i in tasks):
        return
    scores.mkdir(exist_ok=True)
    schedules = RAW/"score_batches"
    jobs = schedules/(stage+"_"+kind+"_jobs.json")
    seal(jobs,tasks)
    name = "tco-ln-score-" + uuid.uuid4().hex[:15]
    base = schedules/name
    base.mkdir(exist_ok=False)
    cmd = docker_base(name) + mount(PREP,"/prepared") + mount(RAW,"/evidence") + mount(scores,"/scores",True)
    cmd.extend([IMAGE,"python","-B","-u","scripts/release_tco_final_ln.py","inside",
                "--index","/evidence/score_batches/"+jobs.name])
    if independent:
        cmd.append("--independent")
    cid = None
    code = None
    t0 = time.monotonic()
    seal(base/"start.json",dict(stage=stage,kind=kind,jobs_sha256=sha(jobs),utc=utc(),image=IMAGE,gpu=False))
    try:
        cid = subprocess.check_output(cmd,text=True,timeout=45).strip()
        info, = docker_json(["inspect",cid])
        require(info["Image"] == IMAGE and not info["HostConfig"].get("DeviceRequests")
                and info["HostConfig"]["NetworkMode"] == "none","Scorer container isolation failed")
        with (base/"stdout.log").open("xb") as log:
            code = subprocess.run(DOCKER+["start","--attach",cid],stdout=log,stderr=subprocess.STDOUT,
                                  timeout=max(300,240*len(tasks))).returncode
        require(code == 0,f"CPU scorer stopped, see {base}/stdout.log")
    finally:
        stopped = False
        if cid:
            state, _ = ensure_stopped(cid)
            require(state and not state["State"]["Running"], "Could not verify CPU scorer stop")
            stopped = True
            subprocess.run(DOCKER+["rm",cid],check=True,capture_output=True,timeout=30)
        seal(base/"terminal.json",dict(exit_code=code,stopped=stopped,elapsed_seconds=time.monotonic()-t0,
                                       rental_compute_usd=0))


def rows_from(records):
    repeats = repeat_ids(records)
    rows = []
    arrivals = {}
    for identifier, rec in records.items():
        job = rec["job"]
        row = {k:job[k] for k in ("id","room","environment","seed","arm","role","profile")}
        row.update(status=rec["attempts"][-1]["status"],attempts=rec["attempts"])
        if rec["success"] is not None:
            src = RAW/rec["success"]["path"]
            result = read(src/"complete.json")
            path = RAW/"scores/first"/identifier/"score.json"
            score = read(path)
            require(score["model_receipt_sha256"] == rec["success"]["complete_sha256"]
                    and score["predictions_sha256"] == rec["success"]["predictions_sha256"]
                    and score["input_sha256"] == job["input_sha256"] and score["reference_sha256"] == job["reference_sha256"]
                    and score["model_file_sha256"] == job["input_file_sha256"]
                    and score["reference_file_sha256"] == job["reference_file_sha256"]
                    and score["arm"] == job["arm"] and score["room"] == job["room"]
                    and score["seed"] == job["seed"] and score["scoring_cpu"],"Score provenance mismatch: "+identifier)
            if identifier in repeats:
                repeat_path = RAW/"scores/independent"/identifier/"score.json"
                repeat = read(repeat_path)
                require(repeat["first_score_sha256"] == sha(path),"Different first score used in rescore")
                row["rescore_mm_differences"] = check_against(repeat,score)
                row["rescore_sha256"] = sha(repeat_path)
            if job["steps"]:
                key = (job["room"],job["seed"])
                prefix = result["view_permutations"][:min(job["steps"],10)]
                require(key not in arrivals or arrivals[key] == prefix, "Different view schedule in matched cases")
                arrivals[key] = prefix
            row.update({k:score[k] for k in ("accuracy_mm","completeness_mm","valid_points","input_sha256",
                                               "reference_sha256","predictions_sha256","model_receipt_sha256")})
            row.update(score_sha256=sha(path),scene_seconds=result["scene_seconds"],
                       peak_allocated_bytes=result["peak_allocated"],peak_reserved_bytes=result["peak_reserved"],
                       adam_updates=result["completed_adam_updates"],optimizer_cuda_bytes=result["optimizer"]["cuda_state_bytes"],
                       cold_seconds=result["cold_seconds"],checkpoint_seconds=result["checkpoint_seconds"],
                       retries=result["allocation_retries"],oom_count=result["oom_count"])
        rows.append(row)
    return rows


def report(stage, rows, result):
    lines = [f"# Five-view LayerNorm study — {stage}", "",
             f"{result['scored']} of {result['expected']} comparison slots scored. Joint verdict: **{result['joint_verdict']}**.",
             "", "Per-room medians across three paired seeds, equal weight per room. Ratios below one favor the candidate.",
             "", "| Comparator | Endpoint | Change | One-sided 95% lower | One-sided 95% upper |",
             "|---|---|---:|---:|---:|"]
    for ref, comp in result["comparisons"].items():
        for metric in ("accuracy_mm","completeness_mm","scene_seconds","peak_allocated_bytes","peak_reserved_bytes"):
            v = comp[metric]
            values = [100*(v[k]-1) for k in ("ratio","lower_95","upper_95")]
            lines.append(f"| {ref} | {metric} | {values[0]:+.2f}% | {values[1]:+.2f}% | {values[2]:+.2f}% |")
    lines += ["","| Room | Arm | Accuracy (mm) | Completeness (mm) | Scene (s) | Live (MiB) | Reserved (MiB) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    from statistics import median
    grouped = defaultdict(list)
    for r in rows:
        if r["role"] not in {"duplicate","diagnostics"} and r["status"] == "completed":
            grouped[(r["room"],r["arm"])].append(r)
    for (room,arm), group in sorted(grouped.items()):
        values = [median(r[k] for r in group) for k in ("accuracy_mm","completeness_mm","scene_seconds",
                                                          "peak_allocated_bytes","peak_reserved_bytes")]
        lines.append(f"| {room} | {arm} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.2f} | "
                     f"{values[3]/2**20:.2f} | {values[4]/2**20:.2f} |")
    statuses=Counter(r["status"] for r in rows)
    lines += ["",f"Run coverage including anchors and duplicates: {dict(statuses)}. Rental compute: USD 0.",""]
    return "\n".join(lines)


def endpoint_deltas(path, native_path):
    """Compare camera trajectories in the first camera's frame and separate depth scale from shape."""
    import numpy as np
    def prediction(path):
        with np.load(path, allow_pickle=False) as data:
            return tuple(data[k][0].astype(np.float64) for k in ("depth","extrinsics","intrinsics"))
    da, Ta, Ka = prediction(path)
    db, Tb, Kb = prediction(native_path)
    use = (da>0)&(db>0)&np.isfinite(da)&np.isfinite(db)
    require(use.sum()>0 and Ta.shape==Tb.shape==(5,4,4), "Unusable depth/camera probe")
    # Native extrinsics map world to camera. Compare T_i T_0^-1 to remove the camera coordinate gauge.
    Pa,Pb = Ta@np.linalg.inv(Ta[0]),Tb@np.linalg.inv(Tb[0])
    rel=Pa[1:,:3,:3]@np.swapaxes(Pb[1:,:3,:3],-1,-2)
    rotation_deg=np.rad2deg(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
    ra,rb=np.median(da[use]),np.median(db[use])
    geom=dict(
        positive_pixels=int(use.sum()),depth_scale_ratio=float(ra/rb),
        mean_abs_depth_change_in_native_model_units=float(np.mean(np.abs(da[use]-db[use]))),
        mean_abs_change_after_independent_median_depth_normalization=float(np.mean(np.abs(da[use]/ra-db[use]/rb))),
        mean_camera0_relative_rotation_degrees=float(np.mean(rotation_deg)),
        mean_camera0_relative_translation_change_in_native_model_units=float(np.linalg.norm(Pa[1:,:3,3]-Pb[1:,:3,3],axis=1).mean()),
        mean_absolute_fx_change_px=float(np.abs(Ka[:,0,0]-Kb[:,0,0]).mean()),
        mean_absolute_fy_change_px=float(np.abs(Ka[:,1,1]-Kb[:,1,1]).mean()))
    require(all(np.isfinite(v) for v in geom.values()),"Nonfinite endpoint diagnostic")
    return geom


def negative_branch(records):
    from run_tco_final_ln_local import success
    jobs = [j for j in jobs_for("development") if j["role"]=="development" and j["arm"] in {"L10","LN10"}]
    controls={ (j["room"],j["seed"],j["arm"]):j for j in jobs}
    out=[]
    for uid,rec in records.items():
        j=rec["job"]
        if not rec["success"]:
            out.append(dict(id=uid,status=rec["attempts"][-1]["status"]))
            continue
        folder=RAW/rec["success"]["path"]
        result=read(folder/"complete.json")
        opts=result["optimizer"]
        comparisons={}
        scored=read(RAW/"scores/first"/uid/"score.json")
        for arm in ("L10","LN10"):
            ref=controls[(j["room"],j["seed"],arm)]
            path=success(ref)
            if path is None:
                comparisons[arm] = dict(status="missing_development_control")
                continue
            pred=path/"model/prediction.npz"
            baseline=read(RAW/"scores/first"/ref["id"]/"score.json")
            require(baseline["input_sha256"]==scored["input_sha256"]
                    and baseline["reference_sha256"]==scored["reference_sha256"]
                    and baseline["predictions_sha256"]==sha(pred),"Unmatched model diagnostic")
            comparisons[arm]=dict(endpoint_deltas(folder/"prediction.npz",pred),
                                  accuracy_delta_mm=scored["accuracy_mm"]-baseline["accuracy_mm"],
                                  completeness_delta_mm=scored["completeness_mm"]-baseline["completeness_mm"])
        out.append(dict(id=uid,room=j["room"],seed=j["seed"],arm=j["arm"],role=j["role"],
                        surface_accuracy_mm=scored["accuracy_mm"],surface_completeness_mm=scored["completeness_mm"],
                        diagnostic_only=j["role"]=="diagnostics",loss_by_step=result["losses"],
                        preclip_l2=opts["preclip_norms"],
                        times_clipped={k:sum(value>1.0 for value in v) for k,v in opts["preclip_norms"].items()},
                        ln_displacement_from_initial_by_step=opts.get("ln_delta_norms"),
                        comparisons=comparisons))
    return dict(reference="Matched clean L10 and original-LR LN10 at the same scene and seed",
                interpretation="Per-update displacements are available on the separate untimed diagnostic cases. "
                               "Surface deltas are in mm; unmatched metric translation/depth values are native model units.",
                rows=out)


def release(stage, jobs, records):
    rows = rows_from(records)
    for r in rows:
        require(r["role"] in {"development","duplicate","primary","anchor","followup","replication","diagnostics"},
                "Unknown scientific result role")
    dataset = {"primary":"stanford12", "development":"development", "replication":"arkitscenes"}.get(stage,"development")
    arms = {"LN10":"L10","LN20":"L20","LN10_low":"L10"}
    candidate = next((a for a in ("LN20","LN10_low","LN10") if any(r["arm"] == a and
                     r["role"] != "diagnostics" for r in rows)),None)
    # Follow-up arms are compared to the previously locked development controls.
    component = stage == "followup" and candidate is None and set(r["arm"] for r in rows) == {"LN10_scale","LN10_bias"}
    if stage == "followup":
        dev = read(PUBLIC/"development_rows.json")["rows"]
        controls = {"L40"} if candidate == "LN20" else ({"L10","LN10","L40"} if component else {"L10","L40"})
        rows += [r for r in dev if r["role"] == "development" and r["arm"] in controls]
    room_map = {j["room"]:j["environment"] for j in jobs}
    plan = dict(dataset=dataset,seeds=SEEDS,rooms=[dict(room=r,environment=e) for r,e in sorted(room_map.items())])
    seal(PUBLIC/(stage+"_rows.json"),dict(rows=rows))
    seal(PUBLIC/(stage+"_analysis_plan.json"),plan)
    if stage=="followup" and candidate=="LN10_low":
        seal(PUBLIC/"followup_development_diagnostics.json",negative_branch(records))
    if component:
        reports = {}
        for arm in ("LN10_scale","LN10_bias","LN10"):
            jobs_rows = [r for r in rows if r["arm"] in (arm,"L10","L40")]
            reports[arm] = analyze(plan, jobs_rows, arm, "L10", "L40")
        seal(PUBLIC/"followup_decision.json",dict(dataset="development",branch="all_pass",analyses=reports,
              interpretation="Scale-only and bias-only development comparisons. Independently replicate original LN10."))
        text="\n".join(report(arm, [r for r in rows if r["arm"] in (arm,"L10","L40")], res)
                       for arm, res in reports.items())
        path=PUBLIC/"followup_summary.md"
        require(not path.exists() or path.read_text(encoding="utf8")==text,"Existing ablation summary differs")
        path.write_text(text,encoding="utf8")
        return
    if not candidate:
        raise ValueError("Expected a registered candidate or component experiment")
    result=analyze(plan, rows, candidate, arms[candidate],"L40")
    seal(PUBLIC/(stage+"_decision.json"),result)
    md=report(stage,rows,result)
    path=PUBLIC/(stage+"_summary.md")
    require(not path.exists() or path.read_text(encoding="utf8")==md,"Existing study report differs")
    path.write_text(md,encoding="utf8")
    print(stage,result["joint_verdict"],result["scored"],"/",result["expected"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("score","release","inside"))
    parser.add_argument("--stage",choices=("development","primary","followup","replication"))
    parser.add_argument("--index",type=Path)
    parser.add_argument("--independent",action="store_true")
    args=parser.parse_args()
    if args.action=="inside":
        inside(args.index,args.independent)
        return
    require(args.stage is not None,"A cohort stage is required")
    jobs=jobs_for(args.stage)
    records=ledger(jobs)
    if args.action=="score":
        from run_profiled_d1_local import host_lock
        with host_lock():
            cpu_run(args.stage,records,False)
            cpu_run(args.stage,records,True)
    release(args.stage,jobs,records)


if __name__=="__main__":
    main()
