"""Build and independently replay the final five-view LayerNorm research report on CPU."""
import argparse
from collections import Counter
from statistics import mean
import math

from replay_tco_final_ln import METRICS, analyze, require
from score_tco_final_ln import PROTOCOL, check_against
from tco_final_ln_protocol import IMAGE, LOCK, MANIFEST, PUBLIC, RAW, ROOT, SEEDS, STUDY, read, seal, sha


FOLLOW = ROOT / "manifests" / (STUDY + "_followup.json")
REPLICATION = ROOT / "manifests" / (STUDY + "_replication.json")
AUDIT = PUBLIC / "final_attempt_evidence.json"
VERDICT = PUBLIC / "final_verdict.json"
REPORT = PUBLIC / "README.md"
ALLOWED_RETRY = {"provisioning", "interrupted", "timeout"}
KNOWN_STATUS = ALLOWED_RETRY | {"completed", "oom", "error"}
RESCORE_TOLERANCE_MM = 1e-4


def by_id(rows):
    lookup = {r["id"]: r for r in rows}
    require(len(lookup) == len(rows), "Duplicate case identifier")
    return lookup


def prespecified_rescores(jobs):
    """Select before considering success; a failed selected slot remains a documented gap."""
    selected = {}
    for j in jobs:
        if j["seed"] == 42 and j["role"] in {"primary","development","replication","followup"}:
            key = (j["role"],j["environment"],j["arm"])
            choice = (j["room"],j["id"])
            selected[key] = min(selected.get(key,choice),choice)
    return {v[1] for v in selected.values()}


def raw_rescore_record(model_dir, scores_dir, j, evaluator_pins):
    """Bind independent score and original score to the actual saved model output."""
    uid=j["id"]
    first_path=scores_dir/"first"/uid/"score.json"
    independent_path=scores_dir/"independent"/uid/"score.json"
    first,second=read(first_path),read(independent_path)
    pred_sha=sha(model_dir/"prediction.npz")
    complete_sha=sha(model_dir/"complete.json")
    require(pred_sha==read(model_dir/"complete.json")["predictions_sha256"]==first["predictions_sha256"]
            and complete_sha==first["model_receipt_sha256"],"Saved prediction or model receipt changed for "+uid)
    first_sha=sha(first_path)
    require(first_sha==second["first_score_sha256"] and first["scoring_cpu"] and second["scoring_cpu"]
            and first["protocol"]==PROTOCOL and all(first["evaluator_sha256"].get(k)==v for k,v in evaluator_pins.items())
            and (first["room"],first["arm"],first["seed"])==(j["room"],j["arm"],j["seed"])
            and first["reference_sha256"]==j["reference_sha256"] and first["input_sha256"]==j["input_sha256"]
            and first["reference_file_sha256"]==j["reference_file_sha256"]
            and first["model_file_sha256"]==j["input_file_sha256"],"CPU scorer did not use the locked pair for "+uid)
    diffs=check_against(second,first)
    require(diffs==second["independent_rescore_mm_difference"],"Stored independent differences changed for "+uid)
    return dict(first_sha256=first_sha,independent_sha256=sha(independent_path),prediction_sha256=pred_sha,
                complete_sha256=complete_sha,first_mm={k:first[k] for k in diffs},
                independent_mm={k:second[k] for k in diffs},valid_points=first["valid_points"])


def check_public_rescores(jobs, records, attestations):
    expected=prespecified_rescores(jobs)
    target={k for k in expected if records[k]["status"]=="completed"}
    require(set(attestations)==target,"Missing or extra raw independent rescore attestations")
    max_diff=None
    for uid,entry in attestations.items():
        row=records[uid]
        require(row["score_sha256"]==entry["first_sha256"] and row["rescore_sha256"]==entry["independent_sha256"]
                and row["predictions_sha256"]==entry["prediction_sha256"]
                and row["model_receipt_sha256"]==entry["complete_sha256"] and row["valid_points"]==entry["valid_points"],
                "Public row differs from authenticated raw rescore "+uid)
        for k in ("accuracy_mm","completeness_mm"):
            diff=abs(entry["first_mm"][k]-entry["independent_mm"][k])
            require(row[k]==entry["first_mm"][k] and row["rescore_mm_differences"][k]==diff
                    and math.isfinite(diff) and diff<=RESCORE_TOLERANCE_MM,"Raw mm rescore mismatch for "+uid)
            max_diff=max(max_diff or 0.,diff)
    return dict(prespecified=len(expected),verified=len(target),missing=sorted(expected-target),
                tolerance_mm=RESCORE_TOLERANCE_MM,max_absolute_difference_mm=max_diff)


def check_development_diagnostics(document, jobs, scored, development):
    """Check the public per-step curves and paired endpoint deltas, including missing controls."""
    planned,diag=by_id(jobs),by_id(document["rows"])
    require(set(planned)==set(diag),"Development diagnostics did not cover the locked jobs")
    control={(r["room"],r["seed"],r["arm"]):r for r in development.values() if r["role"]=="development"}
    probes,low,missing,missing_controls={},[],[],[]
    vectors={f"aggregator.global_blocks.23.norm{i}.{k}" for i in (1,2) for k in ("weight","bias")}
    for uid,j in planned.items():
        row,rec=diag[uid],scored[uid]
        if rec["status"]!="completed":
            require(row.get("status")==rec["status"],"Missing diagnostic failure "+uid)
            missing.append(uid)
            continue
        require(all(row[k]==rec[k] for k in ("role","arm","seed","room"))
                and row["surface_accuracy_mm"]==rec["accuracy_mm"]
                and row["surface_completeness_mm"]==rec["completeness_mm"],"Wrong model score for diagnostic "+uid)
        steps=j["steps"]
        require(len(row["loss_by_step"])==steps and all(math.isfinite(x) for x in row["loss_by_step"])
                and set(row["preclip_l2"])=={"lora","ln"},"Missing per-update loss or clips "+uid)
        for role, values in row["preclip_l2"].items():
            require(len(values)==steps and all(math.isfinite(v) and v>=0 for v in values)
                    and row["times_clipped"][role]==sum(v>1. for v in values),"Invalid clipping series "+uid)
        require(set(row["comparisons"])=={"L10","LN10"},"Missing development endpoint comparison "+uid)
        for arm, pair in row["comparisons"].items():
            ref=control[(j["room"],j["seed"],arm)]
            if ref["status"]!="completed":
                require(pair=={"status":"missing_development_control"},"Wrong missing control indication "+uid)
                missing_controls.append(dict(id=uid,control=arm))
                continue
            require(pair["accuracy_delta_mm"]==rec["accuracy_mm"]-ref["accuracy_mm"]
                    and pair["completeness_delta_mm"]==rec["completeness_mm"]-ref["completeness_mm"],
                    "Diagnostic surface deltas disagree with the matched scores "+uid)
            require(pair["positive_pixels"]>0 and pair["depth_scale_ratio"]>0
                    and all(math.isfinite(v) for v in pair.values()),"Invalid depth/camera diagnostic "+uid)
        if j["role"]=="diagnostics":
            disp=row["ln_displacement_from_initial_by_step"]
            require(row["diagnostic_only"] and len(disp)==steps and all(set(v)==vectors and
                    all(math.isfinite(x) and x>=0 for x in v.values()) for v in disp),"Missing LN displacement curves "+uid)
            require(all(x>0 for x in disp[-1].values()),"Selected pretrained parameters never moved "+uid)
            probes[j["room"]]=dict(id=uid,seed=j["seed"],updates=steps,loss_first=row["loss_by_step"][0],
                    loss_last=row["loss_by_step"][-1],ln_clips=row["times_clipped"]["ln"],
                    ln_joint_final_displacement=math.sqrt(sum(x*x for x in disp[-1].values())))
        else:
            require(not row["diagnostic_only"] and j["arm"]=="LN10_low","Unexpected learning-rate diagnostic "+uid)
            pair=row["comparisons"]["LN10"]
            if "accuracy_delta_mm" in pair:
                low.append(pair)
    mean_lower={k:mean(v[k] for v in low) for k in ("accuracy_delta_mm","completeness_delta_mm")} if low else None
    return dict(planned=len(jobs),completed=len(jobs)-len(missing),missing=sorted(missing),
                missing_controls=missing_controls,original_lr_probes=probes,
                matched_lower_lr_vs_original_pairs=len(low),mean_lower_lr_minus_original_mm=mean_lower)


def selected_branch(primary):
    decision = primary.get("decisions_on_complete_rooms", {})
    if primary["joint_verdict"] == "pass":
        return "all_pass"
    applicable = primary["formal_coverage"] and all(decision.get(k) == "pass" for k in
                    ("accuracy_superiority_vs_short", "completeness_vs_short"))
    return "equal_update_pass_only" if applicable else "other"


def candidate_to_replicate(branch, followup):
    if branch == "all_pass":
        require(set(followup["analyses"]) == {"LN10", "LN10_scale", "LN10_bias"}, "Missing development ablation")
        return "LN10", "L10"
    candidate, short = ("LN20", "L20") if branch == "equal_update_pass_only" else ("LN10_low", "L10")
    gate = followup["independent_replication_gate"]
    require(gate["candidate"] == candidate, "Wrong follow-up candidate")
    eligible = (gate["coverage_complete"] and gate.get("preregistered_development_rooms", False)
                and all(followup.get("point_gates", {}).get(k, False) for k in
                        ("better_accuracy_vs_short","accuracy_vs_full","completeness_vs_full","faster_vs_full")))
    require(gate["eligible"] == eligible, "Development promotion differs from observed gates")
    return (candidate, short) if eligible else None


def absolute(result, candidate, short, full):
    """Equal-weight mean of room-specific seed medians; excludes incomplete rooms just as the interval table does."""
    if not result.get("comparisons"):
        return {}
    table = {}
    for arm, reference, side in ((candidate, short, "candidate"), (short, short, "reference"), (full, full, "reference")):
        table[arm] = {m: mean(v[side] for v in result["comparisons"][reference][m]["rooms"].values()) for m in METRICS}
    return table


def checked_stage(stage, jobs, carry, candidate="LN10", short="L10", components=False):
    """Rebuild each published table from individual paired rows and the original job list."""
    plan = read(PUBLIC / (stage+"_analysis_plan.json"))
    rows = read(PUBLIC / (stage+"_rows.json"))["rows"]
    stored = read(PUBLIC / (stage+"_decision.json"))
    mapping = by_id(jobs)
    records = by_id(rows)
    require(set(records) == set(mapping) | set(carry) and not set(carry) & set(mapping),
            "Missing or additional slots in "+stage)
    require(all(records[identifier] == row for identifier, row in carry.items()),
            "Follow-up changed the archived controls")
    environments = {j["room"]:j["environment"] for j in jobs}
    require(plan["rooms"] == [dict(room=r,environment=e) for r,e in sorted(environments.items())]
            and plan["seeds"] == SEEDS, "Analysis roster or seed schedule changed")
    expected_rescores = prespecified_rescores(jobs)
    for key, job in mapping.items():
        row = records[key]
        require(all(row[k]==job[k] for k in ("id","room","environment","seed","role","arm","profile")),
                "Released slot differs from registered case "+key)
        if row["status"] == "completed":
            require(row["adam_updates"] == job["steps"] and row["valid_points"]>0
                    and 0<row["peak_allocated_bytes"]<=8960*2**20
                    and row["input_sha256"] == job["input_sha256"] and row["reference_sha256"] == job["reference_sha256"],
                    "Completed slot missing data/optimizer/allocator evidence "+key)
            if key in expected_rescores:
                require(len(row["rescore_sha256"])==64 and set(row["rescore_mm_differences"])=={"accuracy_mm","completeness_mm"}
                        and all(0<=x<=RESCORE_TOLERANCE_MM for x in row["rescore_mm_differences"].values()),
                        "Independent CPU rescore did not confirm "+key)
    if components:
        require(stored.get("branch")=="all_pass" and set(stored["analyses"])=={"LN10","LN10_bias","LN10_scale"},
                "Missing scale/bias component result")
        for arm, expected in stored["analyses"].items():
            result = analyze(plan, [r for r in rows if r["arm"] in (arm, "L10", "L40")], arm, "L10", "L40")
            require(expected == result, "Component decision failed replay "+arm)
    else:
        result = analyze(plan,rows,candidate,short,"L40")
        require(stored==result, stage+" numerical decision differs from independent replay")
    return stored, {i:records[i] for i in mapping}


def evidence_from_raw(jobs):
    """Public attestation of actual cost and terminal outcomes; images and arrays remain private."""
    expected = by_id(jobs)
    actual = {p.name for p in (RAW/"cases").iterdir() if p.is_dir()}
    require(actual == set(expected), "Private attempt inventory differs from the registered workflow")
    cases = {}
    rescored={}
    selected=prespecified_rescores(jobs)
    pins=read(LOCK)["pinned_files"]
    evaluator_pins={k:pins[k] for k in ("scripts/run_experiment.py","third_party/TCO/evaluation/mv_recon/utils.py")}
    for key, job in expected.items():
        folder = RAW/"cases"/key
        history = []
        for directory in sorted(folder.glob("attempt[0-9][0-9]")):
            t, s = read(directory/"terminal.json"), read(directory/"start.json")
            require(read(directory/"job.json")==job and t["id"]==key and s["job"]==key,
                    "Raw records identify another registered job")
            require(s["image"]==IMAGE and s["registration_sha256"]==sha(MANIFEST) and s["input_file_sha256"]==job["input_file_sha256"],
                    "Raw execution differs from locked runtime/input")
            if job["role"] in ("primary", "anchor"):
                require(s["primary_lock_sha256"]==sha(LOCK), "Primary job differs from fresh-room lock")
            if job["role"] == "replication":
                require(s["replication_lock_sha256"]==sha(REPLICATION), "Replication job differs from fresh-visit lock")
            complete_sha=t.get("complete_sha256")
            if t["status"]=="completed":
                require(complete_sha==sha(directory/"model/complete.json"),"Raw completed receipt changed")
                if key in selected:
                    rescored[key]=raw_rescore_record(directory/"model",RAW/"scores",job,evaluator_pins)
            history.append(dict(attempt=t["attempt"],status=t["status"],stopped=t["stopped"],terminal_sha256=sha(directory/"terminal.json"),
                                start_sha256=sha(directory/"start.json"),exit_code=t["exit_code"],wall_seconds=t["wall_seconds"],
                                complete_sha256=complete_sha,start_cost_usd=s["rental_compute_usd"],terminal_cost_usd=t["rental_compute_usd"]))
        cases[key]=history
    batches=[]
    for path in sorted((RAW/"score_batches").glob("*/terminal.json")):
        receipt=read(path)
        batches.append(dict(terminal_sha256=sha(path),exit_code=receipt["exit_code"],stopped=receipt["stopped"],
                            rental_compute_usd=receipt["rental_compute_usd"]))
    require(batches,"No local CPU scoring batch evidence")
    return dict(study=STUDY,registration_sha256=sha(MANIFEST),image=IMAGE,cases=cases,scoring_batches=batches,
                raw_rescore_attestations=rescored)


def validate_cost_and_coverage(jobs, records, audit, qualification):
    expected=by_id(jobs)
    require(set(audit["cases"])==set(expected) and audit["registration_sha256"]==sha(MANIFEST)
            and audit["image"]==IMAGE, "Public attempt ledger differs from registration")
    roles={}
    failures=[]
    status_counts=Counter()
    count=0
    total=0.
    for uid, job in expected.items():
        history=audit["cases"][uid]
        require(1<=len(history)<=2, "Missing slot or retry limit exceeded: "+uid)
        published = qualification["cases"][uid] if job["role"]=="engineering" else records[uid]
        listed=published["attempts"]
        require(len(listed)==len(history),"Attempt counts differ for "+uid)
        for i,(att,pub) in enumerate(zip(history,listed),1):
            require(att["attempt"]==i and att["status"] in KNOWN_STATUS and att["stopped"] is True
                    and att["start_cost_usd"]==att["terminal_cost_usd"]==0 and att["wall_seconds"]>=0,
                    "Attempt stop, cost, timing or status failed "+uid)
            require(all(att[k]==pub[k] for k in ("attempt","status","terminal_sha256")),
                    "Terminal receipt mismatch: "+uid)
            require(i==1 or history[i-2]["status"] in ALLOWED_RETRY,"Disallowed retry "+uid)
            if job["role"]!="engineering":
                require(all(att[k]==pub[k] for k in ("start_sha256","exit_code","wall_seconds")),"Attempt evidence mismatch")
            if att["status"]=="completed":
                require(i==len(history) and att["exit_code"]==0
                        and att["complete_sha256"]==published.get("model_receipt_sha256",published.get("complete_sha256")),
                        "Completed row lacks its matching model receipt")
            else:
                failures.append(dict(id=uid,attempt=i,status=att["status"],recovered=history[-1]["status"]=="completed"))
            status_counts[att["status"]]+=1
            count+=1
            total+=att["wall_seconds"]
        ok=history[-1]["status"]=="completed"
        if job["role"]!="engineering":
            require(published["status"]==history[-1]["status"],"Scored row differs from the terminal slot")
        role=roles.setdefault(job["role"],dict(expected=0,completed=0,first_try_completed=0,attempts=0))
        role["expected"]+=1
        role["completed"]+=int(ok)
        role["first_try_completed"]+=int(history[0]["status"]=="completed")
        role["attempts"]+=len(history)
    require(audit["scoring_batches"] and all(b["stopped"] and b["rental_compute_usd"]==0
            for b in audit["scoring_batches"]), "CPU scoring batch stop/cost mismatch")
    return dict(slots=len(jobs),completed=sum(r["completed"] for r in roles.values()),attempts=count,by_role=roles,
                status_counts=dict(status_counts),failures=failures,model_attempt_wall_hours=total/3600,
                scoring_batches=len(audit["scoring_batches"]),scoring_batch_errors=sum(b["exit_code"]!=0 for b in audit["scoring_batches"]),
                rental_compute_usd=0)


def same_seed_repeats(dev_rows, duplicate_jobs):
    """Compare only the prospectively duplicated development room, with original and duplicate kept separate."""
    metrics=("accuracy_mm","completeness_mm")
    require({(j["room"],j["seed"],j["arm"]) for j in duplicate_jobs}
            =={("chess/seq-03",42,a) for a in ("L10","LN10","L40")} and len(duplicate_jobs)==3,
            "Unexpected duplicate roster")
    originals={(r["room"],r["seed"],r["arm"]):r for r in dev_rows.values() if r["role"]=="development"}
    pairs={}
    for job in duplicate_jobs:
        arm=job["arm"]
        original=originals.get((job["room"],job["seed"],arm))
        duplicate=dev_rows.get(job["id"])
        statuses=[row["status"] if row else "missing" for row in (original,duplicate)]
        pair=dict(original_id=original["id"] if original else None, duplicate_id=job["id"],
                  original_status=statuses[0],duplicate_status=statuses[1],available=all(s=="completed" for s in statuses))
        if duplicate:
            require(duplicate["role"]=="duplicate" and all(duplicate[k]==job[k] for k in ("room","seed","arm","environment")),
                    "Registered duplicate case mismatch")
        if pair["available"]:
            require(all(original[k]==duplicate[k] for k in ("input_sha256","reference_sha256","valid_points","adam_updates","profile","environment"))
                    and all(duplicate[k]==job[k] for k in ("input_sha256","reference_sha256","profile"))
                    and duplicate["adam_updates"]==job["steps"] and duplicate["valid_points"]>0,
                    "Same-seed comparison does not use identical input, reference, support and updates")
            pair.update(valid_points=duplicate["valid_points"],same_prediction=original["predictions_sha256"]==duplicate["predictions_sha256"],
                        metrics={k:dict(original=original[k], duplicate=duplicate[k], delta_mm=duplicate[k]-original[k],
                                        percent_change=100*(duplicate[k]/original[k]-1)) for k in metrics})
        pairs[arm]=pair
    contrasts={}
    if all(pairs[arm]["available"] for arm in ("LN10","L10")):
        require(pairs["LN10"]["valid_points"]==pairs["L10"]["valid_points"], "Different supports across repeat arms")
        for k in metrics:
            values={role:100*(pairs["LN10"]["metrics"][k][role]/pairs["L10"]["metrics"][k][role]-1)
                    for role in ("original","duplicate")}
            contrasts[k]=dict(**values,ordering_reversed=values["original"]*values["duplicate"]<0)
    return dict(room="chess/seq-03",seed=42,matched_pairs=sum(p["available"] for p in pairs.values()),
                by_arm=pairs,ln10_vs_l10_percent=contrasts)


def compact(result):
    cand,short,full=result["arms"]
    comparisons={}
    for ref, vals in result["comparisons"].items():
        comparisons[ref]={k:{field:v[field] for field in ("ratio","percent_change","lower_95","upper_95","worst_room")}
                          for k,v in vals.items()}
    return dict(dataset=result["dataset"],arms=result["arms"],expected=result["expected"],scored=result["scored"],
                joint_verdict=result["joint_verdict"],formal_coverage=result["formal_coverage"],
                complete_rooms=result["complete_rooms"],decisions=result.get("decisions_on_complete_rooms",{}),
                comparisons=comparisons,mean_room_medians=absolute(result,cand,short,full))


def interpretation(primary, replication):
    if primary["joint_verdict"]=="pass":
        if replication and replication.get("joint_verdict")=="pass":
            return "LN10 satisfied all three decisions on the Stanford rooms and independent ARKitScenes visits."
        return "LN10 satisfied all three decisions on the Stanford rooms. Independent replication did not establish the same joint claim."
    if primary["joint_verdict"]=="supported_violation":
        return "The Stanford data support a violation of at least one registered requirement for LN10."
    if primary["joint_verdict"]=="insufficient_coverage":
        return "Stanford coverage was insufficient for the registered six-room claim."
    return "The Stanford results leave the joint ten-step LayerNorm claim inconclusive."


def collect(mode):
    initial,lock,follow=read(MANIFEST),read(LOCK),read(FOLLOW)
    qualification=read(PUBLIC/"qualification.json")
    require(qualification["passed"] and qualification["rental_compute_usd"]==0 and qualification["registration_sha256"]==sha(MANIFEST)
            and qualification["fixed"]["passed"] and qualification["cpu"]["passed"]
            and set(qualification["cases"])==set(by_id(initial["engineering"])), "Incomplete engineering qualification")
    require(lock["pinned_files"][MANIFEST.relative_to(ROOT).as_posix()]==sha(MANIFEST), "Stanford lock changed")
    dev_jobs=initial["development"]+initial["duplicates"]
    development,dev_rows=checked_stage("development",dev_jobs,{})
    primary_jobs=lock["primary"]+lock["anchor"]
    primary,primary_rows=checked_stage("primary",primary_jobs,{})
    require(follow["primary_decision_sha256"]==sha(PUBLIC/"primary_decision.json"), "Primary decision differs from the outcome-selection lock")
    branch=selected_branch(primary)
    require(follow["branch"]==branch and follow["arms"]==initial["followups"][branch],"Follow-up branch diverged")
    cand,short,controls={"all_pass":("LN10","L10",{"L10","LN10","L40"}),
                         "equal_update_pass_only":("LN20","L20",{"L40"}),"other":("LN10_low","L10",{"L10","L40"})}[branch]
    carry={k:r for k,r in dev_rows.items() if r["role"]=="development" and r["arm"] in controls}
    follow_jobs=follow["jobs"]+follow.get("diagnostics",[])
    followup,follow_rows=checked_stage("followup",follow_jobs,carry,cand,short,branch=="all_pass")
    diagnostics=None
    if branch=="other":
        diag=read(PUBLIC/"followup_development_diagnostics.json")
        diagnostics=check_development_diagnostics(diag,follow_jobs,follow_rows,dev_rows)
        if mode=="write":
            # Re-read saved prediction arrays and model optimizer histories only at private release.
            from release_tco_final_ln import ledger, negative_branch
            require(negative_branch(ledger(follow_jobs))==diag,"Real model diagnostics differ from raw predictions/receipts")
    choice=candidate_to_replicate(branch,followup)
    rep_jobs=[]
    rep_rows={}
    replication=None
    evidence=[MANIFEST,LOCK,FOLLOW,PUBLIC/"qualification.json",PUBLIC/"environment.json",PUBLIC/"scorer_bridge.json",
              PUBLIC/"stanford_inputs.json",PUBLIC/"arkit_training_input.json",PUBLIC/"final_archive_receipt.json"]
    stages=["development","primary","followup"]
    if choice and REPLICATION.is_file():
        rlock=read(REPLICATION)
        require(rlock["scientific_gate"]["candidate"]==choice[0] and rlock["arms"]==[choice[1],choice[0],"L40"]
                and len({r["environment"] for r in rlock["rooms"]})==6,"Changed independent cohort/candidate")
        rep_jobs=rlock["jobs"]
        replication,rep_rows=checked_stage("replication",rep_jobs,{},choice[0],choice[1])
        replication_status=replication["joint_verdict"]
        evidence.extend((REPLICATION,PUBLIC/"arkit_roster_intake.json"))
        stages.append("replication")
    elif choice:
        # A completed metadata intake can document that the six required visits could not be obtained.
        # Pending acquisition or a missing scoring batch is not a terminal research outcome.
        path=PUBLIC/"arkit_roster_intake.json"
        intake=read(path)
        require(intake.get("roster_exhausted") is True and intake.get("ranked_visit_count",0)>0
                and "selected" in intake and len(intake["selected"])<6 and isinstance(intake.get("missing_or_duplicate"),list),
                "Promoted arm has pending replication")
        replication_status="intake_unverified"
        evidence.append(path)
    else:
        require(not REPLICATION.exists() and not (PUBLIC/"replication_decision.json").exists(),"Unpromoted candidate has replication outputs")
        replication_status="not_promoted"
    if branch=="other":
        evidence.append(PUBLIC/"followup_development_diagnostics.json")
    for stage in stages:
        for suffix in ("rows","analysis_plan","decision"):
            evidence.append(PUBLIC/(stage+"_"+suffix+".json"))
    plans=initial["engineering"]+dev_jobs+primary_jobs+follow_jobs+rep_jobs
    records={**dev_rows,**primary_rows,**follow_rows,**rep_rows}
    archive=read(PUBLIC/"final_archive_receipt.json")
    require(archive["registration_sha256"]==sha(MANIFEST) and archive["snapshot"]=="final"
            and archive["two_disks_verified"] and archive["files"]>0 and archive["rental_compute_usd"]==0,
            "Missing verified final private evidence archive")
    require(read(PUBLIC/"environment.json")["runpod_usd"]==0, "Unexpected rental in environment receipt")
    audit=evidence_from_raw(plans) if mode=="write" else read(AUDIT)
    coverage=validate_cost_and_coverage(plans,records,audit,qualification)
    coverage["independent_raw_rescores"]=check_public_rescores(plans,records,audit["raw_rescore_attestations"])
    if mode=="write":
        seal(AUDIT,audit)
    evidence.append(AUDIT)
    result=dict(study=STUDY,primary=compact(primary),development=compact(development),branch=branch,
                followup={name:compact(analysis) for name,analysis in followup["analyses"].items()}
                         if branch=="all_pass" else compact(followup),
                replication=dict(status=replication_status,candidate=choice[0] if choice else None,
                                 result=compact(replication) if replication else None),
                coverage=coverage,development_diagnostics=diagnostics,
                same_seed_repeatability=same_seed_repeats(dev_rows,initial["duplicates"]),summary=interpretation(primary,replication),
                evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence})
    return result


def comparison_table(stage):
    if not stage["comparisons"]:
        return ["Incomplete paired coverage prevented a comparison.",""]
    labels={"accuracy_mm":"Accuracy error","completeness_mm":"Completeness error","scene_seconds":"Scene time",
            "peak_allocated_bytes":"Peak live GPU memory","peak_reserved_bytes":"Peak reserved GPU memory"}
    cand,short,full=stage["arms"]
    lines=["| Endpoint | "+cand+" vs "+short+" | Upper 95% bound | "+cand+" vs "+full+" | Upper 95% bound |",
           "|---|---:|---:|---:|---:|"]
    for k,label in labels.items():
        a,b=(stage["comparisons"][v][k] for v in (short,full))
        lines.append(f"| {label} | {a['percent_change']:+.2f}% | {100*(a['upper_95']-1):+.2f}% | "
                     f"{b['percent_change']:+.2f}% | {100*(b['upper_95']-1):+.2f}% |")
    lines+=["","Absolute values are equal-weight means of the three-seed median for each complete room.","",
            "| Arm | Accuracy (mm) | Completeness (mm) | Scene time (s) | Peak live (MiB) | Peak reserved (MiB) |",
            "|---|---:|---:|---:|---:|---:|"]
    for arm in (short,cand,full):
        x=stage["mean_room_medians"][arm]
        lines.append(f"| {arm} | {x['accuracy_mm']:.3f} | {x['completeness_mm']:.3f} | {x['scene_seconds']:.2f} | "
                     f"{x['peak_allocated_bytes']/2**20:,.1f} | {x['peak_reserved_bytes']/2**20:,.1f} |")
    return lines+[""]


def render(r):
    p=r["primary"]
    c=r["coverage"]
    lines=["# Five-view TCO with pretrained final-block LayerNorm adaptation","",r["summary"],"",
           "LN10 optimized 4,096 pretrained LayerNorm coefficients in the last global block alongside standard rank-4 LoRA. "
           "Each run started from a fresh VGGT checkpoint. L10 and L40 used standard LoRA for ten and forty updates. "
           "All three used five RGB views, camera checkpointing, joint RGB+ED rendering and an 8,960 MiB PyTorch budget. "
           "[Registered design](../../docs/TCO_FINAL_GLOBAL_LAYERNORM_HANDOFF.md) · [Weight and signed-gradient checks](qualification.json)","",
           "## Stanford fresh rooms","",
           f"The paired comparison scored {p['scored']} of {p['expected']} slots. Lower errors and shorter times are better. "
           "Percent changes give equal weight to rooms. The one-sided upper bounds resample rooms within the two enrolled "
           "environments and paired seeds within rooms. They describe variation in these environments. "
           "[Room and seed results](primary_decision.json)",""]
    lines+=comparison_table(p)
    if p["decisions"]:
        speed = 1/p["comparisons"]["L40"]["scene_seconds"]["ratio"]
        lines+=["The equal-room ratio of scene times corresponds to "
                f"**{speed:.2f}×** the forty-step processing speed. Both comparisons use full scene timing, including the final FP32 readout.",""]
        for label,key in (("Accuracy superiority over ten-step LoRA","accuracy_superiority_vs_short"),
                          ("Completeness within 3% of ten-step LoRA","completeness_vs_short"),
                          ("Accuracy within 1% of forty-step LoRA","accuracy_vs_full"),
                          ("Completeness within 3% of forty-step LoRA","completeness_vs_full"),
                          ("Faster scene reconstruction than forty-step LoRA","faster_vs_full")):
            lines.append(f"- {label} **{p['decisions'][key].replace('_',' ')}**.")
    repeats=r["same_seed_repeatability"]
    lines+=["","## Same-seed repeatability on Chess","",
            f"{repeats['matched_pairs']} of three prespecified duplicate pairs were scorable on Chess/seq-03 at seed 42. "
            "Each delta subtracts the original error from its duplicate in millimetres. Positive means worse on the repeat. "
            "[Development case scores](development_rows.json)","",
            "| Arm | Accuracy change | Completeness change |","|---|---:|---:|"]
    for arm in ("L10","LN10","L40"):
        repeat=repeats["by_arm"][arm]
        if not repeat["available"]:
            lines.append(f"| {arm} | Unavailable ({repeat['original_status']}, {repeat['duplicate_status']}) | — |")
            continue
        a,cerr=(repeat["metrics"][k] for k in ("accuracy_mm","completeness_mm"))
        lines.append(f"| {arm} | {a['delta_mm']:+.3f} mm ({a['percent_change']:+.2f}%) | "
                     f"{cerr['delta_mm']:+.3f} mm ({cerr['percent_change']:+.2f}%) |")
    reversed_metrics=[k.split("_")[0] for k,d in repeats["ln10_vs_l10_percent"].items() if d["ordering_reversed"]]
    lines.append("")
    if reversed_metrics:
        lines.append("The ordering of LN10 and L10 reversed on "+" and ".join(reversed_metrics)+" across the two executions.")
    lines.append("These pairs measure execution variability on one room and seed. The registered fresh-room estimates use their own original runs.")
    lines+=["","## Outcome-selected follow-up",""]
    if r["branch"]=="all_pass":
        lines+=["The development ablation separated the updated scale and bias vectors. The original LN10 candidate was selected "
                "for independent replication. [Complete component estimates](followup_decision.json)",""]
        for arm in ("LN10_scale","LN10_bias"):
            lines+=[f"### {arm}","",*comparison_table(r["followup"][arm])]
    else:
        key="a matched 20-update comparison" if r["branch"]=="equal_update_pass_only" else "a reduced LayerNorm learning rate"
        lines+=[f"The registered development follow-up tested {key}. [Follow-up decisions](followup_decision.json)","",
                *comparison_table(r["followup"])]
    if r["development_diagnostics"] is not None:
        diag=r["development_diagnostics"]
        lines+=["The original-rate development diagnostics measured the joint L2 displacement of all four LayerNorm "
                "vectors from their pretrained values after ten updates. The clip count is the number of updates whose "
                "LayerNorm gradient norm exceeded 1 before clipping. [Paired model diagnostics](followup_development_diagnostics.json)","",
                "| Room | Seed | Final joint displacement | LayerNorm clips / updates |",
                "|---|---:|---:|---:|"]
        for room,probe in sorted(diag["original_lr_probes"].items()):
            lines.append(f"| {room} | {probe['seed']} | {probe['ln_joint_final_displacement']:.6g} | "
                         f"{probe['ln_clips']} / {probe['updates']} |")
        lines.append("")
        delta=diag["mean_lower_lr_minus_original_mm"]
        if delta:
            lines += [f"Across {diag['matched_lower_lr_vs_original_pairs']} matched development room–seed pairs, lowering "
                      f"the LayerNorm learning rate changed mean surface accuracy error by {delta['accuracy_delta_mm']:+.4f} mm "
                      f"and completeness error by {delta['completeness_delta_mm']:+.4f} mm relative to the original rate. "
                      "These are arithmetic means of paired differences. The formal development gate uses per-room seed medians.",""]
        if diag["missing"] or diag["missing_controls"]:
            lines += [f"{len(diag['missing'])} diagnostic or sensitivity slots and {len(diag['missing_controls'])} matched "
                      "control comparisons were unavailable. The verdict records their IDs.",""]
    rep=r["replication"]
    if rep["result"]:
        lines+=["## Independent ARKitScenes visits","",
                f"The locked {rep['candidate']} candidate reached **{rep['status'].replace('_',' ')}** on ARKitScenes. "
                "[Independent paired decisions](replication_decision.json)","",*comparison_table(rep["result"])]
    elif rep["status"]=="not_promoted":
        lines+=["The locked development criterion for releasing an independent ARKitScenes candidate was missed. "
                "[Development promotion check](followup_decision.json)",""]
    else:
        lines+=["The acquisition audit documented insufficient eligible ARKitScenes visits for measuring the promoted candidate. "
                "[Intake evidence](arkit_roster_intake.json)",""]
    lines+=["## Execution and reproducibility","",
            f"On the local RTX 3080, {c['completed']}/{c['slots']} model slots completed across {c['attempts']} attempts, "
            f"including engineering controls and repeats. Cumulative model-attempt wall time was {c['model_attempt_wall_hours']:.2f} hours. "
            "RunPod compute cost was **USD 0**. [Attempt ledger](final_attempt_evidence.json) · [Archive receipt](final_archive_receipt.json)","",
            "| Stage role | Registered | Completed | First attempt completed | Attempts |","|---|---:|---:|---:|---:|"]
    for role,row in c["by_role"].items():
        lines.append(f"| {role} | {row['expected']} | {row['completed']} | {row['first_try_completed']} | {row['attempts']} |")
    checks=c["independent_raw_rescores"]
    lines += ["",f"Independent CPU scoring reloaded the saved FP32 predictions for **{checks['verified']} of "
              f"{checks['prespecified']}** prespecified room–seed–arm slots."]
    if checks["max_absolute_difference_mm"] is not None:
        lines[-1]+=f" The largest absolute score difference across both directed errors was {checks['max_absolute_difference_mm']:.3g} mm."
    if checks["missing"]:
        lines[-1]+=f" The selected slots missing predictions were {', '.join(checks['missing'])}."
    lines+=["","Replay the decisions on CPU from the repository root using the public files.","",
            "```powershell","python -B -S scripts/finalize_tco_final_ln.py verify","```","",
            "[Final verdict and evidence digests](final_verdict.json) · [Development](development_summary.md) · "
            "[Primary](primary_summary.md) · [Follow-up](followup_summary.md)",""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("write","verify"))
    args=p.parse_args()
    result=collect(args.action)
    document=render(result)
    if args.action=="write":
        seal(VERDICT,result)
        if REPORT.exists():
            require(REPORT.read_text(encoding="utf-8")==document,"Existing final report differs from replay")
        else:
            REPORT.write_bytes(document.encode("utf-8"))
    else:
        require(read(VERDICT)==result and REPORT.read_text(encoding="utf-8")==document,"Final verdict or report differs")
    print(result["summary"],result["coverage"]["completed"],"/",result["coverage"]["slots"],"slots")


if __name__=="__main__":
    main()
