"""Build and independently replay the final five-view LayerNorm research report on CPU."""
import argparse
from collections import Counter
from statistics import mean

from replay_tco_final_ln import METRICS, analyze, require
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
    independent = {}
    for j in jobs:
        if j["seed"] == 42 and j["role"] in ("primary","development","replication","followup"):
            key = j["environment"], j["arm"]
            independent[key] = min(independent.get(key,(j["room"], j["id"])), (j["room"],j["id"]))
    expected_rescores = {x[1] for x in independent.values()}
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
    return dict(study=STUDY,registration_sha256=sha(MANIFEST),image=IMAGE,cases=cases,scoring_batches=batches)


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
    if branch=="other":
        diag=read(PUBLIC/"followup_development_diagnostics.json")
        require(isinstance(diag["rows"],list),"Missing development diagnostics")
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
        require("selected" in intake and len(intake["selected"])<6 and intake["missing_or_duplicate"],
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
    if mode=="write":
        seal(AUDIT,audit)
    evidence.append(AUDIT)
    result=dict(study=STUDY,primary=compact(primary),development=compact(development),branch=branch,
                followup={name:compact(analysis) for name,analysis in followup["analyses"].items()}
                         if branch=="all_pass" else compact(followup),
                replication=dict(status=replication_status,candidate=choice[0] if choice else None,
                                 result=compact(replication) if replication else None),
                coverage=coverage,summary=interpretation(primary,replication),
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
