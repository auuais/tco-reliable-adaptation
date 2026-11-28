"""Describe allocator retries and matched schedule timing without changing any Track A decision."""
import argparse
from collections import Counter, defaultdict
import math
from statistics import mean, median

from tco_two_track_protocol import MANIFEST, PUBLIC, STUDY, digest, read, seal


def retry_class(row):
    if row["status"]!="completed":
        return "incomplete"
    count=row.get("allocation_retries")
    if count is None:
        return "missing_counter"
    if type(count)!=int or count<0:
        raise ValueError("Invalid CUDA allocator retry counter")
    return "zero_retries" if count==0 else "one_or_more_retries"


def distribution(rows):
    groups=("zero_retries","one_or_more_retries","missing_counter","incomplete")
    result={}
    for group in groups:
        selected=[r for r in rows if retry_class(r)==group]
        seconds=[r["scene_seconds"] for r in selected if r["status"]=="completed"]
        result[group]=dict(count=len(selected), median_seconds=median(seconds) if seconds else None,
            min_seconds=min(seconds,default=None),max_seconds=max(seconds,default=None))
    return result


def summarize_pair(rows, candidate, reference, twin):
    index={(r["room"],r["seed"],r["arm"]):r for r in rows}
    rooms=sorted({room for room,s,a in index if a==candidate} & {room for room,s,a in index if a==reference})
    all_medians,strict_medians={},{}
    point_rows=[]
    totals=Counter()
    for room in rooms:
        a_seeds={s for r,s,a in index if r==room and a==candidate}
        b_seeds={s for r,s,a in index if r==room and a==reference}
        expected=sorted(a_seeds | b_seeds)
        pairs=[]
        for seed in expected:
            a=index.get((room,seed,candidate)); b=index.get((room,seed,reference))
            values=dict(room=room,seed=seed)
            if not a or not b:
                group="not_jointly_registered"
            elif any(r["status"]!="completed" for r in (a,b)):
                group="incomplete"
            else:
                if a["input_sha256"]!=b["input_sha256"] or a["reference_sha256"]!=b["reference_sha256"]:
                    raise ValueError("Timing pairs use different data")
                ac,bc=retry_class(a),retry_class(b)
                group=("missing_counter" if "missing_counter" in (ac,bc) else
                       "both_zero" if ac==bc=="zero_retries" else "both_complete_at_least_one_retry")
                if twin:
                    if a["adam_state_counts"]!=b["adam_state_counts"]:
                        raise ValueError("Parameter schedule changed between matched cached/uncached controls")
                    # The public pilot publishes its actual per-step active inventory. Final
                    # score rows publish actual view sampling plus the final optimizer counters.
                    for k in ("active_counts","first_ten_view_permutations"):
                        if k in a and k in b and a[k]!=b[k]:
                            raise ValueError("Cached/uncached schedule mismatch: "+k)
                values.update(candidate_seconds=a["scene_seconds"],reference_seconds=b["scene_seconds"],
                              candidate_retries=a.get("allocation_retries"),reference_retries=b.get("allocation_retries"),
                              ratio=a["scene_seconds"]/b["scene_seconds"])
                pairs.append((a,b,group))
            totals[group]+=1
            point_rows.append(dict(**values,group=group))
        if len(pairs)==len(expected) and expected:
            ratio=median(a["scene_seconds"] for a,b,g in pairs)/median(b["scene_seconds"] for a,b,g in pairs)
            all_medians[room]=ratio
            if len(expected)==3 and all(g=="both_zero" for a,b,g in pairs):
                strict_medians[room]=ratio
    return dict(candidate=candidate,reference=reference,parameter_schedule_matched=twin,registered_pair_distribution=dict(totals),
        pairs=point_rows,all_completed_room_median_ratios=all_medians,
        all_completed_room_mean_ratio=mean(all_medians.values()) if all_medians else None,
        all_registered_rooms_and_seeds_complete=bool(rooms) and len(all_medians)==len(rooms),
        both_no_retry_full_three_seed_room_ratios=strict_medians,
        both_no_retry_full_three_seed_room_mean_ratio=mean(strict_medians.values()) if strict_medians else None)


def compact(rows):
    if len(rows)!=len({(r["room"],r["seed"],r["arm"]) for r in rows}):
        raise ValueError("Duplicate allocator diagnostic row")
    by_arm=defaultdict(list)
    cases=[]
    for r in rows:
        if r["status"]=="completed" and not (math.isfinite(r["scene_seconds"]) and r["scene_seconds"]>0):
            raise ValueError("Invalid synchronized time")
        by_arm[r["arm"]].append(r)
        row={k:r.get(k) for k in ("id","arm","room","seed","status","allocation_retries","scene_seconds","cold_seconds")}
        row.update(retry_class=retry_class(r),attempt_statuses=[a["status"] for a in r.get("attempts",[])])
        cases.append(row)
    tables={}
    for arm, group in sorted(by_arm.items()):
        rooms=sorted({r["room"] for r in group})
        tables[arm]=dict(all=distribution(group),rooms={room:distribution([r for r in group if r["room"]==room]) for room in rooms})
    twins={("Refresh1","Refresh1Uncached"),("Narrow","NarrowUncached")}
    possible={(a,b) for a in ("Refresh1","Refresh2","Narrow","Distilled","Cache10","OfflineOriginal")
              for b in ("Full10","Full40","Cache10") if a!=b}
    possible.update(twins|{("Distilled","OfflineOriginal")})
    pairs={a+"/"+b:summarize_pair(rows,a,b,(a,b) in twins) for a,b in sorted(possible) if a in by_arm and b in by_arm}
    return dict(cases=cases,by_arm_and_room=tables,matched_pairs=pairs,
        interpretation="All registered slots remain in the primary analysis. The both-zero diagnostic conditions on a measured consequence of the arm and cap. It is descriptive. A retry indicates allocator pressure; this observation does not establish the cause of a delay. Incomplete slots include final OOMs, timeouts and errors; missing telemetry is never treated as zero.")


def audit(stages,pilot):
    rows,sources=[],[]
    if pilot:
        path=PUBLIC/"a_resource_pilot.json"
        ledger=MANIFEST.with_name(STUDY+"_pilot.json")
        plan={j["id"]:j for j in read(ledger)["jobs"]}
        for raw in read(path)["cases"]:
            j=plan[raw["id"]]
            rows.append({**j,**raw,"status":"completed"})
        if len(rows)!=len(plan):
            raise ValueError("Pilot enrollment incomplete")
        sources=[dict(source=path.name,sha256=digest(path)),dict(source=ledger.name,sha256=digest(ledger))]
        name="pilot"
    else:
        for stage in stages:
            file=PUBLIC/("a_"+stage+"_rows.json")
            ledger=MANIFEST.with_name(STUDY+"_"+stage+".json")
            part=read(file)
            if part["plan_sha256"]!=digest(ledger) or {r["id"] for r in part["rows"]}!={j["id"] for j in read(ledger)["jobs"]}:
                raise ValueError("Changed or incomplete stage roster")
            rows+=part["rows"]
            sources.append(dict(source=file.name,sha256=digest(file)))
        name="_".join(stages)
    seal(PUBLIC/f"a_allocator_{name}.json",dict(study=STUDY,sources=sources,registration_sha256=digest(MANIFEST),
         pilot_one_seed=pilot,**compact(rows)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot",action="store_true")
    group.add_argument("--stages",nargs="+")
    args=p.parse_args()
    audit(args.stages,args.pilot)
