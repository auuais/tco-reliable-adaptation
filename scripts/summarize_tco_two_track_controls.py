"""Replay the paired resource/quality controls, keeping all historical comparisons explicitly retrospective."""
import argparse
from collections import Counter, defaultdict

from replay_tco_two_track import ALL, A_CANDIDATES, GEOMETRY, a_analysis, classify, compare, validated_index
from tco_two_track_protocol import MANIFEST, NEW_SEEDS, PUBLIC, ROOT, SEEDS, STUDY, digest, read, seal

OLD=ROOT/"results/tco_one_percent_v1"
EXTRA=("cold_seconds",)


def historical_rows(file):
    record=read(file)
    converted=[]
    for row in record["rows"]:
        arm = ({"four":"Full"+str(row["steps"]), "zero":"ZERO", "cached12":"Cache"+str(row["steps"])}).get(row["arm"])
        if not arm:
            continue
        out={k:row[k] for k in ("id","input_sha256","reference_sha256","dataset","seed","frames","scene_seconds",
                                "steps","cold_seconds","output_sha256")}
        out.update(room=row["scene"],arm=arm,status="completed",accuracy_mm=1000*row["metrics"]["accuracy_mean_m"],
                   completeness_mm=1000*row["metrics"]["completeness_mean_m"], valid_points=row["metrics"]["valid_points"],
                   peak_allocated_bytes=row["peak_allocated"], peak_reserved_bytes=row["peak_reserved"])
        converted.append(out)
    return converted


def make_packets(rows):
    packets={}
    for row in rows:
        p=dict(scene=row["room"],input_sha256=row["input_sha256"],reference_sha256=row["reference_sha256"])
        if packets.setdefault(row["room"],p)!=p:
            raise ValueError("Unmatched input/reference within reported cohort")
    return [packets[k] for k in sorted(packets)]


def table(rows,packets,seeds,comparisons,metrics,which_single=()):
    wanted=set(a for pair in comparisons for a in pair)
    arm_map={a:[seeds[0]] if a in which_single else seeds for a in wanted}
    subset=[r for r in rows if r["arm"] in wanted]
    index,_=validated_index(subset,packets,arm_map,metrics)
    rooms=[p["scene"] for p in packets]
    out={}
    for a,b in comparisons:
        out[a+"/"+b]=compare(index,rooms,a,b,seeds,arm_map[b],metrics)
    return out


def capacity(rows):
    arms=defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)
    result={}
    for arm, rr in arms.items():
        clean=[r for r in rr if r.get("status","completed")=="completed"]
        result[arm]=dict(registered=len(rr),completed=len(clean),failures=dict(Counter(r["status"] for r in rr
            if r.get("status","completed")!="completed")), max_peak_allocated_bytes=max(
            (r["peak_allocated_bytes"] for r in clean if r.get("peak_allocated_bytes")), default=None),
            max_peak_reserved_bytes=max((r["peak_reserved_bytes"] for r in clean if r.get("peak_reserved_bytes")), default=None))
    return result


def capacity_advantage(rows,comparisons):
    index={(r["room"],r["seed"],r["arm"]):r["status"] for r in rows}
    return {a+"/"+b: [{"room":room,"seed":seed} for (room,seed,arm),status in sorted(index.items())
                      if a==arm and status=="completed" and index.get((room,seed,b))=="oom"] for a,b in comparisons}


def historical():
    groups={}
    for phase,file in (("development",OLD/"evidence.json"),("transfer",OLD/"confirmation_evidence.json")):
        rows=historical_rows(file)
        for ds in sorted({r["dataset"] for r in rows}):
            cohort=[r for r in rows if r["dataset"]==ds]
            arms={r["arm"] for r in cohort}
            references=[a for a in ("ZERO","Full4","Full5","Full10","Full40") if a in arms]
            targets=[a for a in ("Cache10","Full10","Full40") if a in arms]
            comparisons=[(a,b) for a in targets for b in references if a!=b]
            seeds=SEEDS if phase=="development" else NEW_SEEDS
            if sorted({r["seed"] for r in cohort}) != sorted(seeds):
                raise ValueError("Archived repeat seeds differ from phase registration")
            summary=table(cohort,make_packets(cohort),seeds,comparisons,ALL+EXTRA)
            groups[ds+"_"+phase]=dict(paired=summary,seeds=seeds,source_sha256=digest(file),coverage=capacity(cohort))
    seal(PUBLIC/"a_historical_baselines.json",dict(study=STUDY,groups=groups,cap_mib=8960,
         provenance="Within the archived one-percent study. Archived transfer results belong only to those recipes.",
         timer="Original synchronized scene and cold elapsed seconds; no current-machine timing claim."))


def current(stages):
    rows=[]
    sources=[]
    for stage in stages:
        path=PUBLIC/("a_"+stage+"_rows.json")
        src=read(path)
        if src["registration_sha256"]!=digest(MANIFEST):
            raise ValueError("Current evidence from another design")
        rows+=src["rows"]
        sources.append(dict(stage=stage,sha256=digest(path)))
    packets=read(MANIFEST)["design"]["rooms_dev"]
    arms={r["arm"] for r in rows}
    quality_arms=[a for a in (*A_CANDIDATES,"Cache10","Full10","Full40","OfflineOriginal") if a in arms]
    params=[("Refresh1","Refresh1Uncached"),("Narrow","NarrowUncached"),("Distilled","OfflineOriginal")]
    comparisons=[(a,b) for a in quality_arms for b in ("Full10","Full40","Cache10") if a!=b and b in arms]
    comparisons+= [(a,b) for a,b in params if a in arms and b in arms]
    fields=ALL+EXTRA if all("cold_seconds" in r for r in rows if r["status"]=="completed") else ALL
    same=table(rows,packets,SEEDS,comparisons,fields)
    old=historical_rows(OLD/"evidence.json")
    # All nine historical Full40 predictions on these exact inputs were independently
    # rescored by Track B. Keep this geometry-only context even if a current control
    # timed out; the same-stack timing and primary failure denominator stay separate.
    historical_controls=("ZERO","Full5","Full6","Full10","Full40")
    substituted=[{**r,"arm":"historical_"+r["arm"]} for r in old if r["arm"] in historical_controls]
    cross=[(a,"historical_"+b) for a in quality_arms for b in historical_controls]
    different=table(rows+substituted,packets,SEEDS,cross,GEOMETRY)
    prior_file=PUBLIC/"a_retrospective_zero_readouts.json"
    if prior_file.is_file():
        prior=[{**r,"arm":"historical_ZERO_supplied"} for r in read(prior_file)["rows"] if r["arm"]=="ZERO_supplied"]
        different.update(table(rows+prior,packets,SEEDS,[(a,"historical_ZERO_supplied") for a in quality_arms],
                               GEOMETRY,which_single=("historical_ZERO_supplied",)))
    scope=dict(Refresh1="uncached twin has identical stage and Adam schedule",
        Narrow="no full 10-update uncached evidence unless present in paired table; qualification alone tests two updates",
        Refresh2="no registered full matrix for an identically refreshing uncached twin",
        Distilled="same factor inventory and 24 offline updates as OfflineOriginal; compare against ordinary Cache10 too")
    nom=PUBLIC/"a_nomination.json"
    final=nom.is_file() and stages==read(nom)["stages"] and prior_file.is_file()
    label="a_comparator_summary.json" if final else "a_comparator_"+"_".join(stages)+("_with_prior" if prior_file.is_file() else "_raw")+".json"
    seal(PUBLIC/label,dict(study=STUDY,current_same_stack=same,
        current_vs_historical_geometry_only=different, parameter_matched_control_scope=scope,
        current_sources=sources, historical_development_sha256=digest(OLD/"evidence.json"),
        historical_supplied_readouts_sha256=digest(prior_file) if prior_file.is_file() else None,
        capacity=capacity(rows),candidate_completed_where_control_oom=capacity_advantage(rows,comparisons),
        cap_mib=8960, rental_compute_usd=0,
        limitations="Historical controls are reference/input-matched diagnostics. They cannot validate current resource gates or a frozen equal-time claim."))


def fresh(dataset):
    stage=dataset+"_fresh"
    source=PUBLIC/("a_"+stage+"_rows.json")
    cp=MANIFEST.with_name(STUDY+"_"+stage+"_cohort.json")
    plan=read(cp)
    zr=PUBLIC/("a_"+stage+"_zero_readouts.json")
    first=read(source)["rows"]
    extra=read(zr)["rows"]
    raw={r["room"]:r for r in first if r["arm"]=="ZERO"}
    for r in extra:
        if r["arm"]!="ZERO":
            continue
        actual=raw[r["room"]]
        if actual["status"]!=r["status"]:
            raise ValueError("Contradictory ZERO status")
        if actual["status"]=="completed":
            if (actual["predictions_sha256"]!=r["predictions_sha256"] or actual["valid_points"]!=r["valid_points"]
                    or any(abs(actual[k]-r[k])>1e-4 for k in GEOMETRY)):
                raise ValueError("Zero native output did not replay")
    merged=[r for r in first if r["arm"]!="ZERO"]+extra
    arm=plan["candidate"]
    stats=a_analysis(merged,plan["rooms"],NEW_SEEDS,[arm],plan["controls"],fresh=True,singleton_controls=plan["singleton_controls"])
    resource_rows=table(merged,plan["rooms"],NEW_SEEDS,[(arm,c) for c in plan["controls"]],EXTRA,
                        which_single=plan["singleton_controls"])
    matched=stats["candidates"][arm]["comparisons"][plan["time_control"]]
    criterion={m:(1.,True) for m in GEOMETRY}
    pareto=classify(matched,{**criterion,"scene_seconds":(1.,False)},True)
    affordable=read(PUBLIC/"a_time_match.json")["affordable"]
    positive=pareto["passed"] and affordable and stats["coverage"]["scored"]==stats["coverage"]["expected"]
    outcome=dict(analysis=stats,comparable_time_control=plan["time_control"],paired_time_geometry=matched,
                 actual_time_ratio=matched["metrics"].get("scene_seconds"),closest_affordable_in_development=affordable,
                 per_metric_superiority=classify(matched,criterion,True),equal_budget_pareto=pareto,
                 equal_budget_pareto_supported=positive,cold_time=resource_rows,common_cap=capacity(merged),cap_mib=8960,
                 candidate_completed_where_control_oom=capacity_advantage(merged,[(arm,c) for c in plan["controls"]]),
                 cohort_sha256=digest(cp),rows_sha256=digest(source),zero_rows_sha256=digest(zr))
    seal(PUBLIC/("a_"+stage+"_comparator_decision.json"),outcome)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("historical","current","fresh"))
    p.add_argument("--stages",nargs="+")
    p.add_argument("--dataset",choices=("arkitscenes","hypersim"))
    args=p.parse_args()
    if args.action=="historical":
        historical()
    elif args.action=="current":
        current(args.stages or read(PUBLIC/"a_nomination.json")["stages"])
    elif args.dataset:
        fresh(args.dataset)
    else:
        p.error("fresh requires --dataset")
