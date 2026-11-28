"""Independent CPU analysis of equal-step TCO acceleration and post-adaptation fusion.

The room is the unit of inference. Fusion of three runs supplies one estimate per room.
"""
import argparse
from collections import Counter
from itertools import product
import math
from pathlib import Path
import random
from statistics import mean, median

from replay_tco_final_ln import grade, scalar_summary
from tco_two_track_protocol import ARMS, DEV, MANIFEST, NEW_SEEDS, PUBLIC, SEEDS, STRESS, read, seal


GEOMETRY = ("accuracy_mm", "completeness_mm")
RESOURCES = ("scene_seconds", "peak_allocated_bytes", "peak_reserved_bytes")
ALL = GEOMETRY + RESOURCES
STATUSES = {"completed", "oom", "timeout", "error", "interrupted", "provisioning", "nonfinite", "failed"}
A_CANDIDATES = ("Refresh1", "Refresh2", "Narrow", "Distilled")
A_GATES = {"accuracy_mm": (1.01, False), "completeness_mm": (1.03, False), "scene_seconds": (1., True)}
B_GATES = {"accuracy_mm": (1., True), "completeness_mm": (1.03, False)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def arm_steps(arm):
    if arm in ARMS:
        return ARMS[arm]
    if arm.startswith("Full") and arm[4:].isdigit():
        return int(arm[4:])
    if arm in {"ZERO", "PriorZERO"}:
        return 0
    return None


def validated_index(rows, packets, arms, metrics):
    """Each arm is mapped to its pre-assigned seed list, including [None] for a fused triplet."""
    idx = {}
    expected = {(p["scene"], a, s) for p in packets for a in arms for s in arms[a]}
    by_room = {p["scene"]: p for p in packets}
    require(len(by_room) == len(packets) and expected, "Duplicate or empty room plan")
    room_support = {}
    for r in rows:
        room, arm, seed = r["room"], r["arm"], r.get("seed")
        key = room, arm, seed
        require(key in expected and key not in idx, f"Unregistered or duplicate slot: {key}")
        state = r.get("status", "completed")
        require(state in STATUSES, f"Unknown slot state: {state}")
        for field in ("input_sha256", "reference_sha256"):
            require(r.get(field) == by_room[room][field], f"Different {field} in {key}")
        if state == "completed":
            for m in metrics:
                require(type(r.get(m)) in (float,int) and math.isfinite(r[m]) and r[m] > 0,
                        f"Invalid {m} in {key}")
            require(type(r.get("valid_points")) is int and r["valid_points"] > 0, f"No support in {key}")
            require(room_support.setdefault(room, r["valid_points"]) == r["valid_points"],
                    f"Changed evaluation support in {room}")
            steps = arm_steps(arm)
            if steps is not None:
                require(r.get("steps") == steps, f"Unexpected Adam update count in {key}")
        idx[key] = r
    return idx, expected


def clean_row(index, key):
    return key in index and index[key].get("status", "completed") == "completed"


def coverage(index, expected):
    return dict(expected=len(expected),scored=sum(clean_row(index,k) for k in expected),
                missing=[list(k) for k in sorted(expected-set(index),key=str)],
                failed=[dict(room=k[0],arm=k[1],seed=k[2],status=index[k]["status"])
                        for k in sorted(expected & set(index),key=str) if not clean_row(index,k)])


def aggregate(values, reducer):
    return tuple(reducer(v) for v in zip(*values))


def compare(index, rooms, candidate, reference, candidate_seeds, reference_seeds, metrics, unit="paired", draws=10000):
    """Paired seed medians or a single fused group against the mean single run.

    A deterministic singleton reference (e.g. zero-shot) is held fixed within a paired seed draw.
    For a fused group, only rooms are resampled: resampling its input runs cannot create another output.
    """
    require(unit in {"paired","triplet"}, "Invalid inference unit")
    require(len(candidate_seeds) == len(set(candidate_seeds)) and len(reference_seeds) == len(set(reference_seeds)),
            "Duplicated constituent seed")
    if unit == "triplet":
        require(candidate_seeds == [None] and len(reference_seeds) in (1,3), "Expected one fixed triplet per room")
    else:
        require(len(candidate_seeds) == 3 and (candidate_seeds == reference_seeds or len(reference_seeds)==1),
                "Three genuinely paired seeds required")
    pairs, medians, ratios = {}, {}, {}
    need = {(room,a,s) for room in rooms for a,seeds in ((candidate,candidate_seeds),(reference,reference_seeds)) for s in seeds}
    for room in rooms:
        if any(not clean_row(index,(room,a,s)) for a,seeds in ((candidate,candidate_seeds),(reference,reference_seeds)) for s in seeds):
            continue
        nums = [tuple(index[room,candidate,s][m] for m in metrics) for s in candidate_seeds]
        dens = [tuple(index[room,reference,s][m] for m in metrics) for s in reference_seeds]
        reducer = mean if unit == "triplet" else median
        c, r = aggregate(nums, reducer), aggregate(dens, reducer)
        medians[room] = dict(candidate=dict(zip(metrics,c)),reference=dict(zip(metrics,r)))
        ratios[room] = tuple(x / y for x, y in zip(c,r))
        if unit == "triplet":
            pairs[room] = [ratios[room]]
        else:
            pairs[room] = []
            for select in product(range(3),repeat=3):
                c = aggregate([nums[i] for i in select], median)
                r = aggregate([dens[i] for i in select], median) if len(dens)==3 else dens[0]
                pairs[room].append(tuple(x/y for x,y in zip(c,r)))
    out = dict(candidate=candidate,reference=reference,unit=unit,expected_slots=len(need),coverage_complete=len(pairs)==len(rooms),
               complete_rooms=list(pairs), missing_or_failed=[list(k) for k in sorted(need,key=str) if not clean_row(index,k)],
               per_room=medians,metrics={})
    if not pairs:
        return out
    rng = random.Random(20261128)
    complete = list(pairs)
    samples = [[] for _ in metrics]
    for _ in range(draws):
        selected = []
        for _ in complete:
            room = rng.choice(complete)
            selected.append(pairs[room][0] if unit=="triplet" else rng.choice(pairs[room]))
        for values, store in zip(zip(*selected),samples):
            store.append(math.fsum(values)/len(complete))
    for i,m in enumerate(metrics):
        room_ratios = {room:ratio[i] for room,ratio in ratios.items()}
        out["metrics"][m] = dict(scalar_summary(room_ratios,samples[i]),room_ratios=room_ratios,
            equal_room_candidate=mean(medians[room]["candidate"][m] for room in complete),
            equal_room_reference=mean(medians[room]["reference"][m] for room in complete))
    return out


def classify(comparison, gates, fresh):
    if not comparison["coverage_complete"] or (fresh and len(comparison["complete_rooms"]) < 6):
        return dict(grades={},passed=False,verdict="insufficient_coverage")
    grades = {}
    for m,(limit,strict) in gates.items():
        summary = comparison["metrics"][m]
        if fresh:
            grades[m] = grade(summary,limit,strict)
        else:
            x = summary["ratio"]
            passed = x < limit if strict else x <= limit
            grades[m] = "pass" if passed else "point_miss"
    verdict = ("pass" if all(v=="pass" for v in grades.values()) else
               ("supported_violation" if "supported_violation" in grades.values() else
                ("inconclusive" if fresh else "point_miss")))
    return dict(grades=grades,passed=verdict=="pass",verdict=verdict)


def a_analysis(rows, packets, seeds, candidates, controls, fresh=False, draws=10000, singleton_controls=()):
    require("Full10" in controls, "Missing equal-update reference in roster")
    require(set(candidates).issubset(A_CANDIDATES) and len(candidates)==len(set(candidates)), "Ineligible A candidate")
    require(not set(candidates) & set(controls), "Candidates and controls overlap")
    rooms = [p["scene"] for p in packets]
    arm_map = {a:([seeds[0]] if a in singleton_controls else seeds) for a in list(candidates)+list(controls)}
    require(len(rows)==len({(r["room"],r["arm"],r.get("seed")) for r in rows}),"Duplicate A run")
    index, expected = validated_index(rows, packets, arm_map, ALL)
    out = dict(coverage=coverage(index,expected),arms=arm_map,point_only=not fresh,candidates={})
    for candidate in candidates:
        tables = {}
        for control in controls:
            tables[control] = compare(index, rooms, candidate, control, seeds,arm_map[control],ALL,draws=draws)
        analysis = dict(equal_step=classify(tables["Full10"],A_GATES,fresh),comparisons=tables)
        if "Full40" in tables:
            full=tables["Full40"]
            analysis["full_length_quality"]=classify(full,{"accuracy_mm":(1.01,False),"completeness_mm":(1.03,False)},fresh)
            stretch={"scene_seconds":(.25,False),"peak_allocated_bytes":(.764,False)}
            analysis["stretch_point"]=classify(full,stretch,False)
            analysis["stretch_bounds"]=classify(full,stretch,True) if fresh else None
        out["candidates"][candidate]=analysis
    if fresh and out["coverage"]["scored"]!=out["coverage"]["expected"]:
        for analysis in out["candidates"].values():
            analysis["equal_step"]["passed"]=False
            analysis["equal_step"]["verdict"]="insufficient_coverage"
    return out


def next_hybrid(development):
    original=development["candidates"]["Refresh1"]
    full=original["comparisons"]["Full40"]
    equal=original["comparisons"]["Full10"]
    # Only a recorded hybrid OOM, not missing output or a transient provisioning failure, establishes no-fit.
    oom=any(x["status"]=="oom" and x["arm"]=="Refresh1" for x in development["coverage"]["failed"])
    if oom or (full["coverage_complete"] and full["metrics"]["peak_allocated_bytes"]["ratio"]>.764):
        return "Narrow"
    if (equal["coverage_complete"] and equal["metrics"]["scene_seconds"]["ratio"]<1.
            and not original["equal_step"]["passed"]):
        return "Refresh2"
    return None


def choose_a(development, stress=None):
    """Rank only scientific candidates with complete eligible point comparisons."""
    ranks=[]
    for arm,entry in development["candidates"].items():
        if not entry["equal_step"]["passed"]:
            continue
        if stress is not None and not stress["candidates"].get(arm,{}).get("equal_step",{}).get("passed"):
            continue
        metrics=entry["comparisons"]["Full10"]["metrics"]
        ranks.append((metrics["scene_seconds"]["ratio"], metrics["peak_allocated_bytes"]["ratio"], A_CANDIDATES.index(arm),arm))
    return min(ranks)[-1] if ranks else None


def b_pilot(rows, packets, seeds, draws=10000, sham_atol_mm=1e-4):
    rooms=[p["scene"] for p in packets]
    require(len(rooms)==3 and len(seeds)==3 and len(set(seeds))==3,"Expected three original development rooms and three input seeds")
    arm_map={"native":seeds,"reprojected":seeds,"triplicate_anchor":[seeds[0]],"fusion":[None]}
    index,expected=validated_index(rows,packets,arm_map,GEOMETRY)
    for room in rooms:
        k=room,"fusion",None
        if clean_row(index,k):
            require(index[k].get("seeds")==seeds,"Fusion constituents or anchor order changed")
    compare_to=lambda arm,s:compare(index,rooms,"fusion",arm,[None],s,GEOMETRY,unit="triplet",draws=draws)
    native=compare_to("native",seeds)
    reproj=compare_to("reprojected",seeds)
    anchor=compare_to("native",[seeds[0]])
    sham={}
    for room in rooms:
        k1,k2=(room,"triplicate_anchor",seeds[0]),(room,"reprojected",seeds[0])
        if clean_row(index,k1) and clean_row(index,k2):
            sham[room] = {m:abs(index[k1][m]-index[k2][m]) for m in GEOMETRY}
    coverage_report=coverage(index,expected)
    complete=coverage_report["expected"]==coverage_report["scored"]
    sham_pass=len(sham)==len(rooms) and all(n<=sham_atol_mm for pair in sham.values() for n in pair.values())
    wins=sum(v<1 for v in native["metrics"].get("accuracy_mm",{}).get("room_ratios",{}).values())
    gates = {
        "complete":complete, "sham_surface_verified":sham_pass, "native_accuracy_wins_at_least_two":wins>=2,
        "more_accurate_than_native":native["coverage_complete"] and native["metrics"]["accuracy_mm"]["ratio"]<1.,
        "more_accurate_than_reprojected":reproj["coverage_complete"] and reproj["metrics"]["accuracy_mm"]["ratio"]<1.,
        "native_completeness_allowed":native["coverage_complete"] and native["metrics"]["completeness_mm"]["ratio"]<=1.03}
    return dict(study="tco_two_track_v1",coverage=coverage_report,unit="one_triplet_per_room",sham_surface_error_differences=sham,
           rooms_with_lower_accuracy=wins,point_gates=gates,eligible=all(gates.values()),
           native=native,reprojected=reproj,anchor=anchor,independent_rescore_tolerance_mm=sham_atol_mm,
           interpretation="Descriptive three-room gate. Synthetic invariance, fixed output support and prediction provenance must pass qualification.")


def b_fresh(rows,packets,seeds,candidate,longer,grouped,draws=10000):
    rooms=[p["scene"] for p in packets]
    group=[None] if grouped else seeds
    long_seeds=[seeds[0]] if grouped else seeds
    arms={candidate:group,"Full40":seeds,longer:long_seeds}
    require(candidate not in ("Full40",longer),"Candidate cannot be native comparator")
    idx,expected=validated_index(rows,packets,arms,ALL)
    if grouped:
        for r in rows:
            if r["arm"]==candidate and r.get("status","completed")=="completed":
                require(r.get("seeds")==seeds,"Unexpected independent fusion triplet")
    tables={ref:compare(idx,rooms,candidate,ref,group,arms[ref],ALL,
                        "triplet" if grouped else "paired",draws) for ref in ("Full40",longer)}
    complete=coverage(idx,expected)
    check=classify(tables["Full40"],B_GATES,True)
    if complete["expected"]!=complete["scored"]:
        check.update(passed=False,verdict="insufficient_coverage")
    return dict(primary=check,coverage=complete,comparisons=tables,
                both_errors_superior=classify(tables["Full40"],{m:(1.,True) for m in GEOMETRY},True),
                comparable_compute=classify(tables[longer],B_GATES,True))


def from_files(stage,folder,cohort=None):
    reg=read(MANIFEST)
    if stage=="b-development":
        return b_pilot(read(folder/"b_development_rows.json")["rows"],reg["design"]["rooms_dev"],SEEDS)
    if stage in ("a-development","a-stress"):
        artifact=read(folder/(stage.replace("-","_")+"_rows.json"))
        rows=artifact["rows"]
        dev=stage=="a-development"
        packets=reg["design"]["rooms_dev" if dev else "rooms_stress"]
        seeds=SEEDS if dev else NEW_SEEDS
        if dev:
            candidates=["Refresh1"]+[arm for arm in A_CANDIDATES[1:] if any(r["arm"]==arm for r in rows)]
            controls=["Full10","Full40","Cache10","Refresh1Uncached"]
            controls.extend(a for a in ("OfflineOriginal","NarrowUncached") if any(r["arm"]==a for r in rows))
        else:
            plan=read(cohort) if cohort else artifact
            candidates=plan["candidates"]
            controls=plan["controls"]
        result=a_analysis(rows,packets,seeds,candidates,controls)
        if dev:
            result["bounded_alternative"]=next_hybrid(result)
        result["eligible_fastest"]=choose_a(result)
        return result
    require(cohort,"A prospective cohort plan is required for fresh outcomes")
    plan=read(cohort)
    require(len(plan["rooms"])==6 and len({x["scene"] for x in plan["rooms"]})==6 and plan["seeds"]==NEW_SEEDS,
            "Unexpected count of fresh visits/seeds")
    require(plan.get("dataset") in ("arkitscenes","hypersim"),"Fresh dataset name required")
    rows=read(folder/(stage.replace("-","_")+"_rows.json"))["rows"]
    if stage=="a-fresh":
        return a_analysis(rows,plan["rooms"],plan["seeds"],[plan["candidate"]],plan["controls"],True,
                          singleton_controls=plan.get("singleton_controls",()))
    return b_fresh(rows,plan["rooms"],plan["seeds"],plan["candidate"],plan["longer"],plan["triplet"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("b-development","a-development","a-stress","a-fresh","b-fresh"))
    p.add_argument("--folder",type=Path,default=PUBLIC)
    p.add_argument("--cohort",type=Path)
    args=p.parse_args()
    outcome=from_files(args.stage,args.folder,args.cohort)
    target=args.folder/(args.stage.replace("-","_")+"_independent.json")
    seal(target,outcome)
    print(target)


if __name__=="__main__":
    main()
