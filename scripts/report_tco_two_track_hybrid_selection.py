"""Replay hybrid selection from compact, registered room/seed scores. CPU-only standard library."""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, median
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_two_track_v1"
METRICS = ("accuracy_mm", "completeness_mm", "scene_seconds", "peak_allocated_bytes")
MARGINS = {"accuracy_mm": 1.01, "completeness_mm": 1.03}
PRIORITY = ("Refresh1", "Refresh2", "Narrow")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentiles(counts):
    """Rank-interpolated 5th/95th percentiles, counting multiplicities without expanding the sample."""
    pairs = sorted(counts.items())
    size = sum(counts.values())
    def pick(index):
        total = 0
        for x,n in pairs:
            total += n
            if index < total:
                return x
        raise ValueError("Bad percentile rank")
    def q(fraction):
        pos = (size-1)*fraction
        lo = math.floor(pos)
        return pick(lo)+(pos-lo)*(pick(math.ceil(pos))-pick(lo))
    require(size > 0, "No bootstrap outcomes")
    return [q(.05), q(.95)]


def seed_distribution(a, b):
    """Exact distribution of paired three-seed resampling of two separate medians."""
    require(len(a)==len(b)==3, "Expected three paired seeds")
    out = Counter()
    for draw in product(range(3), repeat=3):
        out[median(a[i] for i in draw)/median(b[i] for i in draw)] += 1
    return out


def room_distribution(distributions):
    """One draw selects a room uniformly and then draws its three paired seeds."""
    one = Counter()
    for d in distributions:
        require(sum(d.values())==27, "Incomplete seed resampling")
        one.update(d)
    n = len(distributions)
    require(1<=n<=3, "Selection replay covers two or three registered rooms")
    aggregate = Counter()
    for sample in product(one.items(), repeat=n):
        val = math.fsum(v for v,_ in sample)/n
        aggregate[val] += math.prod(w for _,w in sample)
    require(sum(aggregate.values())==(n*27)**n, "Incomplete hierarchical resampling")
    return aggregate


def compare(rows, rooms, seeds, arm, ref):
    units = []
    distributions = {m:[] for m in METRICS}
    for room in rooms:
        pairs = []
        missing = []
        for seed in seeds:
            a,b = (rows.get((room,seed,x)) for x in (arm,ref))
            if not all(r and r["status"]=="completed" for r in (a,b)):
                missing.append(dict(seed=seed,candidate=a["status"] if a else "unreported",
                                    reference=b["status"] if b else "unreported"))
                continue
            require(all(a[k]==b[k] for k in ("input_sha256","reference_sha256","valid_points","dataset")),
                    "Unequal surface support or paired inputs")
            require(a["first_ten_view_permutations"]==b["first_ten_view_permutations"],"Unpaired rendering sequence")
            pairs.append((a,b))
        detail = dict(room=room,paired_seeds=[a["seed"] for a,_ in pairs],missing_or_failed=missing,metrics={})
        units.append(detail)
        if len(pairs)!=3:
            continue
        detail["valid_points"]=pairs[0][0]["valid_points"]
        require(all(a["valid_points"]==detail["valid_points"] for a,_ in pairs),"Seed-dependent reference mask")
        for m in METRICS:
            left,right = zip(*[(a[m],b[m]) for a,b in pairs])
            c,r = median(left),median(right)
            dist=seed_distribution(left,right)
            stat=dict(candidate=c,reference=r,delta=c-r,ratio=c/r,
                      exact_seed_ratio_p05_p95=percentiles(dist))
            if m in MARGINS and ref=="Full10":
                stat["excess_error_over_margin_mm"]=c-r*MARGINS[m]
            detail["metrics"][m]=stat
            distributions[m].append(dist)
    complete = [r for r in units if r["metrics"]]
    aggregate = {}
    if complete:
        for m in METRICS:
            bands=room_distribution(distributions[m])
            s=[r["metrics"][m] for r in complete]
            aggregate[m]=dict(candidate=mean(v["candidate"] for v in s),reference=mean(v["reference"] for v in s),
                              mean_signed_difference=mean(v["delta"] for v in s),ratio=mean(v["ratio"] for v in s),
                              exact_hierarchical_ratio_p05_p95=percentiles(bands),
                              bootstrap_outcomes=sum(bands.values()))
    gate=None
    if ref=="Full10" and len(complete)==len(rooms):
        gate={m:aggregate[m]["ratio"]<=limit for m,limit in MARGINS.items()}
        gate["scene_seconds"]=aggregate["scene_seconds"]["ratio"]<1.
        gate["passed"]=all(gate.values())
    return dict(candidate=arm,reference=ref,rooms=units,complete_rooms=len(complete),expected_rooms=len(rooms),
                paired_slots=sum(len(r["paired_seeds"]) for r in units),expected_pairs=len(rooms)*len(seeds),
                aggregate=aggregate,gate=gate)


def load_rows(root, stage, expected, seeds, master, provenance):
    pub = root/"results"/STUDY
    mf=root/"manifests"/(STUDY+"_"+stage+".json")
    file=pub/("a_"+stage+"_rows.json")
    if not file.exists():
        return {}
    ledger,release=read(mf),read(file)
    require(ledger["registration_sha256"]==release["registration_sha256"]==master and release["plan_sha256"]==digest(mf)
            and release["rental_compute_usd"]==0, "Wrong study registration")
    jobs={j["id"]:j for j in ledger["jobs"]}
    require(len(jobs)==len(ledger["jobs"])==len(release["rows"]) and {r["id"] for r in release["rows"]}==set(jobs),
            "Published roster incomplete or duplicated")
    provenance[file.name]=digest(file)
    provenance[mf.name]=digest(mf)
    out={}
    for row in release["rows"]:
        job=jobs[row["id"]]
        require(all(row[k]==job[k] for k in ("room","dataset","seed","arm","steps","input_sha256","reference_sha256")),
                "Ledger/row mismatch")
        room,seed,arm=row["room"],row["seed"],row["arm"]
        require(room in expected and seed in seeds and
                all(row[k]==expected[room][k] for k in ("dataset","input_sha256","reference_sha256")), "Misassigned case")
        key=(room,seed,arm)
        require(key not in out,"Duplicate case")
        if row["status"]=="completed":
            require(row["valid_points"]>0 and row["completed_adam_updates"]==job["steps"]
                    and all(math.isfinite(row[m]) and row[m]>0 for m in METRICS),"Incomplete score/resource row")
        out[key]=row
    return out


def validate_claim(comp, claims):
    """Canonical Monte Carlo bounds and the exact finite-bootstrap percentiles are separate outputs."""
    reported=claims.get(comp["candidate"],{}).get("comparisons",{}).get(comp["reference"])
    if not reported:
        return
    require(reported["coverage_complete"]==(comp["complete_rooms"]==comp["expected_rooms"]), "Published coverage differs")
    for m,s in comp["aggregate"].items():
        theirs=reported["metrics"][m]
        require(math.isclose(s["ratio"],theirs["ratio"],rel_tol=1e-12),"Published aggregate ratio differs")
        for room in comp["rooms"]:
            if room["metrics"]:
                require(math.isclose(room["metrics"][m]["ratio"],theirs["room_ratios"][room["room"]],rel_tol=1e-12),
                        "Published room ratio differs")
        s["canonical_10000_draw_ratio_p05_p95"]=[theirs["lower_95"],theirs["upper_95"]]
    if comp["reference"]=="Full10":
        expected=claims[comp["candidate"]]["equal_step"]["passed"]
        require(bool(comp["gate"] and comp["gate"]["passed"])==expected,"Published gate differs")


def report(root, stage):
    pub=root/"results"/STUDY
    mf=root/"manifests"/(STUDY+".json")
    master=read(mf)
    sha=digest(mf)
    ng=pub/"a_alternative_gate.json"
    pick=pub/"a_hybrid_stress_entry.json"
    so=pub/"a_distillation_trigger.json"
    gate=read(ng)
    alternative=gate["candidate"]
    require(alternative in (None,"Refresh2","Narrow"),"Unexpected conditional hybrid")
    stress=stage=="stress"
    design=master["design"]
    expected={r["scene"]:r for r in design["rooms_stress" if stress else "rooms_dev"]}
    seeds=design["stress_seeds" if stress else "dev_seeds"]
    require(len(seeds)==len(set(seeds))==3,"Unexpected seed count")
    provenance={mf.name:sha,ng.name:digest(ng)}
    selection=read(pick) if pick.exists() else None
    if selection is not None:
        provenance[pick.name]=digest(pick)
    status="complete"
    claims={}
    if stress:
        candidate=selection["candidate"] if selection else None
        arms=[candidate] if candidate else []
        cohorts=["stress_hybrid"] if candidate else []
        if selection is None:
            status="awaiting development selection"
        elif candidate is None:
            status="no hybrid qualified for stress"
        elif not (pub/"a_stress_hybrid_rows.json").exists():
            status="awaiting stress scores"
        if so.exists():
            claims=(read(so)["stress_analysis"] or {}).get("candidates",{})
            provenance[so.name]=digest(so)
    else:
        arms=["Refresh1"]
        cohorts=["development"]
        if alternative:
            cohorts.append("alternative")
            if (pub/"a_alternative_rows.json").exists():
                arms.append(alternative)
            else:
                status="awaiting alternative scores"
        claims=selection["analysis"]["candidates"] if selection else gate["analysis"]["candidates"]
    rows={}
    for c in cohorts:
        extra=load_rows(root,c,expected,seeds,sha,provenance)
        require(rows.keys().isdisjoint(extra),"A stage reused a slot")
        rows.update(extra)
    comparisons=[]
    if rows:
        for a in arms:
            refs=["Full10","Cache10"]
            if a!="Refresh1" and not stress:
                refs.append("Refresh1")
            for ref in refs:
                z=compare(rows,list(expected),seeds,a,ref)
                validate_claim(z,claims)
                comparisons.append(z)
    ranks=[]
    for c in comparisons:
        if c["reference"]=="Full10" and c["gate"] and c["gate"]["passed"]:
            ranks.append((c["aggregate"]["scene_seconds"]["ratio"],c["aggregate"]["peak_allocated_bytes"]["ratio"],
                          PRIORITY.index(c["candidate"]),c["candidate"]))
    computed=min(ranks)[-1] if ranks and status=="complete" else None
    if not stress and selection and status=="complete":
        require(selection["candidate"]==computed,"Published stress entry differs from independent ranking")
    if stress and so.exists() and status=="complete":
        require(read(so)["stress_pass"]==bool(computed),"Published stress gate differs from the scored rows")
    note=("The stress design evaluates the selected hybrid against Full10 and Cache10 on Office and Courtyard. "
          "The other hybrid has no registered stress slots." if stress else
          "The development design pairs the alternative with Refresh1 on identical seeds.")
    return dict(study=STUDY,stage=stage,status=status,conditional_alternative=alternative,rooms=list(expected),seeds=seeds,
                earned_stress_candidate=selection["candidate"] if selection else None,
                passing_choice_for_this_stage=computed,comparisons=comparisons,source_sha256=provenance,scope=note)


def pct(x):
    return f"{100*(x-1):+.2f}%"


def bounds(b):
    return f"{pct(b[0])} to {pct(b[1])}"


def markdown(data):
    lines=[f"# Hybrid {data['stage']} selection", "", f"Status — {data['status']}.",
           "", f"The registered rooms are {', '.join(data['rooms'])}. The paired seeds are {', '.join(map(str,data['seeds']))}.",
           "",data["scope"],"",
           "Accuracy measures prediction-to-reference distance. Completeness measures reference-to-prediction distance. Positive error changes are worse. An error pair always gives accuracy first. Each arm's room estimate is its median across three seeds. The aggregate equally weights within-room ratios. A room with any missing seed pair is excluded from the estimate and prevents a selection pass.",
           "", "Selection requires accuracy error at most +1%, completeness error at most +3% and strictly shorter time than Full10. The reported ranges give the exact 5th and 95th percentiles from descriptive resampling of rooms and paired seeds. Development and stress use previously evaluated rooms.", ""]
    for c in data["comparisons"]:
        lines += [f"## {c['candidate']} versus {c['reference']}", "",
                  f"{c['paired_slots']} of {c['expected_pairs']} seed pairs and {c['complete_rooms']} of {c['expected_rooms']} rooms are complete.", "",
                  "| Room | Candidate errors (mm) | Reference errors (mm) | Change (mm) | Error change (%) | Candidate / reference time (s) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in c["rooms"]:
            s=r["metrics"]
            if not s:
                details=", ".join(f"seed {x['seed']} {x['candidate']}/{x['reference']}" for x in r["missing_or_failed"])
                lines.append(f"| {r['room']} | — | — | — | {details} | — |")
                continue
            a,b,t=(s[k] for k in METRICS[:3])
            lines.append(f"| {r['room']} | {a['candidate']:.3f} / {b['candidate']:.3f} | {a['reference']:.3f} / {b['reference']:.3f} | "
                         f"{a['delta']:+.3f} / {b['delta']:+.3f} | {pct(a['ratio'])} / {pct(b['ratio'])} | {t['candidate']:.2f} / {t['reference']:.2f} |")
        for r in c["rooms"]:
            if r["metrics"]:
                a,b=(r["metrics"][k] for k in METRICS[:2])
                sentence=f"{r['room']} used {r['valid_points']:,} evaluable points. Seed-only bounds were {bounds(a['exact_seed_ratio_p05_p95'])} for accuracy and {bounds(b['exact_seed_ratio_p05_p95'])} for completeness."
                if c["reference"]=="Full10":
                    sentence+=f" Signed distance from the room's +1% / +3% limits was {a['excess_error_over_margin_mm']:+.3f} / {b['excess_error_over_margin_mm']:+.3f} mm (positive exceeds the limit)."
                lines+=["",sentence]
        if c["aggregate"]:
            a,b,t,p=(c["aggregate"][k] for k in METRICS)
            lines+=["",f"Equal-room error changes were {pct(a['ratio'])} and {pct(b['ratio'])}. The hierarchical bounds were {bounds(a['exact_hierarchical_ratio_p05_p95'])} and {bounds(b['exact_hierarchical_ratio_p05_p95'])}, respectively. Speedup was {1/t['ratio']:.3f}× and peak allocated memory changed {pct(p['ratio'])}."]
        if c["reference"]=="Full10":
            gate=c["gate"]
            lines+=["","Full10 point gate — "+("insufficient coverage" if gate is None else "passed" if gate["passed"] else "missed") + "."]
        lines.append("")
    if data["status"]=="complete":
        lines += ["The selection rule retained "+(data["passing_choice_for_this_stage"] or "no hybrid")+".",""]
    return "\n".join(lines)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("alternative","stress"))
    parser.add_argument("--json",action="store_true",help="Include every denominator, distribution bound and source digest.")
    parser.add_argument("--out",type=Path,help="Create a new report. Existing files must be identical.")
    args=parser.parse_args()
    data=report(ROOT,args.stage)
    payload=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n" if args.json else markdown(data)
    if args.out:
        require(data["status"] in ("complete","no hybrid qualified for stress"),"Stage is pending; use stdout for progress")
        if args.out.exists():
            require(args.out.read_text(encoding="utf8")==payload,"Existing report differs")
        else:
            args.out.write_text(payload,encoding="utf8")
    else:
        sys.stdout.buffer.write((payload.rstrip()+"\n").encode("utf8"))
