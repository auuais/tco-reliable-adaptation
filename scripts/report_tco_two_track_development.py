"""Render the registered development results from released rows and the independent replay."""
from statistics import median

from tco_two_track_protocol import PUBLIC, read


def write_report():
    rows=read(PUBLIC/"a_development_rows.json")["rows"]
    analysis=read(PUBLIC/"a_development_independent.json")
    branch=read(PUBLIC/"a_alternative_gate.json")
    result=analysis["candidates"]["Refresh1"]
    if branch["candidate"]!=analysis["bounded_alternative"] or branch["analysis"]["candidates"]["Refresh1"]!=result:
        raise ValueError("Operator and independent development decisions disagree")
    valid=[r for r in rows if r["status"]=="completed"]
    lines=["# Ten-update early refresh on three development rooms","",
           f"{len(valid)} of {len(rows)} registered model slots delivered scored reconstructions. "
           "Each room used the same five input images for all arms and its three assigned seeds. "
           "Accuracy measures distance from the predicted surface to the measured surface. Completeness measures the reverse distance. Smaller is better.","",
           "## Paired comparisons","",
           "Each percentage is the mean of the three within-room ratios, using the median of three matched seeds per method and room. "
           "A timing ratio below one favors Refresh1. The limits for the primary comparison with Full10 are +1% accuracy error, +3% completeness error and a shorter scene time.","",
           "| Refresh1 compared with | Accuracy change | Completeness change | Scene time change | Peak live change |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for control in ("Full10","Cache10","Refresh1Uncached","Full40"):
        comp=result["comparisons"][control]
        cells=([f"{comp['metrics'][m]['percent_change']:+.2f}%" for m in
                ("accuracy_mm","completeness_mm","scene_seconds","peak_allocated_bytes")]
               if comp["coverage_complete"] else ["Incomplete pairs"]*4)
        lines.append("| "+" | ".join([control,*cells])+" |")
    lines+=["","The bootstrap uncertainty below is descriptive on these previously used rooms. "
            "Development decisions use the registered point estimates. Formal confirmation requires six fresh visits.",""]
    equal=result["comparisons"]["Full10"]
    if equal["coverage_complete"]:
        for field,label in (("accuracy_mm","accuracy error"),("completeness_mm","completeness error"),("scene_seconds","scene time")):
            m=equal["metrics"][field]
            lines.append(f"- Refresh1/Full10 {label} ratio {m['ratio']:.4f}. Its one-sided 95% lower and upper bounds are {m['lower_95']:.4f} and {m['upper_95']:.4f}.")
    else:
        lines.append("The complete registered equal-step comparison was unavailable.")
    lines+=["","## Absolute measurements","",
            "Each entry is a median over the completed seeds shown. A partial Full40 row describes its measured runs only.","",
            "| Room | Arm | Scored seeds | Accuracy (mm) | Completeness (mm) | Scene time (s) | Peak live (GiB) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for room in sorted({r["room"] for r in rows}):
        for arm in ("Full10","Cache10","Refresh1","Refresh1Uncached","Full40"):
            group=[r for r in valid if (r["room"],r["arm"])==(room,arm)]
            seeds=", ".join(str(r["seed"]) for r in sorted(group,key=lambda r:r["seed"])) or "None"
            values=([f"{median(r[k] for r in group):.3f}" for k in ("accuracy_mm","completeness_mm","scene_seconds")]+
                    [f"{median(r['peak_allocated_bytes'] for r in group)/2**30:.3f}"] if group else ["—"]*4)
            lines.append("| "+" | ".join([room,arm,seeds,*values])+" |")
    failures=[r for r in rows if r["status"]!="completed"]
    lines+=["","## Gate and coverage","",
            f"The equal-step development comparison gave Refresh1 a verdict of **{result['equal_step']['verdict']}**. "
            f"The registered follow-up schedule is **{branch['candidate'] or 'none'}**."]
    if "peak_ratio_lower_bound_from_cap" in result:
        low=result["peak_ratio_lower_bound_from_cap"]
        lines+=["",f"Even if every Full40 control had used the entire 8.75 GiB cap, the mean Refresh1/Full40 "
                   f"peak ratio would be at least {low:.4f}. This bound can rule out the 0.764 memory target. "
                   "The missing Full40 errors and elapsed times remain unknown."]
    if failures:
        lines+=["","| Room | Arm | Seed | Terminal outcome | Recorded attempts |","| --- | --- | ---: | --- | ---: |"]
        for r in failures:
            lines.append(f"| {r['room']} | {r['arm']} | {r['seed']} | {r['status']} | {len(r['attempts'])} |")
    lines+=["","All costs here are local. RunPod spend was USD 0. [Case-level scores and timing](a_development_rows.json) · "
            "[Independent ratios and uncertainty](a_development_independent.json) · [Follow-up decision](a_alternative_gate.json)"]
    target=PUBLIC/"a_development_summary.md"
    payload=("\n".join(lines)+"\n").encode()
    if target.exists() and target.read_bytes()!=payload:
        raise ValueError("Prior report differs from the published evidence")
    if not target.exists():
        target.write_bytes(payload)
    print(target)


if __name__=="__main__":
    write_report()
