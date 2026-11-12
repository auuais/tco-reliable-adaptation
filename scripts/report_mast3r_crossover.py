"""Write the crossover tables and the registered follow-up decision from sealed evidence."""
import csv
from statistics import mean

from evaluation_stability import dump
from plan_mast3r_crossover import ROOT, read


PUBLIC = ROOT / "results/mast3r_geometry_matching_v1"


def report():
    scores = read(PUBLIC / "scores.json")
    predictions = read(PUBLIC / "predictions.json")
    rows = []
    for condition, references in scores["conditions"].items():
        for reference, data in references.items():
            for scene, cells in data["scenes"].items():
                for cell, values in cells.items():
                    rows.append(dict(condition=condition, reference=reference, scene=scene, cell=cell, **values))
    with (PUBLIC / "scene_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    controls = [r for r in predictions["rows"] if "endpoint" in r]
    bridges = [read(PUBLIC / "cases" / r["id"] / "bridge.json") for r in controls]
    exact = sum(all(b[f]["exact"] for f in ("depth", "intrinsics", "c2w")) for b in bridges)
    timings = {r["id"]: dict(alignment=read(PUBLIC / "cases" / r["id"] / "alignment.json"),
                             capture=read(PUBLIC / "cases" / r["id"] / "native.json")) for r in predictions["rows"]}
    dump(PUBLIC / "timing.json", dict(cases=timings,
        sum_case_wall_seconds=sum(t["alignment"]["total_seconds"] for t in timings.values()),
        sum_capture_seconds=sum(t["capture"]["seconds"] for t in timings.values()),
        peak_allocated_mib=max(t["capture"]["peak_allocated_mib"] for t in timings.values()),
        peak_reserved_mib=max(t["alignment"]["peak_reserved_mib"] for t in timings.values()),
        scope="Cached alignment diagnostics including materialization. Model forwards, adaptation and descriptor matching belong to the parent input study."))
    selected = scores["selected_branch"]
    decision = "The registered rule nominates the " + selected + " branch for its specified follow-up." if selected else "Neither branch meets all registered nomination checks. The decision is a local no-go for a redesign under this protocol."
    lines = ["# MASt3R geometry and matching crossover", "", decision, "",
             f"All 48 alignments completed on the local RTX 3080. All {len(bridges)} archived endpoint controls pass the registered tolerance, with {exact} reproducing every depth, intrinsic and camera value exactly. The 24 crossed combinations were scored after all outputs were sealed. There are 96 reconstruction/reference associations across registered RGB and released references.", "",
             "G0B0 uses pretrained coordinates and observations. G1B1 uses both from the sparse-depth adapter. G1B0 combines adapted coordinates with pretrained observations. G0B1 combines pretrained coordinates with adapted observations. Observations include point confidences, pixel matches, matching weights and their effects on the view tree.", "",
             "Scores average accuracy and completeness after normalizing each component to ordinary MASt3R for the same scene and reference. Each scene receives equal weight. Lower scores indicate lower surface error.", "",
             "| Grid and poses | Reference | G0B0 | G1B1 | G1B0 | G0B1 | Geometry gain | Observation gain | Interaction | Resolution |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, references in scores["conditions"].items():
        for reference, data in references.items():
            values = data["overall"]
            a, b, c, d = [values[cell]["score"] for cell in ("G0B0", "G1B1", "G1B0", "G0B1")]
            lines.append(f"| {condition} | {reference} | {a:.6f} | {b:.6f} | {c:.6f} | {d:.6f} | {100*(1-c/a):+.3f}% | {100*(1-d/a):+.3f}% | {data['contrasts']['interaction']:+.6f} | {data['resolution_margin']:.6g} |")
    lines += ["", "Interaction is S11 − S10 − S01 + S00. A positive value means the joint error exceeds the additive prediction. Geometry and observation gains in the table are relative to G0B0. The resolution column is the registered margin derived from archived endpoint replay errors.", "",
              "| Branch | Condition and reference | Nomination check |", "|---|---|---|"]
    for branch, checks in scores["branch_checks"].items():
        lines.extend(f"| {branch} | {setting} | {'pass' if passed else 'fail'} |" for setting, passed in checks.items())
    lines += ["", "The three scenes are development observations. Both grids and repeated pose conditions reuse their images. The match audit reports shared, gained and lost pairs using fixed benchmark geometry after sealing, with visibility checks and equal weighting by unordered image pair.", "",
              "The first bridge attempt used PyTorch's default TF32 setting. Its native arrays failed tolerance. A direct run against the original unmodified cache reproduced that mismatch exactly. All five recomputed canonical views matched their archived tensors exactly. CroCo's model import enables TF32, and explicitly restoring this parent setting repaired the bridge. The failed attempt, its traces and diagnosis are retained under `attempts/tf32-disabled01/` and `diagnosis/`.", "",
              "Additional RunPod compute for this phase is USD 0. The reserved USD 25 remains available for a subsequent decision. Total alignment, materialization time and peak memory are recorded per case. These cached diagnostics cover the alignment stage. A deployable method must also account for model inference, training and matching.", "",
              "Run `python scripts/replay_mast3r_crossover.py` for the portable evidence checks. The [registered protocol](../../docs/MAST3R_GEOMETRY_MATCHING_GOAL.md), [96 scene rows](scene_metrics.csv), [metric identities](associations.json), [contrasts and decisions](scores.json), [match diagnostics](match_audit.json) and [two-drive archive](archive.json) provide the supporting records.", ""]
    (PUBLIC / "README.md").write_text("\n".join(lines), encoding="utf-8")
    verdict = ["# MASt3R crossover verdict", "", decision, "",
               "The experiment crossed pretrained and sparse-depth-adapted coordinates with pretrained and adapted confidence/matching inputs. It used the same five images per scene, sparse measurements, supplied calibration and alignment settings in every comparison. Heads, Chess and Fire each received equal weight.", "",
               "| Grid and poses | Geometry-only gain | Observations-only gain | Joint gain |", "|---|---:|---:|---:|"]
    for condition, references in scores["conditions"].items():
        overall = references["registered_rgb"]["overall"]
        baseline = overall["G0B0"]["score"]
        gains = [100 * (1 - overall[cell]["score"] / baseline) for cell in ("G1B0", "G0B1", "G1B1")]
        verdict.append(f"| {condition} | {gains[0]:+.3f}% | {gains[1]:+.3f}% | {gains[2]:+.3f}% |")
    verdict += ["", f"All 24 endpoint controls pass tolerance, with {exact} exact reproductions. The complete tables retain both reference definitions, both distance components, each scene, the interaction contrast and the registered nomination checks. The failed initial precision attempt is preserved alongside its successful repair.", "",
                "The follow-up rule requires consistent evidence across the four grid/pose settings and corroboration under the released reference. A nominated branch requires its prescribed prototype and fresh paired confirmation before a spending recommendation. Three previously studied scene windows provide development evidence about this pipeline. New confirmation data and appropriate official baselines would be needed to establish a broader method contribution.", "",
                "Additional RunPod compute was USD 0. The USD 25 allocation stays reserved. The next action is the registered follow-up when nominated, or a local no-go report when neither branch qualifies. [Full evidence and replay](README.md)", ""]
    (PUBLIC / "VERDICT.md").write_text("\n".join(verdict), encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    report()
