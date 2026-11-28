"""Release the archived three-run fusion pilot after both isolated CPU score jobs finish."""
import math
from statistics import mean

from run_tco_two_track_fusion import SEEDS, require
from run_tco_two_track_fusion_local import OLD, PUB, RAW, REG, ROOT, SRC
from tco_final_ln_protocol import read, seal, sha


METRICS = ("accuracy_mm", "completeness_mm")
FIRST = ([("native", s) for s in SEEDS] + [("reprojected", s) for s in SEEDS] +
         [("triplicate_anchor", 42), ("fusion", None)])
REPEAT = [("native", 42), ("reprojected", 43), ("triplicate_anchor", 42), ("fusion", None)]


def compare(a, b):
    return {k: a[k] / b[k] for k in METRICS}


def avg(scores):
    return {k: mean(row[k] for row in scores) for k in METRICS}


def max_diff(a, b):
    return max(abs(a[k]-b[k]) for k in METRICS)


def report(rooms, predictions, equal, passed):
    def pair(values):
        return f'{values["accuracy_mm"]:.3f} / {values["completeness_mm"]:.3f}'

    def change(values):
        return f'{100*(values["accuracy_mm"]-1):+.2f}% / {100*(values["completeness_mm"]-1):+.2f}%'

    line = [
        "# Three-run surface fusion on archived Full40 predictions", "",
        "**Development gate " + ("passed.**" if passed else "failed.**") +
        " The three development rooms supplied one prespecified triplet each. Every room used its first seed (42) as anchor.", "",
        "Each entry gives mean accuracy error (prediction to reference) followed by mean completeness error (reference to prediction). Both use nearest-neighbour distances after alignment, and smaller is better. The comparative percentages give the arithmetic mean of within-room ratios.", "",
        "| Room | Native single, mm | Reprojected single, mm | Native anchor, mm | Fusion, mm | Fusion vs native | Fusion vs reprojected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in [*rooms, dict(**equal,room="Equal-room mean")]:
        line.append("| " + " | ".join([r["room"],pair(r["native"]),pair(r["reprojected"]),pair(r["anchor"]),
                                      pair(r["fusion"]),change(r["fusion_over_native"]),change(r["fusion_over_reprojected"])])+" |")
    line += ["", "## Coverage, disagreement and matched controls", "",
             "| Room | Evaluable / emitted | Three-source coverage | Anchor fallback in fusion | Singleton fallback for 42 / 43 / 44 | Per-view median log-depth range |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r,p in zip(rooms,predictions):
        s=p["support"]
        divider=s["pixels"]
        full=s["support_count_histogram_0_through_3"][3]/divider
        fills=" / ".join(f"{100*n/divider:.2f}%" for n in s["per_seed_anchor_fill"])
        disagreement=s["per_view_median_supported_log_depth_range"]
        line.append(f'| {r["room"]} | {r["valid_points"]:,} / {divider:,} | {100*full:.2f}% | '
                    f'{100*s["fusion_anchor_fallback"]/divider:.3f}% | {fills} | {min(disagreement):.4f} to {max(disagreement):.4f} |')
    cost=mean(r["historical_three_trajectory_seconds"] for r in rooms)
    fuse=mean(p["fusion_only_cpu_seconds"] for p in predictions)
    cover=[p["support"]["support_count_histogram_0_through_3"][3]/p["support"]["pixels"] for p in predictions]
    fallback=[p["support"]["fusion_anchor_fallback"]/p["support"]["pixels"] for p in predictions]
    delta=[math.expm1(x) for p in predictions for x in p["support"]["per_view_median_supported_log_depth_range"]]
    line += ["", "The supplied camera rig placed the independent runs into one coordinate system. The anchor camera grid stayed fixed for all reprojected singles and for fusion. All nine native scores matched their archived distances. Three copies of the anchor gave the identical saved point bytes as its singleton in each room. A second, separately launched scorer reproduced the four registered Chess control and fusion rows.",
             "", f"Across rooms, all three sources covered {100*min(cover):.2f}% to {100*max(cover):.2f}% of pixels. Fusion fell back to the anchor in {100*min(fallback):.3f}% to {100*max(fallback):.3f}%. The candidate was therefore able to combine outputs across nearly the entire image. Depending on the view, the median supported farthest depth exceeded the nearest by {100*min(delta):.1f}% to {100*max(delta):.1f}%.",
             "", "The disagreement column gives the smallest and largest of the five per-view medians of the supported log-depth spread. Exponentiating the spread gives the largest-to-smallest supported depth ratio. The reprojected-single control includes camera-grid resampling and anchor fill. A log median favors the middle estimate; it has no signal identifying the most accurate seed when two seeds share a bias. These controls do not independently identify the contribution of the fixed anchor camera estimates.",
             "", f"Historical synchronized model times summed across a room's three ordinary Full40 trajectories averaged {cost:.1f} s. The fusion kernel in the CPU worker averaged {fuse:.2f} s. These components were measured separately. They do not establish a matched end-to-end latency or a comparison to 120 ordinary updates.",
             "", "## Decision", "",
             f"Accuracy beat the native single mean in {sum(r['fusion_over_native']['accuracy_mm']<1 for r in rooms)} of three rooms. The gate required lower accuracy error than both single-run baselines, mean completeness within +3% of native, at least two rooms improving on native accuracy and a passing sham.",
             "", ("The registered late-checkpoint and effective-LoRA experiments are eligible." if passed else
                  "The registered follow-up stops here. Three-trajectory fusion did not earn late-checkpoint experiments, a 120-update competitor or fresh held-out evaluation."),
             "", "The three rooms were previously used in development. They support a pilot decision only. Local CPU work incurred USD 0 rental.",
             "", "[Room-level decision](b_development_decision.json) · [Scorer rows and independent repeats](b_development_rows.json) · [Support, fits and reference-isolation evidence](b_development_predictions.json)", ""]
    path=PUB/"b_development_summary.md"
    body="\n".join(line)
    if path.exists():
        require(path.read_text(encoding="utf8")==body,"Release summary changed")
    else:
        path.write_text(body,encoding="utf8")


def release():
    reg = read(REG)
    lock = read(SRC/"inputs.json")
    lock_hash = sha(SRC/"inputs.json")
    qual = read(PUB/"b_fusion_qualification.json")
    require(sha(REG) == lock["registration_sha256"] == qual["registration_sha256"], "Registration mismatch")
    require(lock_hash == qual["source_lock_sha256"] and qual["qualified"], "Unqualified source lock")
    evaluator = reg["pins"]["scripts/run_experiment.py"]
    require(evaluator == sha(ROOT/"scripts/run_experiment.py"), "Scorer changed")
    require(reg["pins"]["results/tco_one_percent_v1/evidence.json"] == sha(OLD/"evidence.json"), "Baseline changed")
    baseline = {row["id"]: row for row in read(OLD/"evidence.json")["rows"]}
    jobs = {}
    for stage in ("infer", "leak-check", "score", "independent"):
        runs = sorted((RAW/"jobs").glob("tco-two-track-b-"+stage+"-*"))
        successes = [d for d in runs if (d/"terminal.json").is_file() and
                     read(d/"terminal.json")["exit_code"] == 0 and read(d/"terminal.json")["stopped"]]
        require(bool(successes), "Unfinished local job: "+stage)
        jobs[stage] = [{"start": read(d/"start.json"), "terminal": read(d/"terminal.json")} for d in successes]
    require(jobs["infer"][0]["terminal"]["utc"] < jobs["score"][0]["start"]["utc"], "Scores predate inference")

    first, repeats, predictions, rooms = [], [], [], []
    replay_diffs, archived_diffs = [], []
    room_by_name = {r["room"]: r for r in lock["rooms"]}
    require(set(room_by_name) == {r["scene"] for r in reg["b"]["dev_triplets"]}, "Unexpected room list")
    for triplet in reg["b"]["dev_triplets"]:
        name = triplet["scene"]
        room = room_by_name[name]
        slug = room["slug"]
        p = read(RAW/"points"/slug/"prediction.json")
        ablated = read(RAW/"points_reference_ablation"/slug/"prediction.json")
        require(p["source_lock_sha256"] == lock_hash and triplet["seeds"] == list(SEEDS) == p["seeds"]
                and triplet["anchor"] == p["anchor_seed"] == 42 and
                not p["inference_reference_mounted"] and not ablated["inference_reference_mounted"],
                "Prediction group or model interface mismatch")
        require(p["points_sha256"] == ablated["points_sha256"] and p["geometry"] == ablated["geometry"],
                "Decoy reference changed predictions")
        expected_hashes = p["points_sha256"]
        require(set(expected_hashes) == {"project0", "project1", "project2", "triplicate_anchor", "fusion"},
                "Unexpected prediction set")
        require(all(sha(RAW/folder/slug/(k+".npy")) == h for folder in ("points", "points_reference_ablation")
                    for k, h in expected_hashes.items()), "Corrupt point bytes")
        decoy_hash = sha(RAW/"decoy_reference"/(slug+".npz"))
        require(decoy_hash != room["reference_file_sha256"], "Unchanged decoy")
        runs = {r["seed"]: r for r in room["archives"]}
        scores = {}
        for stage, targets in (("first",FIRST), ("independent",REPEAT if name=="chess/seq-03" else [])):
            folder = RAW/"scores"/stage/slug
            require(len(list(folder.glob("*.json"))) == len(targets), f"Incomplete or extra {stage}/{slug} rows")
            for arm, seed in targets:
                tag = arm + (f"_{seed}" if seed is not None else "")
                row = read(folder/(tag+".json"))
                require((row["room"], row["arm"], row["seed"]) == (name, arm, seed)
                        and row["seeds"] == (list(SEEDS) if arm == "fusion" else [seed])
                        and row["independent"] == (stage == "independent")
                        and row["rental_compute_usd"] == 0, "Row label mismatch")
                src = (runs[seed]["digest"] if arm == "native" else
                       expected_hashes["project"+str(SEEDS.index(seed)) if arm == "reprojected" else arm])
                require(row["source_lock_sha256"] == lock_hash and row["predictions_sha256"] == src and
                        row["evaluator_sha256"] == evaluator and row["input_sha256"] == room["input_sha256"] and
                        row["reference_sha256"] == room["reference_sha256"], "Score provenance mismatch")
                require(all(math.isfinite(row[k]) and row[k] > 0 for k in METRICS), "Nonpositive metric")
                require(row["emitted_points"] == p["emitted_points"] == 5*392*518 > row["valid_points"] > 0,
                        "Wrong evaluation support")
                if stage == "independent":
                    replay_diffs.append(max_diff(row, scores[(arm,seed)]))
                    require(row["valid_points"] == scores[(arm,seed)]["valid_points"], "Replay support changed")
                    repeats.append(row)
                else:
                    scores[arm,seed] = row
                    first.append(row)
                if stage == "first" and arm == "native":
                    old = baseline[triplet["ids"][SEEDS.index(seed)]]
                    require(runs[seed]["array_sha256"] == old["output_sha256"] and
                            row["valid_points"] == old["metrics"]["valid_points"], "Archived single mismatch")
                    archived_diffs.append(max_diff(row,{k:old["metrics"][k.replace("_mm","_mean_m")]*1000
                                                        for k in METRICS}))
        require(len({r["valid_points"] for r in scores.values()}) == 1, "Support varies across methods")
        native = avg([scores["native",s] for s in SEEDS])
        repro = avg([scores["reprojected",s] for s in SEEDS])
        fusion = avg([scores["fusion",None]])
        anchor = avg([scores["native",42]])
        gridanchor = avg([scores["reprojected",42]])
        sham = avg([scores["triplicate_anchor",42]])
        details = p["geometry"]["views"]
        total = sum(v["pixels"] for v in details)
        support = {
            "pixels": total, "fusion_anchor_fallback": sum(v["fusion_anchor_fallback"] for v in details),
            "per_seed_anchor_fill": [sum(v["anchor_fill"][i] for v in details) for i in range(3)],
            "per_seed_visible": [sum(v["sources"][i] for v in details) for i in range(3)],
            "per_seed_collision_drops": [sum(v["collisions"][i] for v in details) for i in range(3)],
            "support_count_histogram_0_through_3": [sum(v["sources_per_pixel"][i] for v in details) for i in range(4)],
            "per_view_median_supported_log_depth_range": [v["median_supported_log_range"] for v in details],
        }
        require(sum(support["support_count_histogram_0_through_3"]) == total == p["emitted_points"], "Support receipt")
        rooms.append(dict(room=name, native=native, reprojected=repro, fusion=fusion, anchor=anchor,
                          reprojected_anchor=gridanchor, triplicate_anchor=sham,
                          fusion_over_native=compare(fusion,native), fusion_over_reprojected=compare(fusion,repro),
                          fusion_over_anchor=compare(fusion,anchor), valid_points=scores["fusion",None]["valid_points"],
                          sham_max_abs_score_mm=max_diff(sham,gridanchor),
                          sham_point_bytes_identical=expected_hashes["project0"]==expected_hashes["triplicate_anchor"],
                          emitted_points=total, historical_three_trajectory_seconds=sum(
                              baseline[i]["scene_seconds"] for i in triplet["ids"])))
        predictions.append(dict(room=name, archives=room["archives"], input_sha256=room["input_sha256"],
                                actual_reference_file_sha256=room["reference_file_sha256"], decoy_file_sha256=decoy_hash,
                                prediction_receipt_sha256=sha(RAW/"points"/slug/"prediction.json"),
                                decoy_run_receipt_sha256=sha(RAW/"points_reference_ablation"/slug/"prediction.json"),
                                points_sha256=expected_hashes, decoy_all_five_arms_byte_identical=True,
                                support=support, rig_fits=p["geometry"]["rig_fits"], failed_gauges=p["geometry"]["failed_gauges"],
                                unprojection_max_abs_per_seed=p["native_unprojection_max_abs"],
                                sham_max_abs_m=p["sham_max_abs_m"], fusion_only_cpu_seconds=p["fusion_cpu_seconds"]))
    require(len(first)==24 and len(repeats)==4, "Incomplete primary or independent batch")
    require(max(replay_diffs) <= 1e-4 and max(archived_diffs) <= 1e-4, "Scorer did not reproduce immutable baseline")
    series = ("native","reprojected","fusion","anchor","reprojected_anchor","triplicate_anchor",
              "fusion_over_native","fusion_over_reprojected","fusion_over_anchor")
    equal = {s: avg([r[s] for r in rooms]) for s in series}
    gate = reg["b"]["dev_gate"]
    gates = dict(coverage_complete=len(rooms)==gate["rooms"],
                 accuracy_below_native=equal["fusion_over_native"]["accuracy_mm"]<gate["versus_native_accuracy_strict"],
                 accuracy_below_reprojected=equal["fusion_over_reprojected"]["accuracy_mm"]<gate["versus_reprojected_accuracy_strict"],
                 completeness_within_native_limit=equal["fusion_over_native"]["completeness_mm"]<=gate["versus_native_completeness_inclusive"],
                 room_count_pass=sum(r["fusion_over_native"]["accuracy_mm"]<1 for r in rooms)>=gate["rooms_improving_native_accuracy_min"],
                 sham_pass=all(r["sham_max_abs_score_mm"]<=1e-4 and r["sham_point_bytes_identical"] for r in rooms))
    passed=all(gates.values())
    context=dict(study=reg["study"], registration_sha256=sha(REG), source_lock_sha256=lock_hash, rental_compute_usd=0)
    seal(PUB/"b_development_predictions.json",dict(**context,rooms=predictions,jobs=jobs,
         reference_ablation="Independent prediction bytes with the real evaluator withheld and an all-invalid, constant-depth decoy mounted separately. All five saved outputs per room matched the original reference-free run."))
    seal(PUB/"b_development_rows.json",dict(**context,rows=first,independent_rows=repeats,
         independent_max_abs_score_mm=max(replay_diffs),native_archive_max_abs_score_mm=max(archived_diffs),
         predictions_file_sha256=sha(PUB/"b_development_predictions.json")))
    decision=dict(**context,rooms=rooms,equal_room=equal,gates=gates,passed=passed,
                  rows_sha256=sha(PUB/"b_development_rows.json"),
                  accuracy_rooms_better_than_native=sum(r["fusion_over_native"]["accuracy_mm"]<1 for r in rooms),
                  analysis="Three seeds define one fusion and one mean single-run baseline per room. Division within room, then equal-room arithmetic mean of ratios. Distances in mm, ratios dimensionless.",
                  next_action=("Run the registered 24/32/40 checkpoint readouts and effective LoRA control."
                               if passed else "Close Track B development. Its checkpoint, longer-run and fresh-data experiments were conditional on a pass."),
                  uncertainty="Three previously used rooms. This development gate is descriptive. Fresh superiority remains untested.")
    seal(PUB/"b_development_decision.json",decision)
    report(rooms,predictions,equal,passed)
    print("Gate:",gates,"Passed:",passed,"Equal-room:",equal)


if __name__=="__main__":
    release()
