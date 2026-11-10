"""Test an RGB-track camera-selection rule on clean and perturbed development priors."""
import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from camera_depth_swap import Rig, median_baseline, relative_rig, unproject
from evaluation_stability import dump, sha
from prior_corruptions import corrupt_priors, tensor_hash
from score_saved_geometry import read, store_map
from track_camera_selection import select_geometry

ROOT = Path(__file__).resolve().parents[1]
STUDY = "track_camera_selection_v13"
MANIFEST = ROOT / "manifests" / (STUDY + ".json")
PUBLIC = ROOT / "results" / STUDY
RAW = ROOT / "reports" / STUDY
SCENES = ["heads/seq-01", "chess/seq-03", "fire/seq-03"]
PROFILES = ["clean", "rotation5", "translation10", "focal_plus10", "mixed_moderate", "mixed_severe"]
SEEDS = [1729, 1730, 1731]
BASELINES = {"initial": "000_unchanged", "initial_fitted": "000_per_view", "direct_prior": "111_unchanged",
             "prior_global_scale": "111_shared", "prior_per_view_scale": "111_per_view"}
METRICS = ["accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean"]


def make_plan():
    sources = ["scripts/screen_track_cameras.py", "scripts/track_camera_selection.py", "scripts/camera_depth_swap.py",
               "scripts/fixed_match_validation.py", "scripts/prior_corruptions.py", "scripts/score_saved_geometry.py",
               "scripts/run_repeat_calibration.py", "configs/diagnostic_profiles.json",
               "results/learned_support_v8/support.json", "results/depth_trust_v11/summary.json"]
    return dict(study=STUDY, sources={p: sha(ROOT / p) for p in sources}, scenes=SCENES, profiles=PROFILES,
                perturbation_seeds=SEEDS, baselines=BASELINES, clean_cases=3, noisy_cases=45,
                method="Form eight combinations of entire relative rotation, center and intrinsics sets from the frozen prediction and supplied priors. Align centers using median trajectory baselines. Estimate per-view depth scales by median of pair medians of triangulated axial depth divided by frozen model depth. Require positive depth in both cameras, >=1 degree ray angle, >=5 observations per edge. Bound scale to [0.5,2]. Include raw and globally scaled full-model and full-prior cameras. Select by bidirectional pixel reprojection on the opposite RGB track fold, swapping folds then averaging. Refit selected scales with all tracks.",
                support="Same frozen DISK/LightGlue correspondences and connected-track folds as v8. Both folds are used for model selection. Independent reference geometry is opened only in the dense evaluator.",
                primary="Report each profile separately with every scene/draw. Clean has one deterministic case per scene. Reference zero uses initial cameras and depth in identical arithmetic. Clean comparator includes scene means of the original 3-seed TCO and half-depth baselines. For each of clean and pooled noisy cases compare to strongest fixed baseline by mean normalized distance. To advance require >=5% gain in each group, no extra harmful cases and no scene regression >1%. Positive results require a matched noisy TCO run and repeat comparison before paid transfer.",
                limits="Clean priors derive from 7-Scenes reference cameras. Synthetic errors exercise the declared profiles. No held-out dataset or measured latency in this screen.",
                additional_runpod_usd=0)


def prepare(evidence, support):
    from repeatability_capture import array_digest

    RAW.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir(parents=True, exist_ok=False)
    entries, rows = {}, []
    support_meta = {s["scene"]: s for s in read(ROOT / "results/learned_support_v8/support.json")["scenes"]}
    for scene in SCENES:
        rid = "repeat6_" + scene.replace("/", "_") + "_both_released_seed42_r0"
        folder = evidence / "runs" / rid
        cap = read(folder / "capture.json")["records"]["initial_predictions"]
        if sha(folder / cap["file"]) != cap["file_sha256"]:
            raise ValueError("Initial head file mismatch")
        heads = dict(np.load(folder / cap["file"], allow_pickle=False))
        if array_digest(heads) != cap["tensor_sha256"]:
            raise ValueError("Initial head tensor mismatch")
        initial = relative_rig(heads["extrinsics"][0], heads["intrinsics"][0])
        depth = heads["depth"][0, ..., 0]
        supplied = dict(np.load(folder / "supplied_priors.npz", allow_pickle=False))
        metadata = read(folder / "prior_metadata.json")
        if any(tensor_hash(supplied[k]) != metadata[k + "_sha256"] for k in ("intrinsics", "extrinsics")):
            raise ValueError("Supplied prior tensor mismatch")
        pairs = []
        for rec in support_meta[scene]["pairs"]:
            path = support / rec["file"]
            if sha(path) != rec["sha256"]:
                raise ValueError("RGB correspondence file changed")
            pairs.append(dict(i=rec["i"], j=rec["j"], **dict(np.load(path, allow_pickle=False))))
        for profile in PROFILES:
            for seed in SEEDS[:1] if profile == "clean" else SEEDS:
                e, k, provenance = corrupt_priors(supplied["extrinsics"], supplied["intrinsics"], profile, seed, scene)
                prior = relative_rig(e, k)
                prior = Rig(prior.rotation, prior.center * median_baseline(initial) / median_baseline(prior), prior.intrinsic)
                winner, solutions, errors = select_geometry(initial, prior, depth, pairs)
                outputs = {}
                for method, name in dict(BASELINES, selected=winner).items():
                    rig, scales = solutions[name]
                    points = unproject(rig, depth * scales[:, None, None])
                    outputs[method] = store_map(RAW, points, scene, rid, entries)
                rows.append(dict(scene=scene, profile=profile, perturbation_seed=seed, source_run=rid, winner=winner,
                                 validation_pixels=errors, selected_depth_scales=solutions[winner][1].tolist(),
                                 prior_extrinsics_sha256=provenance["extrinsics_sha256"],
                                 prior_intrinsics_sha256=provenance["intrinsics_sha256"], outputs=outputs))
        print("Prepared", scene, flush=True)
    dump(PUBLIC / "seal.json", dict(manifest_sha256=sha(MANIFEST), entries=entries, rows=rows))


def summarize():
    seal = read(PUBLIC / "seal.json")
    if seal["manifest_sha256"] != sha(MANIFEST) or len(seal["rows"]) != 48:
        raise ValueError("Unexpected seal")
    scored = {}
    for key, entry in seal["entries"].items():
        record = read(RAW / key / "evaluation.json")
        if entry["sha256"] != record["sha256"]:
            raise ValueError("Saved evaluation hash mismatch")
        scored[key] = record
    rows = []
    for case in seal["rows"]:
        zero = scored[case["outputs"]["initial"]]
        for method, key in case["outputs"].items():
            rec = scored[key]
            if rec["reference_sha256"] != zero["reference_sha256"]:
                raise ValueError("Reference mismatch")
            m = {k: rec["metrics"][k] for k in METRICS}
            ratios = [m[k] / zero["metrics"][k] for k in METRICS[:2]]
            rows.append(dict(scene=case["scene"], profile=case["profile"], seed=case["perturbation_seed"],
                             method=method, winner=case["winner"], key=key, metrics=m,
                             score=mean(ratios), harmful=max(ratios) > 1.01))
    prior_run = read(ROOT / "results/depth_trust_v11/summary.json")["rows"]
    extra = []
    for scene in SCENES:
        zero = next(r["metrics"] for r in rows if r["scene"] == scene and r["method"] == "initial")
        for mode in ("tco", "half_shape"):
            original = [r for r in prior_run if r["scene"] == scene and r["mode"] == mode and r["repeat"] == 0]
            if len(original) != 3:
                raise ValueError("Expected three adapted seeds")
            m = {k: mean(r["metrics"][k] for r in original) for k in METRICS}
            ratios = [m[k] / zero[k] for k in METRICS[:2]]
            extra.append(dict(method=mode, profile="clean", scene=scene, metrics=m, score=mean(ratios), harmful=max(ratios)>1.01))
    tables = {}
    for group in PROFILES + ["all_noisy"]:
        subset = [r for r in rows + extra if (r["profile"] != "clean" if group == "all_noisy" else r["profile"] == group)]
        table = {}
        for method in sorted({r["method"] for r in subset}):
            selected = [r for r in subset if r["method"] == method]
            table[method] = dict(n=len(selected), score=mean(r["score"] for r in selected),
                                 harmful=sum(r["harmful"] for r in selected),
                                 metrics={k: mean(r["metrics"][k] for r in selected) for k in METRICS},
                                 scenes={scene: mean(r["score"] for r in selected if r["scene"]==scene) for scene in SCENES})
        tables[group] = table
    decisions = {}
    for group in ("clean", "all_noisy"):
        table = tables[group]
        best = min((b for b in table if b != "selected"), key=lambda b: table[b]["score"])
        b, c = table[best], table["selected"]
        gain = 1 - c["score"] / b["score"]
        checks = [gain >= .05, c["harmful"] <= b["harmful"], all(c["scenes"][s] <= 1.01*b["scenes"][s] for s in SCENES)]
        decisions[group] = dict(best_fixed=best, relative_gain=gain, gain_pass=checks[0], harm_pass=checks[1],
                                scenes_pass=checks[2], passed=all(checks))
    dump(PUBLIC / "summary.json", dict(manifest_sha256=sha(MANIFEST), seal_sha256=sha(PUBLIC / "seal.json"),
         unique_evaluations=len(scored), rows=rows, matched_clean_tco_scene_means=extra, tables=tables, decisions=decisions,
         passes_screen=all(d["passed"] for d in decisions.values()), additional_runpod_usd=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["freeze", "prepare", "summarize"])
    p.add_argument("--evidence", type=Path)
    p.add_argument("--support", type=Path)
    args = p.parse_args()
    plan = make_plan()
    if args.action == "freeze":
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        dump(MANIFEST, plan)
    elif read(MANIFEST) != plan:
        raise ValueError("Source or protocol changed")
    elif args.action == "prepare":
        prepare(args.evidence, args.support)
    else:
        summarize()


if __name__ == "__main__":
    main()
