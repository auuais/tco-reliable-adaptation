"""Cross initial and adapted cameras and depth, plus direct use of camera priors."""
import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from camera_depth_swap import Rig, median_baseline, relative_rig, unproject
from evaluation_stability import dump, sha
from pointmap_camera_diagnostic import recover
from score_saved_geometry import read, store_map

ROOT = Path(__file__).resolve().parents[1]
STUDY = "camera_depth_swap_v12"
MANIFEST = ROOT / "manifests" / (STUDY + ".json")
PUBLIC = ROOT / "results" / STUDY
RAW = ROOT / "reports" / STUDY
MODES = ("initial_initial", "final_initial", "initial_final", "final_final", "prior_initial")
DISTANCES = ("accuracy_mean_m", "completeness_mean_m")


def make_plan():
    cohort = read(ROOT / "manifests/depth_trust_v11.json")["runs"]
    sources = ["scripts/run_camera_depth_swap.py", "scripts/camera_depth_swap.py", "scripts/score_saved_geometry.py",
               "scripts/pointmap_camera_diagnostic.py", "scripts/run_repeat_calibration.py",
               "results/depth_trust_v11/summary.json", "results/pointmap_precision_v9/summary.json"]
    return dict(study=STUDY, sources={p: sha(ROOT / p) for p in sources}, runs=cohort, modes=MODES,
                gauge="First camera frame. Divide final depth and final camera centers by median(final depth / initial depth). Scale supplied centers to initial median pairwise baseline, using the predicted rig only.",
                precision="FP64 relative-camera construction and ray arithmetic, one final FP32 cast for all cells.",
                analysis="Use repeat zero in the three-scene, three-seed table. Report all 13 trajectories. Two endpoint controls must agree in both distances to within 0.5 percent of the matched-precision initial and unchanged TCO scores. Report both additive-factor contrasts and their interaction under this chosen gauge.",
                prior_source="Clean reference-derived camera priors supplied to every original run. This setting does not test erroneous measurements.",
                additional_runpod_usd=0)


def prepare(plan, evidence):
    from prior_corruptions import tensor_hash
    from repeatability_capture import array_digest

    RAW.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir(parents=True, exist_ok=False)
    entries, rows = {}, []
    for run in plan["runs"]:
        folder = evidence / "runs" / run["run_id"]
        record = read(folder / "capture.json")["records"]["initial_predictions"]
        if sha(folder / record["file"]) != record["file_sha256"] or sha(folder / "predictions.npy") != run["input_points_sha256"]:
            raise ValueError("Source output hash mismatch")
        heads = dict(np.load(folder / record["file"], allow_pickle=False))
        if array_digest(heads) != record["tensor_sha256"]:
            raise ValueError("Initial tensor hash mismatch")
        z0 = heads["depth"][0, ..., 0]
        initial = relative_rig(heads["extrinsics"][0], heads["intrinsics"][0])
        fitted = [recover(view) for view in np.load(folder / "predictions.npy", allow_pickle=False)]
        z1 = np.stack([z for _, z in fitted])
        scale = float(np.median(z1 / z0))
        final = relative_rig(np.stack([c["extrinsics"] for c, _ in fitted]),
                             np.stack([c["intrinsics"] for c, _ in fitted]), scale)
        prior = dict(np.load(folder / "supplied_priors.npz", allow_pickle=False))
        metadata = read(folder / "prior_metadata.json")
        if any(tensor_hash(prior[k]) != metadata[k + "_sha256"] for k in ("intrinsics", "extrinsics")):
            raise ValueError("Supplied prior hash mismatch")
        direct = relative_rig(prior["extrinsics"], prior["intrinsics"])
        direct_scale = median_baseline(initial) / median_baseline(direct)
        direct = Rig(direct.rotation, direct.center * direct_scale, direct.intrinsic)
        combinations = ((initial, z0), (final, z0), (initial, z1 / scale), (final, z1 / scale), (direct, z0))
        for mode, (rig, depth) in zip(MODES, combinations):
            key = store_map(RAW, unproject(rig, depth), run["scene"], run["run_id"], entries)
            rows.append(dict(**run, mode=mode, key=key, final_to_initial_scale=scale, prior_scale=direct_scale))
    dump(PUBLIC / "seal.json", dict(manifest_sha256=sha(MANIFEST), entries=entries, rows=rows))


def summarize():
    seal = read(PUBLIC / "seal.json")
    if seal["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Wrong seal")
    zeros = {r["scene"]: r for r in read(ROOT / "results/pointmap_precision_v9/summary.json")["rows"]
             if r["mode"] == "upstream_cpu_float32"}
    tco = {r["run_id"]: r["metrics"] for r in read(ROOT / "results/depth_trust_v11/summary.json")["rows"]
           if r["mode"] == "tco"}
    rows, errors = [], []
    for row in seal["rows"]:
        evidence = read(RAW / row["key"] / "evaluation.json")
        zero = zeros[row["scene"]]
        if evidence["sha256"] != seal["entries"][row["key"]]["sha256"] or evidence["reference_sha256"] != zero["reference_tensor_sha256"]:
            raise ValueError("Scored a different prediction or reference")
        m = evidence["metrics"]
        ratios = [m[k] / zero["metrics"][k] for k in DISTANCES]
        if row["mode"] in ("initial_initial", "final_final"):
            control = zero["metrics"] if row["mode"] == "initial_initial" else tco[row["run_id"]]
            errors.extend(abs(m[k] / control[k] - 1) for k in DISTANCES)
        rows.append(dict(**row, metrics={k: m[k] for k in (*DISTANCES, "normal_consistency_mean")},
                         score=mean(ratios), harmful=max(ratios) > 1.01))
    cells = [r for r in rows if r["repeat"] == 0]
    modes = {}
    for mode in MODES:
        selected = [r for r in cells if r["mode"] == mode]
        if len({(r["scene"], r["seed"]) for r in selected}) != 9:
            raise ValueError("Incomplete primary grid")
        modes[mode] = dict(score=mean(r["score"] for r in selected), harmful=sum(r["harmful"] for r in selected),
                           metrics={k: mean(r["metrics"][k] for r in selected) for k in selected[0]["metrics"]},
                           scenes={s: mean(r["score"] for r in selected if r["scene"] == s) for s in zeros})
    effects = {}
    for scene in ("overall", *zeros):
        values = {m: v["score"] if scene == "overall" else v["scenes"][scene] for m, v in modes.items()}
        camera = values["final_initial"] - values["initial_initial"]
        depth = values["initial_final"] - values["initial_initial"]
        effects[scene] = dict(camera_at_initial_depth=camera, depth_at_initial_cameras=depth,
                              interaction=values["final_final"] - values["initial_initial"] - camera - depth)
    dump(PUBLIC / "summary.json", dict(seal_sha256=sha(PUBLIC / "seal.json"), rows=rows, modes=modes, effects=effects,
         unique_evaluations=len(seal["entries"]), endpoint_max_relative_distance_difference=max(errors),
         comparable_endpoints=max(errors) <= .005))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "prepare", "summarize"))
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    plan = make_plan()
    if args.action == "freeze":
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        dump(MANIFEST, plan)
    elif read(MANIFEST) != dict(plan, modes=list(MODES)):
        raise ValueError("Execution differs from frozen plan")
    elif args.action == "prepare":
        prepare(plan, args.evidence)
    else:
        summarize()


if __name__ == "__main__":
    main()
