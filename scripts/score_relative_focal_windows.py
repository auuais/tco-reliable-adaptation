"""Replay the frozen relative focal methods on newly collected image windows."""
import argparse
import importlib.metadata
from pathlib import Path
import sys
import zipfile

import numpy as np

from calibration_audit import resized_k
from collect_relative_focal_windows import OUT, PLAN, RAW, ROOT, SCENES, UPSTREAM, check, source
from evaluation_stability import dump, sha
from fit_sensor_extrinsic import homogeneous
from focal_zoom_gradient import CASES, LEVELS, noisy_poses, observe
from focal_zoom_observations import SCALE
from measure_focal_crop_response import focal_xy
from relative_focal_confirmation import aggregate, confirmation
from replay_focal_track_decisions import load_folds
from score_saved_geometry import read
from visual_focal_baseline import METHODS, estimate, fundamental_pairs, keys, pairs


CONFIGS = [(case, v, sign) for v in range(5) for sign in (-1, 1) for case in CASES[:3]]
CONFIGS.append(("uniform_clean", None, 0))


def pack():
    rows = []
    destination = OUT/"redetected_tracks.zip"
    part = destination.with_suffix(".partial")
    with zipfile.ZipFile(part, "w") as z:
        for scene in SCENES:
            tags = ["base"] + [f"v{v}_{z}" for v in range(5) for z in ("wide", "tight")]
            for tag in tags:
                folder = RAW/scene.replace("/", "_")/tag
                camera, tracks = (read(folder/(n+".json")) for n in ("camera", "tracks"))
                for record in (camera, tracks):
                    if (record["scene"], record["batch"], record["manifest_sha256"]) != (scene, tag, sha(PLAN)):
                        raise ValueError("Wrong replication input")
                if tracks["rgb_sha256"] != camera["rgb_sha256"] or sha(folder/"tracks.npz") != tracks["matches_sha256"]:
                    raise ValueError("Camera pixels or correspondence bytes disagree")
                rows.append(dict(camera=camera, tracks=tracks))
                z.write(folder/"tracks.npz", keys(scene, tag))
    part.replace(destination)
    dump(OUT/"observations.json", rows)


def assess(rows):
    base = aggregate(rows)
    m = base["primary"]["methods"]
    simple = m["relative"]
    def better_unconditional(methods):
        return min(methods[k]["focal_gradient"] for k in ("retain", "consensus"))
    checks = dict(valid_edits=simple["valid_edits"] <= 3,
                  gain=simple["focal_gradient"] <= .8 * better_unconditional(m),
                  scenes=all(t["methods"]["relative"]["focal_gradient"] <= better_unconditional(t["methods"])
                             for t in base["by_scene"].values()),
                  raw_with_folds=simple["focal_gradient"] <= m["raw_with_folds"]["focal_gradient"])
    primary_pass = base.pop("supports_new_windows")
    simpler_pass = all(checks.values())
    if simpler_pass and (not primary_pass or simple["focal_gradient"] <= m["relative_with_folds"]["focal_gradient"]):
        pilot = "relative"
    elif primary_pass:
        pilot = "relative_with_folds"
    else:
        pilot = None
    return dict(**base, registered_replication_pass=primary_pass, model_only_checks=checks, dense_pilot_method=pilot)


def camera_rig(archives, scene, expected):
    from utils.geometry import closed_form_inverse_se3
    _, poses, actual = source(archives, scene)
    if actual != expected:
        raise ValueError("Unexpected source poses")
    h = np.asarray(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    return h @ homogeneous(closed_form_inverse_se3(poses))


def tag_for(case, view, sign):
    return "base" if case in ("metadata_fault", "uniform_clean") else f"v{view}_" + ("wide" if sign < 0 else "tight")


def score(archives):
    import torch
    from audit_masked_camera_gradients import gradient
    if importlib.metadata.version("poselib") != "2.0.5":
        raise ValueError("Wrong solver version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = check()
    obs = {(x["camera"]["scene"], x["camera"]["batch"]):x for x in read(OUT/"observations.json")}
    results, gradients, solver, fundamentals = [], [], [], {}
    with zipfile.ZipFile(OUT/"redetected_tracks.zip") as z:
        for scene in SCENES:
            native = camera_rig(archives, scene, spec["inputs"][scene])
            base = np.repeat(resized_k(525., 518, 392, uniform=False)[None], 5, axis=0).astype(float)
            base[:,[0,1],[0,1]] *= SCALE
            for case, view, sign in CONFIGS:
                observed = base.copy()
                if case in ("valid_zoom", "metadata_fault"):
                    observed[view,[0,1],[0,1]] *= SCALE**sign
                tag = tag_for(case, view, sign)
                o = obs[scene,tag]
                code = np.asarray(o["camera"]["encoding"], np.float32)
                folds = load_folds(z, o["tracks"])
                batch = (code, folds, o["camera"], o["tracks"])
                key = keys(scene,tag)
                if key not in fundamentals:
                    fundamentals[key] = fundamental_pairs(pairs(z,o), observed)
                continuous, snapped, detail = estimate(fundamentals[key], observed)
                solver.append(dict(scene=scene, case=case, view=view, direction=sign, batch=tag,
                                   continuous=continuous[:,[0,1],[0,1]].tolist(),
                                   snapped=snapped[:,[0,1],[0,1]].tolist(), **detail))
                for level in LEVELS:
                    r, original = observe(scene, native, base, batch, case, view, sign, level)
                    _, edits = confirmation(observed[:,[0,1],[0,1]], focal_xy(o["camera"]), r["edits"]["both_folds"])
                    if edits["raw"] != r["edits"]["model_nearest"]:
                        raise ValueError("Model-only comparator does not match the parent reference")
                    edits.update({m:r["edits"][m] for m in ("retain","consensus","both_folds")})
                    errors = {m:r["errors"]["consensus" if edit else "retain"] for m,edit in edits.items()}
                    pose = noisy_poses(native, scene, level[1], level[2], level[3])
                    extra = np.stack([gradient(code, pose, k, None) for k in (continuous, snapped)])
                    correct = original[2,:,7:].astype(float)
                    norm = max(1e-6, float(np.linalg.norm(correct)))
                    reference = {m:float(np.linalg.norm(g[:,7:].astype(float)-correct))/norm for m,g in zip(METHODS,extra)}
                    results.append(dict(**{k:r[k] for k in ("scene","case","view","direction","batch","noise","seed","rgb_sha256",
                                                "fold_gains","log_focal_discrepancy")},
                                        edited=edits, errors=errors, classical_gradients=reference))
                    gradients.append(np.concatenate((original, extra)))
    summary = assess(results)
    dump(OUT/"solver_outputs.json",dict(fundamentals=fundamentals,calibrations=solver))
    with (OUT/"gradient_arrays.zip").open("xb") as f:
        np.savez_compressed(f, variants=np.stack(gradients))
    names = ("redetected_tracks.zip","observations.json","gradient_arrays.zip","solver_outputs.json")
    dump(OUT/"summary.json", dict(manifest_sha256=sha(PLAN), inputs={n:sha(OUT/n) for n in names},
                                  rows=results, results=summary, additional_cloud_usd=0))
    print("Primary results:", summary["primary"], "Dense development candidate:", summary["dense_pilot_method"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack","score"))
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(UPSTREAM))
    check()
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = ("observations.json","redetected_tracks.zip") if args.action == "pack" else (
        "summary.json","gradient_arrays.zip","solver_outputs.json")
    if any((OUT/n).exists() for n in outputs):
        raise FileExistsError("Replication output already exists")
    if args.action == "pack":
        pack()
    else:
        score(args.archives)
