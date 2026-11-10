"""Freeze, produce and score camera-frame interventions on three development sequences."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_sensor_frame import SCENES, effective_camera
from calibration_variants import calibration_prior
from calibration_audit import resized_k
from camera_depth_swap import Rig, median_baseline, relative_rig, unproject
from evaluation_stability import dump, sha
from fit_sensor_extrinsic import edge, fit, homogeneous, signed_distances
from prior_corruptions import tensor_hash
from score_saved_geometry import read, store_map

ROOT = Path(__file__).resolve().parents[1]
STUDY = "sensor_extrinsic_v15"
RAW, PUBLIC = ROOT/"reports"/STUDY, ROOT/"results"/STUDY
MANIFEST = ROOT/"manifests"/(STUDY+".json")
REFERENCES = ("released", "registered_rgb", "rgb_sfm")
METHODS = ("initial", "direct", "intrinsics_only", "frame_only", "fixed_rgb", "fitted_rgb", "tco", "half_shape")
METRICS = ("accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean")


def plan():
    sources = ["scripts/screen_sensor_extrinsic.py", "scripts/fit_sensor_extrinsic.py",
               "scripts/camera_depth_swap.py", "scripts/calibration_variants.py", "scripts/run_repeat_calibration.py",
               "scripts/audit_sensor_frame.py", "scripts/audit_public_rgb_poses.py", "scripts/calibration_audit.py",
               "results/depth_trust_v11/sealed_predictions.json", "results/camera_depth_swap_v12/summary.json",
               "results/pointmap_precision_v9/summary.json", "results/learned_support_v8/support.json",
               "results/sensor_frame_v14/alternative_rgb_sfm.json", "manifests/sensor_frame_v14.json"]
    return dict(study=STUDY, sources={s:sha(ROOT/s) for s in sources}, scenes=list(SCENES),
                methods=list(METHODS), references=list(REFERENCES), adaptation_seeds=[42,123,2026],
                candidate="For each target development scene, fit one extrinsic on RGB matches and metric supplied poses from the OTHER two scenes. Only edges with at least 12 matches enter the fit. Apply the transform to target supplied poses with registration intrinsics, then unproject frozen VGGT axial depth. Normalize camera translations to the median baseline of the predicted rig. No reference labels or target matches in fitting.",
                fit="All eligible edges contribute equally to the sum of robust Sampson costs (2 px soft L1). Add squared standardized 6D deviation from physical registration. Units 1.5 degrees and 0.025 m, coordinate bounds +/-3. Hold focal and principal point fixed. Report Jacobian singular values without prior rows, active bounds and test-scene match errors.",
                reference="Same registered axial depth and mask for all references. Primary uses original point targets bit-for-bit. Secondary uses registration intrinsics and physical frame. A further sensitivity uses published RGB SfM with its own focal, centered principal point and median baseline matched to physical poses. Reference conventions are analyzed separately.",
                gate="Clean development only. Compare fitted_rgb to strongest fixed comparator on each reference. Mean of per-scene accuracy and completeness ratios to initial. Advance to noisy controls only if released and registered references each show >=5% relative improvement, no scene >1% worse than best fixed method, no extra harmful scenes. RGB SfM is a sensitivity. No transfer without noisy and repeat checks.",
                controls="For unchanged TCO and half-depth average the three released clean adaptation seeds within scene before aggregation. Verify released direct and initial controls to 0.5% per distance against v12; check exact released references and input hashes. Methods share scenes and evaluate identical frames. Clean controls derive from benchmark poses.",
                additional_cloud_usd=0)


def source_run(scene):
    return "repeat6_"+scene.replace("/", "_")+"_both_released_seed42_r0"


def archive_points(raw_path, points_sha, scene, method, seed=None):
    if sha(raw_path) != points_sha:
        raise ValueError(f"Changed archived point map {raw_path}")
    return dict(scene=scene, method=method, seed=seed, path=raw_path.relative_to(ROOT).as_posix(), sha256=points_sha)


def prepare(evidence, support):
    from repeatability_capture import array_digest
    physical = np.array(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    meta = {s["scene"]:s for s in read(ROOT/"results/learned_support_v8/support.json")["scenes"]}
    heads_report = {r["scene"]:r for r in read(ROOT/"results/pointmap_precision_v9/summary.json")["rows"]
                    if r["mode"]=="upstream_cpu_float32"}
    inputs, observations = {}, {}
    for scene in SCENES:
        src = evidence/"runs"/source_run(scene)
        cap = read(src/"capture.json")["records"]["initial_predictions"]
        heads = dict(np.load(src/cap["file"], allow_pickle=False))
        if sha(src/cap["file"]) != cap["file_sha256"] or array_digest(heads) != heads_report[scene]["initial_heads_tensor_sha256"]:
            raise ValueError("Frozen model heads changed")
        z = heads["depth"][0,...,0]
        initial = relative_rig(heads["extrinsics"][0], heads["intrinsics"][0])
        camera = dict(np.load(src/"supplied_priors.npz", allow_pickle=False))
        metadata = read(src/"prior_metadata.json")
        if any(tensor_hash(camera[k]) != metadata[k+"_sha256"] for k in camera):
            raise ValueError("Supplied camera changed")
        e, k = homogeneous(camera["extrinsics"]), camera["intrinsics"]
        rgb_k = calibration_prior(k, z.shape[2], z.shape[1], "axis_registration")
        inputs[scene] = z, initial, e, k, rgb_k
        observations[scene] = []
        for pair in meta[scene]["pairs"]:
            path = support/pair["file"]
            if sha(path) != pair["sha256"]:
                raise ValueError("Track archive changed")
            xy = np.load(path, allow_pickle=False)
            if len(xy["xy_i"]) >= 12:
                observations[scene].append(edge(e, rgb_k, pair["i"], pair["j"], xy["xy_i"], xy["xy_j"]))
    RAW.mkdir(parents=True, exist_ok=False)
    PUBLIC.mkdir(parents=True, exist_ok=False)
    rows, fits, entries = [], {}, {}
    for scene in SCENES:
        z, initial, e, k, rgb_k = inputs[scene]
        training = [s for s in SCENES if s != scene]
        transform, stats = fit(physical, [pair for s in training for pair in observations[s]])
        test = {m:float(np.median(np.abs(np.concatenate(signed_distances(t, observations[scene])))))
                for m,t in (("identity",np.eye(4)), ("fixed_rgb",physical), ("fitted_rgb",transform))}
        fits[scene] = dict(training_scenes=training, matrix=transform.tolist(), test_median_sampson_px=test, **stats)
        for method, h, intrinsic in (("initial",None,None), ("direct",np.eye(4),k), ("intrinsics_only",np.eye(4),rgb_k),
                                     ("frame_only",physical,k), ("fixed_rgb",physical,rgb_k), ("fitted_rgb",transform,rgb_k)):
            rig = initial
            if h is not None:
                prior = relative_rig(h @ e, intrinsic)
                rig = Rig(prior.rotation, prior.center*median_baseline(initial)/median_baseline(prior), prior.intrinsic)
            key = store_map(RAW, unproject(rig,z), scene, source_run(scene), entries)
            rows.append(archive_points(RAW/key/"points.npy", entries[key]["sha256"], scene, method))
    old = read(ROOT/"results/depth_trust_v11/sealed_predictions.json")["records"]
    for r in old:
        if r["mode"] in ("half_shape","tco") and r["repeat"] == 0:
            rows.append(archive_points(ROOT/"reports/depth_trust_v11"/r["job_id"]/"predictions.npy", r["output_sha256"],
                                       r["scene"], r["mode"], r["seed"]))
    expected = {(s,m,seed) for s in SCENES for m in METHODS
                for seed in ([42,123,2026] if m in ("tco","half_shape") else [None])}
    if {(r["scene"],r["method"],r["seed"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Incomplete prediction grid")
    dump(PUBLIC/"seal.json", dict(manifest_sha256=sha(MANIFEST), rows=rows, fits=fits))


def make_reference(arrays, scene, name, subset):
    from scipy.spatial.transform import Rotation
    from audit_public_rgb_poses import rig_as_extrinsic
    old = arrays["reference_points"]
    if name == "released":
        return old
    e = np.stack([effective_camera(p) for p in arrays["extrinsics"]])
    z = np.einsum("nhwj,nj->nhw",old,e[:,2,:3])+e[:,2,3,None,None]
    _, height, width = z.shape
    k = calibration_prior(arrays["intrinsics"],width,height,"axis_registration")
    new = np.array(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"]) @ e
    if name == "rgb_sfm":
        reference = read(ROOT/"results/sensor_frame_v14/alternative_rgb_sfm.json")
        if sha(subset) != reference["subset_sha256"]:
            raise ValueError("Unexpected published pose subset")
        data = read(subset)[scene]
        p = np.array([data["poses"][key] for key in data["keys"]])
        sfm = np.repeat(np.eye(4)[None],len(p),axis=0)
        sfm[:,:3,:3] = Rotation.from_quat(p[:,[1,2,3,0]]).as_matrix()
        sfm[:,:3,3] = p[:,4:7]
        k = np.stack([resized_k(f,width,height) for f in p[:,7]])
        rig = relative_rig(sfm,k)
        scale = median_baseline(relative_rig(new,k))/median_baseline(rig)
        new = rig_as_extrinsic(Rig(rig.rotation,rig.center*scale,k))
    # Invalid depth pixels may be zero. Preserve the original common mask.
    y,x = np.mgrid[:height,:width]
    pixels = np.stack((x,y,np.ones_like(x)),axis=-1)
    def lift(intrinsics, poses):
        result = []
        for depth,ki,cam in zip(z,intrinsics,poses):
            rays = pixels @ np.linalg.inv(ki).T
            c2w = np.linalg.inv(cam)
            result.append((rays*depth[...,None])@c2w[:3,:3].T+c2w[:3,3])
        return np.stack(result)
    if np.max(np.abs(lift(arrays["intrinsics"],e)-old)) > 1e-6:
        raise ValueError("Released depth or projection convention differs")
    return lift(k,new).astype(old.dtype)


def score_group(scene, name, evidence, subset):
    import open3d as o3d
    import torch
    from repeatability_capture import array_digest
    from run_repeat_calibration import evaluate
    seal = read(PUBLIC/"seal.json")
    if seal["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Incorrect seal")
    run = read(evidence/"runs"/source_run(scene)/"run.json")
    arrays = dict(np.load(evidence/run["evaluation_inputs_relative"],allow_pickle=False))
    expected = next(r["reference_tensor_sha256"] for r in read(ROOT/"results/pointmap_precision_v9/summary.json")["rows"]
                    if r["scene"]==scene)
    if array_digest(arrays) != expected or run["evaluation_tensor_sha256"] != expected:
        raise ValueError("Changed source reference")
    points = make_reference(arrays,scene,name,subset)
    mask = arrays["valid_mask"]
    ref_hash = array_digest(dict(points=points,mask=mask))
    data = dict(images=torch.from_numpy(arrays["images"]),pointclouds=points,valid_mask=mask)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for r in seal["rows"]:
        if r["scene"] != scene:
            continue
        path = ROOT/r["path"]
        if sha(path) != r["sha256"]:
            raise ValueError("Sealed prediction changed")
        folder = RAW/"scores"/scene.replace("/","_")/name/(r["method"]+"_"+str(r["seed"]))
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder/"evaluation.json"
        identity = dict(scene=scene,reference=name,method=r["method"],seed=r["seed"],reference_sha256=ref_hash,
                        prediction_sha256=r["sha256"],seal_sha256=sha(PUBLIC/"seal.json"))
        if dest.exists():
            if any(read(dest)[k]!=v for k,v in identity.items()):
                raise ValueError("Resume identity differs")
            continue
        np.random.seed(42)
        torch.manual_seed(42)
        o3d.utility.random.seed(42)
        metrics = evaluate(np.load(path,allow_pickle=False),data,folder)
        if not all(np.isfinite(metrics[k]) for k in METRICS):
            raise ValueError("Nonfinite score")
        dump(dest,dict(**identity,metrics={k:metrics[k] for k in METRICS}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","prepare","score","group"))
    p.add_argument("--evidence",type=Path)
    p.add_argument("--support",type=Path)
    p.add_argument("--subset",type=Path)
    p.add_argument("--scene",choices=SCENES)
    p.add_argument("--reference",choices=REFERENCES)
    args = p.parse_args()
    if args.action == "freeze":
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        dump(MANIFEST,plan())
    elif read(MANIFEST) != plan():
        raise ValueError("Protocol differs from frozen version")
    elif args.action == "prepare":
        prepare(args.evidence,args.support)
    elif args.action == "group":
        score_group(args.scene,args.reference,args.evidence,args.subset)
    else:
        def run(pair):
            scene,name = pair
            log = RAW/"logs"/(scene.replace("/","_")+"_"+name)
            log.parent.mkdir(exist_ok=True)
            with log.with_suffix(".log").open("a") as out:
                subprocess.run([sys.executable,__file__,"group","--scene",scene,"--reference",name,
                                "--evidence",str(args.evidence),"--subset",str(args.subset)],check=True,
                               stdout=out,stderr=subprocess.STDOUT,timeout=3600,
                               env=dict(os.environ,OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",OMP_NUM_THREADS="2"))
            print("Completed",scene,name,flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run,[(s,n) for n in REFERENCES for s in SCENES]))


if __name__ == "__main__":
    main()
