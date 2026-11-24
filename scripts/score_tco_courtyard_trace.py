"""Rescore sealed Courtyard outputs and compare the registered adaptation traces."""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import run_experiment as base
from profiled_m3_dense import evaluate, paired_rmse
from run_profiled_d0 import ROOT, read, sha, write_new
from run_tco_frozen_prefix import signature
from tco_courtyard_trace import MILESTONES, read_arrays
from utils.geometry import unproject_depth_to_world_points_torch


PREVIOUS = "tco_full40_recompute_v1"
ARCHIVE_SHA = "4e09280cb9917ab8e155f63cfeddc484f84d9c05a9d6e926dac3b5cb5f8b5864"
ARCHIVE = Path("/data/research_results") / PREVIOUS / "snapshots/study-v1"
ENDPOINTS = ("accuracy_mean_m", "completeness_mean_m")
SEEDS = (7, 77, 777)
GROUPS = ("c8_off", "c1_off", "c1_on", "native_repeat")
RESCORE_ATOL_M = 1e-5


def data_signatures(supplied, reference):
    return dict(input_sha256=signature(dict(rgb=supplied["images"].numpy(), poses=supplied["extrs"],
                                            intrinsics=supplied["intrs"])),
                reference_sha256=signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"])))


def verify_signatures(receipt, expected):
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError("The diagnostic input or fixed reference differs from the original Courtyard window")


def vectors(scores, mask):
    support = mask.reshape(mask.shape[0], -1).sum(axis=1)
    slices = np.split(np.asarray(scores), np.cumsum(support)[:-1])
    return [float(np.mean(values)) if len(values) else None for values in slices]


def distances(predicted, reference, mask):
    # Each distance is assigned to the view of its source point. The target
    # cloud is the full registered scene, including points from other views.
    pred_to_ref = cKDTree(reference).query(predicted, workers=4)[0]
    ref_to_pred = cKDTree(predicted).query(reference, workers=4)[0]
    return dict(accuracy_mean_m=float(pred_to_ref.mean()), completeness_mean_m=float(ref_to_pred.mean()),
                accuracy_mean_m_per_view=vectors(pred_to_ref, mask),
                completeness_mean_m_per_view=vectors(ref_to_pred, mask))


def geometry_metrics(points, reference, seed):
    base.o3d.utility.random.seed(seed)
    mask, target = reference["valid_mask"], reference["pointclouds"]
    result = evaluate(points, target, mask)
    arrays = result["arrays"]
    metrics, diag = result["metrics"], result["diagnostics"]
    pre = distances(arrays["similarity_aligned_predicted"], arrays["reference_points"], mask)
    post = distances(arrays["aligned_predicted"], arrays["reference_points"], mask)
    for key in ENDPOINTS:
        if not np.isclose(post[key], metrics[key], atol=1e-12, rtol=1e-12):
            raise ValueError("Independent directed distance disagrees with diagnostic score")
    return dict(similarity_only=pre, similarity_and_icp=post,
                transforms={key: metrics[key] for key in ("similarity_scale", "similarity_rotation",
                                                          "similarity_translation", "icp_transformation")},
                icp={key: diag[key] for key in ("icp_fitness", "icp_inlier_rmse_m", "icp_correspondence_count")},
                paired_pixel_rmse_m=dict(before_alignment=diag["paired_pixel_rmse_before_similarity_m"],
                                         similarity_only=diag["paired_pixel_rmse_after_similarity_m"],
                                         similarity_and_icp=diag["paired_pixel_rmse_after_icp_m"]),
                valid_points=metrics["valid_points"],
                valid_points_per_view=diag["validation"]["valid_points_per_view"])


def rescore_archived(out, supplied, reference):
    """Evaluate the six immutable outputs and bind them to released receipts."""
    out = Path(out)
    archive_manifest = ARCHIVE / "manifest.json"
    if sha(archive_manifest) != ARCHIVE_SHA:
        raise ValueError("The previous study archive manifest has changed")
    archive_files = {row["file"]: row for row in read(archive_manifest)["selection"]["files"]}
    previous = read(ROOT / f"results/{PREVIOUS}/evidence.json")
    receipts = {row["id"]: row for row in previous["runs"]}
    signatures = data_signatures(supplied, reference)
    rows = []
    for seed in SEEDS:
        for policy in ("c8_off", "c1_on"):
            identifier = f"conf_eth3d_courtyard_{policy}_{seed}"
            public = receipts[identifier]
            if public["verdict"] != "completed":
                raise ValueError(f"The archived run must have a successful terminal receipt: {identifier}")
            terminal, = [attempt for attempt in public["attempts"] if attempt["verdict"] == "completed"]
            relative = Path("reports") / PREVIOUS / "cases" / identifier / terminal["attempt"] / "case"
            files = {}
            for name in ("prediction.npz", "complete.json"):
                archived = archive_files[(relative / name).as_posix()]
                candidate = ROOT / relative / name
                source = candidate if candidate.is_file() else ARCHIVE / "data" / relative / name
                if sha(source) != archived["sha256"]:
                    raise ValueError(f"Archived {name} changed for {identifier}")
                files[name] = source
            completion = read(files["complete.json"])
            if sha(files["complete.json"]) != terminal["complete_sha256"] or completion != public["result"]:
                raise ValueError(f"Published receipt differs from the archived receipt for {identifier}")
            verify_signatures(completion, signatures)
            with np.load(files["prediction.npz"], allow_pickle=False) as contents:
                arrays = dict(contents)
            if signature(arrays) != completion["output_sha256"]:
                raise ValueError(f"Archived prediction array signature differs for {identifier}")
            directory = out / identifier
            directory.mkdir(exist_ok=False)
            base.o3d.utility.random.seed(seed)
            production = base.evaluate(arrays["canonical"], reference, directory)
            diagnostic = geometry_metrics(arrays["canonical"], reference, seed)
            original = completion["metrics"]
            deltas = {
                "production_minus_published_m": {key: production[key] - original[key] for key in ENDPOINTS},
                "diagnostic_minus_production_m": {
                    key: diagnostic["similarity_and_icp"][key] - production[key] for key in ENDPOINTS},
            }
            parity = (original["valid_points"] == production["valid_points"] == diagnostic["valid_points"] and
                      all(abs(delta) <= RESCORE_ATOL_M for metrics in deltas.values() for delta in metrics.values()))
            rows.append(dict(id=identifier, seed=seed, policy=policy, parity=parity,
                             receipt_sha256=terminal["complete_sha256"], output_sha256=completion["output_sha256"],
                             original={key: original[key] for key in (*ENDPOINTS, "valid_points")},
                             last_loss=completion["losses"][-1], deltas=deltas, diagnostics=diagnostic))
    result = dict(rows=rows, atol_m=RESCORE_ATOL_M, passed=all(row["parity"] for row in rows),
                  archive_manifest_sha256=ARCHIVE_SHA, **signatures, rental_compute_usd=0)
    write_new(out / "archived.json", result)
    return result


def relative_l2(new, old):
    new, old = new.astype(np.float64), old.astype(np.float64)
    return float(np.linalg.norm((new - old).ravel()) / max(1e-12, np.linalg.norm(old.ravel())))


def depth_raster(arrays):
    return arrays["depth"][0, ..., 0]


def first_camera_frame(extrinsics):
    """Express all world-to-camera transforms in camera zero's coordinates."""
    transform = np.asarray(extrinsics[0], dtype=np.float64)
    rotation, translation = transform[:, :3, :3], transform[:, :3, 3]
    relative_rotation = rotation @ rotation[0].T
    relative_translation = translation - (relative_rotation @ translation[0])
    centers = -(relative_rotation.transpose(0, 2, 1) @ relative_translation[..., None])[..., 0]
    return relative_rotation, relative_translation, centers


def point_cloud(depth, extrinsics, intrinsics):
    with torch.no_grad():
        values = [torch.from_numpy(np.asarray(arr, dtype=np.float32).copy())
                  for arr in (depth, extrinsics, intrinsics)]
        return unproject_depth_to_world_points_torch(*values)[0].numpy()


def masked_quantiles(raster, mask):
    return [np.quantile(view[valid], (0.05, 0.5, 0.95)).tolist() if valid.any() else None
            for view, valid in zip(raster, mask)]


def characteristics(arrays, mask):
    depth = depth_raster(arrays)
    _, _, centers = first_camera_frame(arrays["extrinsics"])
    med = float(np.median(depth[mask]))
    focal = np.diagonal(arrays["intrinsics"][0], axis1=-2, axis2=-1)[:, :2]
    return dict(depth_q05_q50_q95=masked_quantiles(depth, mask),
                confidence_q05_q50_q95=masked_quantiles(arrays["depth_conf"][0], mask),
                median_depth=med, centers_in_first_camera=centers.tolist(),
                centers_per_median_depth=(centers / med).tolist() if med > 0 else None, focal_px=focal.tolist())


def compare_prediction(new, old, mask):
    new_depth, old_depth = depth_raster(new), depth_raster(old)
    new_rot, _, new_centers = first_camera_frame(new["extrinsics"])
    old_rot, _, old_centers = first_camera_frame(old["extrinsics"])
    rotation = new_rot @ old_rot.transpose(0, 2, 1)
    cos_theta = (np.trace(rotation, axis1=1, axis2=2) - 1) / 2
    # Near identity, the skew term avoids interpreting diagonal roundoff
    # as a large apparent rotation.
    sine = np.linalg.norm(np.stack((rotation[:, 2, 1] - rotation[:, 1, 2],
                                   rotation[:, 0, 2] - rotation[:, 2, 0],
                                   rotation[:, 1, 0] - rotation[:, 0, 1]), axis=1), axis=1) / 2
    angle = np.rad2deg(np.arctan2(sine, cos_theta))
    new_scale, old_scale = float(np.median(new_depth[mask])), float(np.median(old_depth[mask]))
    raster_depth_ratios = []
    for current, reference, valid in zip(new_depth, old_depth, mask):
        median = float(np.median(reference[valid])) if valid.any() else 0.0
        raster_depth_ratios.append(float(np.median(current[valid]) / median) if median > 0 else None)
    normalized_centers = None
    normalized_depth = None
    if min(new_scale, old_scale) > 0:
        normalized_centers = np.linalg.norm(new_centers / new_scale - old_centers / old_scale, axis=1).tolist()
        normalized_depth = relative_l2(new_depth[mask] / new_scale, old_depth[mask] / old_scale)
    return dict(
        raw_depth_relative_l2=relative_l2(new_depth[mask], old_depth[mask]),
        median_depth_normalized_relative_l2=normalized_depth, per_view_median_depth_ratio=raster_depth_ratios,
        confidence_relative_l2=relative_l2(new["depth_conf"][0][mask], old["depth_conf"][0][mask]),
        first_camera_relative_rotation_deg=angle.tolist(),
        first_camera_center_delta=np.linalg.norm(new_centers - old_centers, axis=1).tolist(),
        first_camera_center_delta_per_median_depth=normalized_centers,
        intrinsics_delta=(new["intrinsics"][0] - old["intrinsics"][0]).tolist(),
        canonical_points_relative_l2=relative_l2(new["canonical"][mask], old["canonical"][mask]))


def comparison_scores(new, old):
    return {stage: {key: new[stage][key] - old[stage][key] for key in ENDPOINTS}
            for stage in ("similarity_only", "similarity_and_icp")}


def hybrid_scores(native, candidate, reference, seed):
    mask = reference["valid_mask"]
    sources = dict(native=native, candidate=candidate)
    median = {name: float(np.median(depth_raster(arrays)[mask])) for name, arrays in sources.items()}
    if min(median.values()) <= 0:
        raise ValueError("Hybrid sensitivity requires a positive common-mask median depth")
    combinations = {}
    for convention in ("raw_scale", "native_median_depth_scale"):
        rows = {}
        for depth_name, depth_data in sources.items():
            depth_scale = 1.0 if convention == "raw_scale" else median["native"] / median[depth_name]
            for camera_name, camera_data in sources.items():
                rotation, translation, _ = first_camera_frame(camera_data["extrinsics"])
                camera_scale = 1.0 if convention == "raw_scale" else median["native"] / median[camera_name]
                extrinsics = np.concatenate((rotation, (camera_scale * translation)[..., None]), axis=-1)[None]
                points = point_cloud(depth_data["depth"] * depth_scale, extrinsics, camera_data["intrinsics"])
                score = geometry_metrics(points, reference, seed)
                rows[f"depth_{depth_name}/camera_{camera_name}"] = score["similarity_and_icp"]
        interaction = {
            key: (rows["depth_candidate/camera_candidate"][key] - rows["depth_candidate/camera_native"][key]
                  - rows["depth_native/camera_candidate"][key] + rows["depth_native/camera_native"][key])
            for key in ENDPOINTS}
        combinations[convention] = dict(cells=rows, interaction_m=interaction)
    return dict(first_camera_anchored=True, positive_depth_medians=median, comparisons=combinations)


def sealed_case(directory, filename, digest_key, signatures, identifier):
    directory = Path(directory)
    receipt = read(directory / "complete.json")
    if receipt["id"] != identifier or receipt["rental_compute_usd"] != 0:
        raise ValueError(f"Wrong completed case for {identifier}")
    actual_digest = sha(directory / filename)
    if receipt[digest_key] != actual_digest:
        raise ValueError(f"Changed milestone receipt for {identifier}")
    verify_signatures(receipt, signatures)
    return receipt, read(directory / filename)


def summaries(comparisons):
    """Describe onset relative to the single native repeat, without inference."""
    result = {}
    for group in ("chunk_shape", "same_chunk_checkpoint"):
        result[group] = {}
        for stage in ("similarity_only", "similarity_and_icp"):
            result[group][stage] = {}
            for endpoint in ENDPOINTS:
                exceeds = [int(step) for step, row in comparisons.items()
                           if abs(row[group]["score_delta_m"][stage][endpoint]) >
                           abs(row["native_repeat"]["score_delta_m"][stage][endpoint])]
                first = min(exceeds) if exceeds else None
                persistent = first is not None and exceeds == [step for step in MILESTONES if step >= first]
                result[group][stage][endpoint] = dict(first_completed_updates=first,
                                                      checkpoints_exceeding_repeat=exceeds,
                                                      persistent_to_40=persistent)
    return result


def score_traces(out, supplied, reference, trace_cases, readout_cases, clean_case):
    """Compare all three matched seeds with a common FP32-head readout."""
    out = Path(out)
    signatures = data_signatures(supplied, reference)
    mask = reference["valid_mask"]
    expected_traces = {f"trace_{group}_{seed}" for seed in SEEDS for group in GROUPS}
    if set(trace_cases) != expected_traces or set(readout_cases) != {
            item.replace("trace_", "readout_", 1) for item in expected_traces}:
        raise ValueError("Courtyard traces or separate-process readouts are incomplete")
    result = {}
    for seed in SEEDS:
        print(f"Scoring archived trajectory states for Courtyard seed {seed}", flush=True)
        cases, training, hashes = {}, {}, {}
        for group in GROUPS:
            identifier = f"trace_{group}_{seed}"
            readout_id = f"readout_{group}_{seed}"
            trace_dir = Path(trace_cases[identifier])
            readout_dir = Path(readout_cases[readout_id])
            original, trace = sealed_case(trace_dir, "trace.json", "trace_sha256", signatures, identifier)
            restored, readout = sealed_case(readout_dir, "readout.json", "readout_sha256", signatures, readout_id)
            if readout["source_trace_sha256"] != original["trace_sha256"] or restored["trace_job"] != identifier:
                raise ValueError(f"Readout came from a different trace for {identifier}")
            if tuple(map(int, trace["milestones"])) != MILESTONES or tuple(map(int, readout["milestones"])) != MILESTONES:
                raise ValueError(f"Missing milestones in {identifier}")
            cases[group] = (trace_dir, trace["milestones"], readout_dir, readout["milestones"])
            training[group] = trace["loss_trajectory"]
            hashes[group] = dict(trace=original["trace_sha256"], readout=restored["readout_sha256"],
                                 init_factors=trace["milestones"]["0"]["adapters"]["arrays_sha256"],
                                 step40_parity=readout["production_step40_parity"])
        if len({row["init_factors"] for row in hashes.values()}) != 1:
            raise ValueError(f"Trace arms differ in adapter initialization at seed {seed}")
        real_draws = [[(row["render_permutation"], row["renderer_rng_before"], row["renderer_rng_after"])
                       for row in rows] for rows in training.values()]
        if any(draws != real_draws[0] or len(draws) != 40 for draws in real_draws):
            raise ValueError(f"Trace arms differ in training draws at seed {seed}")
        comparison_rows, milestones, hybrids = {}, {}, {}
        for step in map(str, MILESTONES):
            snapshots = {}
            scores = {}
            adaptation_scores = {}
            for group, (trace_dir, trace_rows, readout_dir, readout_rows) in cases.items():
                snapshot = read_arrays(readout_dir / "milestones", readout_rows[step]["prediction"])
                if readout_rows[step]["adapter_arrays_sha256"] != trace_rows[step]["adapters"]["arrays_sha256"]:
                    raise ValueError(f"Milestone adapter binding differs at {group}, step {step}")
                snapshots[group] = snapshot
                scores[group] = geometry_metrics(snapshot["canonical"], reference, seed)
                if step != "40":
                    raw = read_arrays(trace_dir / "milestones", trace_rows[step]["prediction"])
                    raw["canonical"] = point_cloud(*(raw[key] for key in ("depth", "extrinsics", "intrinsics")))
                    adaptation_scores[group] = dict(
                        differences_from_common_readout=compare_prediction(raw, snapshot, mask),
                        geometry=geometry_metrics(raw["canonical"], reference, seed))
            comparisons = {}
            for label, new, old in (("chunk_shape", "c1_off", "c8_off"),
                                    ("same_chunk_checkpoint", "c1_on", "c1_off"),
                                    ("historical_policy", "c1_on", "c8_off"),
                                    ("native_repeat", "native_repeat", "c8_off")):
                comparisons[label] = dict(predictions=compare_prediction(snapshots[new], snapshots[old], mask),
                                          score_delta_m=comparison_scores(scores[new], scores[old]))
            comparison_rows[step] = comparisons
            milestones[step] = dict(
                common_readout=scores,
                common_predictions={key: characteristics(value, mask) for key, value in snapshots.items()},
                adaptation_readout=adaptation_scores)
            if seed == 777 and step in ("0", "40"):
                hybrids[step] = hybrid_scores(snapshots["c8_off"], snapshots["c1_on"], reference, seed)
        result[str(seed)] = dict(source_hashes=hashes, training=training, milestones=milestones,
                                 comparisons=comparison_rows, observed_onset=summaries(comparison_rows),
                                 hybrids=hybrids)
    clean = read(Path(clean_case) / "complete.json")
    verify_signatures(clean, signatures)
    if clean["id"] != "trace_clean_777" or clean["rental_compute_usd"] != 0:
        raise ValueError("The observer-effect control differs from the registered clean native run")
    traced = result["777"]["milestones"]["40"]["common_readout"]
    observer = dict(clean_native_m={key: clean["metrics"][key] for key in ENDPOINTS},
                    traced_native_m={key: traced["c8_off"]["similarity_and_icp"][key] for key in ENDPOINTS},
                    within_repeat_delta_m={key: abs(traced["native_repeat"]["similarity_and_icp"][key] -
                                                    traced["c8_off"]["similarity_and_icp"][key]) for key in ENDPOINTS})
    observer["clean_minus_traced_m"] = {key: observer["clean_native_m"][key] - observer["traced_native_m"][key]
                                       for key in ENDPOINTS}
    report = dict(seeds=result, observer_effect_seed777=observer, matched_render_rng=True,
                  onset_rule="Absolute paired endpoint difference exceeds absolute matched native-repeat difference",
                  adaptation_points="CPU FP32 unprojection from exported adaptation tensors",
                  common_points="Production no-grad readout at each sealed adapter milestone",
                  **signatures, rental_compute_usd=0)
    write_new(out / "diagnosis.json", report)
    return report
