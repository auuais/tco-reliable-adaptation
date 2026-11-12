"""Seal the crossover, score both references, and apply the registered branch rule."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import shutil
from statistics import mean
import subprocess

import numpy as np

from evaluation_stability import dump
from neighbor_sparse_depth import evaluate_points, point_map, tensor_digest
from plan_mast3r_crossover import ROOT, PARENT, MANIFEST, read, require, sha256, text_sha256
from prepare_dense_focal_pilot import write_arrays
from score_mast3r_transfer import resample


PUBLIC = ROOT / "results/mast3r_geometry_matching_v1"
RAW = ROOT / "reports/mast3r_geometry_matching_v1"
EVALUATOR_SOURCES = ("scripts/score_depth_scale_study.py", "scripts/run_repeat_calibration.py",
                     "third_party/TCO/evaluation/mv_recon/utils.py")


def seal():
    plan = read(MANIFEST)
    session = read(PUBLIC / "session.json")
    rows, coordinates = [], {}
    (RAW / "points").mkdir(parents=True, exist_ok=True)
    for job in plan["runs"]:
        tag = job["id"]
        folder = PUBLIC / "cases" / tag
        done = read(folder / "complete.json")
        require(done["job"] == job and done["manifest_sha256"] == sha256(MANIFEST), "Job identity changed")
        require(done["sources"] == session["sources"], "Mixed runner versions")
        for name, digest in done["sources"].items():
            require(text_sha256(ROOT / name) == digest, f"Runner source changed: {name}")
        for relative, digest in done["files"].items():
            require(sha256(ROOT / relative) == digest, f"Changed output: {relative}")
        if "endpoint" in job:
            require(read(folder / "bridge.json")["pass"], f"Endpoint unresolved: {tag}")
        native = RAW / "cases" / tag / "native.npz"
        with np.load(native) as data:
            depth, k = resample(data["depth"], data["intrinsics"].astype(float))
            points = point_map(depth, k, np.linalg.inv(data["c2w"].astype(float))[:, :3], np.ones(5))
        require(np.isfinite(points).all(), f"Nonfinite points: {tag}")
        digest = tensor_digest(points)
        write_arrays(RAW / "points" / (digest + ".npz"), dict(points=points))
        coordinates[tag] = points[:, ::9, ::9]
        destination = PUBLIC / "native_outputs" / (tag + ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(native, destination)
        require(sha256(destination) == sha256(native), "Native export differs")
        rows.append(dict(**job, input_key=job["scene"] + "_base", prediction=digest,
                         native_sha256=sha256(native), completion_sha256=sha256(folder / "complete.json")))
    write_arrays(PUBLIC / "replay_points.npz", coordinates)
    record = dict(manifest_sha256=sha256(MANIFEST), session_sha256=sha256(PUBLIC / "session.json"),
                  scoring_source_sha256=text_sha256(ROOT / "scripts/score_mast3r_crossover.py"),
                  evaluator_sources={name: text_sha256(ROOT / name) for name in EVALUATOR_SOURCES}, rows=rows)
    if (PUBLIC / "predictions.json").exists():
        require(read(PUBLIC / "predictions.json") == record, "Prediction seal changed")
    dump(PUBLIC / "predictions.json", record)
    print("Sealed all 48 predictions before hybrid scoring.", flush=True)


def normalized(metrics, ordinary):
    accuracy = metrics["accuracy_mean_m"] / ordinary["accuracy_mean_m"]
    completeness = metrics["completeness_mean_m"] / ordinary["completeness_mean_m"]
    return dict(score=(accuracy + completeness) / 2, accuracy=accuracy, completeness=completeness,
                accuracy_cm=100 * metrics["accuracy_mean_m"], completeness_cm=100 * metrics["completeness_mean_m"],
                normal_consistency=metrics["normal_consistency_mean"])


def summarize(predictions, observations, parent_predictions, parent_metrics):
    plan = read(MANIFEST)
    parent_rows = {r["tag"]: r for r in parent_predictions["rows"]}
    ordinary = {r["scene"]: r for r in parent_predictions["rows"] if r["arm"] == "zero" and r["post"] == "ordinary"}
    conditions = {}
    for grid in ("centered", "shifted"):
        for pose in ("clean", "noisy"):
            references = {}
            for reference in plan["references"]:
                scenes, errors = {}, []
                for scene in plan["scenes"]:
                    baseline = parent_metrics[scene + "_base"][ordinary[scene]["prediction"]][reference]
                    cells = {}
                    for row in predictions["rows"]:
                        if row["scene"] != scene or row["grid"] != grid or row["pose_packet"] != scene + "_" + pose:
                            continue
                        cell = f'G{row["geometry"]}B{row["bundle"]}'
                        cells[cell] = normalized(observations[row["input_key"]][row["prediction"]][reference]["metrics"], baseline)
                        if "endpoint" in row:
                            prior = parent_rows[row["endpoint"]["tag"]]
                            old = normalized(parent_metrics[row["input_key"]][prior["prediction"]][reference], baseline)
                            errors.append(abs(cells[cell]["score"] - old["score"]))
                    require(set(cells) == {"G0B0", "G1B1", "G1B0", "G0B1"}, "Missing factorial cell")
                    scenes[scene] = cells
                overall = {cell: {field: mean(c[cell][field] for c in scenes.values())
                                  for field in next(iter(scenes.values()))[cell]} for cell in next(iter(scenes.values()))}
                s00, s11, s10, s01 = (overall[c]["score"] for c in ("G0B0", "G1B1", "G1B0", "G0B1"))
                interaction = s11 - s10 - s01 + s00
                require(abs((s00 - s11) - ((s00 - s10) + (s00 - s01) - interaction)) < 1e-12, "Contrast identity failed")
                references[reference] = dict(scenes=scenes, overall=overall,
                    resolution_margin=max(1e-6, 2 * max(errors)), endpoint_score_errors=errors,
                    contrasts=dict(geometry_gain=s00-s10, observation_gain=s00-s01,
                                   preserving_observations_gain=s11-s10, interaction=interaction),
                    worst_scenes={cell: max(scenes, key=lambda scene: scenes[scene][cell]["score"]) for cell in overall})
            conditions[grid + "_" + pose] = references
    checks = {"geometry": {}, "observations": {}}
    for name, references in conditions.items():
        for reference, values in references.items():
            scores = {cell: value["score"] for cell, value in values["overall"].items()}
            t = values["resolution_margin"]
            if reference == "registered_rgb":
                checks["geometry"][name + "/" + reference] = scores["G1B0"] < min(scores["G0B0"], scores["G1B1"]) - t
                checks["observations"][name + "/" + reference] = scores["G0B1"] < scores["G0B0"] - t and scores["G0B1"] <= scores["G1B1"] + t
            else:
                checks["geometry"][name + "/" + reference] = scores["G1B0"] <= min(scores["G0B0"], scores["G1B1"]) + t
                checks["observations"][name + "/" + reference] = scores["G0B1"] < scores["G0B0"] and scores["G0B1"] <= scores["G1B1"] + t
    qualified = [branch for branch, values in checks.items() if all(values.values())]
    gains = {branch: min(1 - refs["registered_rgb"]["overall"][cell]["score"] / refs["registered_rgb"]["overall"]["G0B0"]["score"]
                        for refs in conditions.values()) for branch, cell in (("geometry", "G1B0"), ("observations", "G0B1"))}
    selected = max(qualified, key=lambda branch: (gains[branch], branch == "geometry")) if qualified else None
    return dict(conditions=conditions, branch_checks=checks, worst_primary_gains=gains,
                selected_branch=selected, next_step="registered follow-up" if selected else "local no-go attribution report")


def score():
    predictions = read(PUBLIC / "predictions.json")
    require(predictions["scoring_source_sha256"] == text_sha256(ROOT / "scripts/score_mast3r_crossover.py"), "Scoring source changed")
    plan = read(MANIFEST)
    for name, digest in predictions["evaluator_sources"].items():
        if name.startswith("third_party/TCO/"):
            revision = read(ROOT / "manifests/canonical_sensor_factorial.json")["upstream_commit"]
            repository, relative = ROOT / "third_party/TCO", name.removeprefix("third_party/TCO/")
        else:
            repository, revision, relative = ROOT, plan["parent_commit"], name
        historical = subprocess.check_output(["git", "-C", str(repository), "show", revision + ":" + relative]).decode().replace("\r\n", "\n")
        require(hashlib.sha256(historical.encode()).hexdigest() == digest == text_sha256(ROOT / name), "Evaluator differs from parent")
    parent_observations = read(PARENT / "observations.json")
    observations, sources, todo = {}, {}, {}
    for row in predictions["rows"]:
        key, digest = row["input_key"], row["prediction"]
        prior = parent_observations[key].get(digest)
        if prior is None:
            todo.setdefault(key, set()).add(digest)
            continue
        observations.setdefault(key, {})[digest] = prior
        sources.setdefault(key, {})[digest] = "parent exact prediction digest and evaluator"
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {key: pool.submit(evaluate_points, key, digests, RAW) for key, digests in todo.items()}
        for key, future in jobs.items():
            _, metrics = future.result()
            for digest, references in metrics.items():
                folder = RAW / "evaluation" / key / digest
                observations.setdefault(key, {})[digest] = {ref: read(folder / ref / "checked.json") for ref in references}
                sources.setdefault(key, {})[digest] = "fresh evaluation"
    dump(PUBLIC / "observations.json", observations)
    associations = []
    for row in predictions["rows"]:
        for reference in plan["references"]:
            key, digest = row["input_key"], row["prediction"]
            observed = observations[key][digest][reference]
            require(observed["identity"]["prediction"] == digest and observed["identity"]["reference"] == reference, "Metric identity differs")
            associations.append(dict(run_id=row["id"], reference=reference, prediction=digest,
                                     source=sources[key][digest], evaluator_sources=predictions["evaluator_sources"], **observed))
    require(len(associations) == 96, "Expected 96 score associations")
    dump(PUBLIC / "associations.json", associations)
    summary = summarize(predictions, observations, read(PARENT / "predictions.json"), read(PARENT / "metrics.json"))
    dump(PUBLIC / "scores.json", dict(seal_sha256=sha256(PUBLIC / "predictions.json"), **summary))
    print("Selected branch:", summary["selected_branch"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "score"))
    args = parser.parse_args()
    (seal if args.action == "seal" else score)()
