"""Diffuse temporal corrections along depth-continuous surfaces, then score sealed maps."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import score_surface_fitted_weights as prior
from evaluation_stability import dump, sha
from prepare_source_spacing import ROOT, RAW, PUBLIC, SCENES
from score_saved_geometry import read

DATA = RAW / "depth_diffusion"
SEAL = PUBLIC / "depth_diffusion_predictions.json"
RESULT = PUBLIC / "depth_diffusion_scores.json"
DESIGN = ROOT / "docs/DEPTH_CORRECTION_DIFFUSION.md"


def diffuse(depth, correction):
    log_depth = np.log(depth)
    conductance = [0.2 * np.exp(-0.5 * (np.diff(log_depth, axis=axis) / 0.02)**2)
                   for axis in (1, 2)]
    step = correction.copy()
    for _ in range(64):
        vertical = conductance[0] * np.diff(step, axis=1)
        horizontal = conductance[1] * np.diff(step, axis=2)
        step[:, :-1] += vertical
        step[:, 1:] -= vertical
        step[:, :, :-1] += horizontal
        step[:, :, 1:] -= horizontal
    np.testing.assert_allclose(step.mean((1, 2)), correction.mean((1, 2)), atol=1e-12)
    if step.min() < correction.min()-1e-12 or step.max() > correction.max()+1e-12:
        raise ValueError("Diffusion exceeded original correction range")
    return step


def infer():
    folds = read(prior.old.model.RESULT)["results"]["folds"]
    hashes = {}
    (DATA / "points").mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        key = scene.split("/")[0] + "_base"
        _, depth, k, e, _ = prior.inputs(key, read(prior.PARENT/"predictions.json")["rows"], read(prior.PLAN))
        fraction, cell = prior.fitting.cues(scene, depth)
        weights = [folds[scene]["weights"][c] for c in prior.old.model.CELLS]
        step = diffuse(depth, np.array(weights + [0.])[cell] * np.log1p(fraction))
        points = prior.point_map(depth * np.exp(step), k, e, np.ones(len(depth)))
        digest = prior.tensor_digest(points)
        prior.write_arrays(DATA/"points"/(digest+".npz"), dict(points=points))
        hashes[key] = digest
    return hashes


def score(item):
    key, digest = item
    _, scores = prior.evaluate_points(key, {digest}, DATA)
    return key, scores[digest]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("infer", "evaluate", "verify"))
    action = parser.parse_args().action
    inputs = dict(weights=sha(prior.old.model.RESULT), estimates=sha(prior.old.model.prior.ESTIMATES),
                  protocol=hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action == "infer":
        if SEAL.exists():
            raise FileExistsError(SEAL)
        dump(SEAL, dict(inputs=inputs, hashes=infer()))
    else:
        seal = read(SEAL)
        if seal["inputs"] != inputs or set(seal["hashes"]) != {s.split("/")[0]+"_base" for s in SCENES}:
            raise ValueError("Changed inference sources or incomplete scene set")
        deps = dict(predictions=sha(SEAL), controls=sha(prior.RESULT))
        fitted = read(prior.RESULT)["observed"]
        if action == "evaluate":
            if RESULT.exists():
                raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs = dict(pool.map(score, seal["hashes"].items()))
            dump(RESULT, dict(inputs=deps, observed=obs, results=prior.summarize(fitted, obs)))
        report = read(RESULT)
        if report["inputs"] != deps or prior.summarize(fitted, report["observed"]) != report["results"]:
            raise ValueError("Surface evidence differs")
        if action == "verify":
            prior.old.verify_map_archive(PUBLIC/"depth_diffusion_archive.json",
                                         {key: {"diffused": h} for key, h in seal["hashes"].items()})
        print(report["results"]["checks"], flush=True)
