"""Fit rigid per-view reference alignments and evaluate on unseen image blocks."""
import argparse
import hashlib
from statistics import mean

import numpy as np

import depth_correction_diffusion as diffusion
import motion_paired_geometry as paired
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, PUBLIC, ROOT, SCENES
from score_saved_geometry import read

temporal = paired.temporal
surface = paired.surface
DESIGN = ROOT/"docs/PER_VIEW_ALIGNMENT.md"
RESULT = PUBLIC/"per_view_alignment.json"
ARMS = ("direct", "weighted", "diffused")
MODES = ("scene", "translation", "rigid")


def moments(points, target):
    # Gram matrix for [x, y, 1] gives exact least-squares losses without raw clouds.
    data = np.column_stack((points, target, np.ones(len(points))))
    return data.T @ data


def fit(gram, fixed_scale=None, fixed_rotation=None):
    count = gram[6,6]
    centers = gram[:6,6] / count
    cov = gram[:6,:6] - count * np.outer(centers, centers)
    if fixed_rotation is None:
        u, _, vt = np.linalg.svd(cov[:3,3:])
        rot = (u * [1., 1., np.sign(np.linalg.det(u@vt))]) @ vt
    else:
        rot = fixed_rotation
    scale = np.sum(rot * cov[:3,3:]) / np.trace(cov[:3,:3]) if fixed_scale is None else fixed_scale
    offset = centers[3:] - scale*centers[:3]@rot
    return scale, rot, np.vstack((scale*rot, -np.eye(3), offset))


def collect():
    measured = {}
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        spec = read(temporal.PLAN)["references"][key]
        path = temporal.REF/spec["file"]
        if sha(path) != spec["sha256"]:
            raise ValueError("Reference changed")
        with np.load(path) as z:
            refs = dict(z)
        valid = refs["valid_mask"]
        y, x = np.indices(valid.shape[1:])
        blocks = (x//32+y//32)%2
        paths = {"direct": (temporal.OLD_POINTS, read(temporal.SEAL)["batches"][key+"_short"]["baseline_sha256"]),
                 "weighted": (surface.DATA, read(surface.SEAL)["hashes"][key]["weighted"]),
                 "diffused": (diffusion.DATA, read(diffusion.SEAL)["hashes"][key])}
        measured[scene] = {}
        for arm, source in paths.items():
            cloud = temporal.load_points(*source)
            values = {}
            for ref in temporal.REFS:
                frames = []
                for i in range(len(FRAMES)):
                    halves = []
                    for block in range(2):
                        use = valid[i] & (blocks==block)
                        halves.append(moments(cloud[i][use], refs[ref][i][use]).tolist())
                    frames.append(halves)
                values[ref] = frames
            measured[scene][arm] = values
    return measured


def score(grams, opposite):
    if grams.shape != (len(FRAMES),2,7,7) or not np.isfinite(grams).all() or np.min(grams[:,:,6,6]) <= 0:
        raise ValueError("Invalid view moments")
    sums = dict.fromkeys(MODES, 0.)
    for fold in range(2):
        train = grams[:,fold]
        scale, rot, scene = fit(train.sum(axis=0))
        for frame, test in zip(train, grams[:,1-fold if opposite else fold]):
            transforms = (scene, fit(frame, scale, rot)[2], fit(frame, scale)[2])
            for mode, transform in zip(MODES, transforms):
                sums[mode] += float(np.sum(transform*(test@transform)))
    if min(sums.values()) <= 0 or (not opposite and not sums["rigid"]<=sums["translation"]+1e-8<=sums["scene"]+2e-8):
        raise ValueError("Invalid alignment loss")
    return {mode:v/grams[:,:,6,6].sum() for mode,v in sums.items()}


def summarize(measured):
    results = {}
    for ref in temporal.REFS:
        results[ref] = {}
        for arm in ARMS:
            modes = {}
            for label, opposite in (("same_blocks", False), ("opposite_blocks", True)):
                scenes = {s:score(np.array(measured[s][arm][ref]), opposite) for s in SCENES}
                modes[label] = dict(scenes=scenes, reduction={m:mean(1-v[m]/v["scene"] for v in scenes.values()) for m in MODES})
            results[ref][arm] = modes
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("measure","verify"))
    deps = {p.name:sha(p) for p in (temporal.PLAN, temporal.SEAL, surface.SEAL, diffusion.SEAL)}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if parser.parse_args().action == "measure":
        if RESULT.exists():
            raise FileExistsError(RESULT)
        observations = collect()
        dump(RESULT, dict(inputs=deps, moments=observations, results=summarize(observations)))
    report = read(RESULT)
    replay = summarize(report["moments"])
    if deps != report["inputs"]:
        raise ValueError("Dependency changed")
    for ref in temporal.REFS:
        for arm in ARMS:
            for label in ("same_blocks","opposite_blocks"):
                for key in ("reduction", "scenes"):
                    a = replay[ref][arm][label][key]
                    b = report["results"][ref][arm][label][key]
                    for unit in ([a] if key=="reduction" else [a[s] for s in SCENES]):
                        if not all(np.isfinite(list(unit.values()))):
                            raise ValueError("Nonfinite replay")
                    for scene in ([None] if key=="reduction" else SCENES):
                        for mode in MODES:
                            np.testing.assert_allclose((a if scene is None else a[scene])[mode],
                                                       (b if scene is None else b[scene])[mode], atol=1e-10, rtol=1e-8)
            print(ref, arm, replay[ref][arm]["opposite_blocks"]["reduction"])
