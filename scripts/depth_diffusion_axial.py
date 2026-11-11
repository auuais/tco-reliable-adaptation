"""Measure corresponding-pixel depth error on sealed diffusion maps."""
import argparse
from statistics import mean

import numpy as np

import depth_correction_diffusion as study
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, PUBLIC, RAW, SCENES
from score_saved_geometry import read

RESULT = PUBLIC/"depth_diffusion_axial.json"


def measure():
    rows = []
    model = study.prior.old.model
    index = read(model.INDEX)["frames"]
    scales = read(model.prior.PRIOR/"source_transplant.json")["batches"]
    for scene in SCENES:
        slug = scene.split("/")[0]
        key = slug + "_base"
        digest = read(study.SEAL)["hashes"][key]
        with np.load(study.DATA/"points"/(digest+".npz")) as archive:
            points = archive["points"]
        if study.prior.tensor_digest(points) != digest:
            raise ValueError("Changed point map")
        _, depth, _, poses, _ = study.prior.inputs(key, read(study.prior.PARENT/"predictions.json")["rows"], read(study.prior.PLAN))
        corrected = np.einsum("nhwj,nj->nhw", points, poses[:,2,:3]) + poses[:,2,3,None,None]
        for i, frame in enumerate(FRAMES):
            ref = RAW/"reference"/slug/f"{frame:06d}.npz"
            if sha(ref) != index[scene][str(frame)]["reference_sha256"]:
                raise ValueError("Changed reference")
            with np.load(ref) as archive:
                mask = archive["valid"]
                truth = archive["depth"][mask]
            def error(z):
                return float(np.sum(np.log(scales[key]["scale_to_metres"] * z[mask] / truth)**2))
            rows.append(dict(scene=scene, frame=frame, direct=error(depth[i]), diffused=error(corrected[i])))
    return rows


def summary(rows):
    if [(r["scene"], r["frame"]) for r in rows] != [(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError("Unexpected evaluation targets")
    return mean(1-r["diffused"]/r["direct"] for r in rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("measure","verify"))
    deps = [sha(path) for path in (study.SEAL, study.prior.old.model.INDEX, study.prior.old.model.prior.PRIOR/"source_transplant.json")]
    if p.parse_args().action == "measure":
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = measure()
        dump(RESULT, dict(inputs=deps, rows=rows, gain=summary(rows)))
    data = read(RESULT)
    if data["inputs"] != deps or summary(data["rows"]) != data["gain"]:
        raise ValueError("Depth comparison differs")
    print(data["gain"])
