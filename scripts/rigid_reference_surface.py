"""Score shape-preserving per-view transforms fitted with registered reference data."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import per_view_alignment as alignment
import rgb_rigid_view as rgb
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

DATA = RAW/"rigid_reference_surface"
SEAL = PUBLIC/"rigid_reference_predictions.json"
RESULT = PUBLIC/"rigid_reference_scores.json"
DESIGN = ROOT/"docs/RIGID_REFERENCE_SURFACE.md"


def infer():
    moments = read(alignment.RESULT)["moments"]
    bases = read(alignment.temporal.SEAL)["batches"]
    out = {}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        train = np.array(moments[scene]["direct"]["registered_rgb"])[:,0]
        scale, scene_rot, scene_affine = alignment.fit(train.sum(axis=0))
        base = bases[key+"_short"]["baseline_sha256"]
        cloud = alignment.temporal.load_points(alignment.temporal.OLD_POINTS,base)
        transforms = []
        for i,gram in enumerate(train):
            _,rot,affine = alignment.fit(gram,fixed_scale=scale)
            rotation = rot@scene_rot.T
            shift = (affine[-1]-scene_affine[-1])@scene_rot.T/scale
            np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(rotation),1.,atol=1e-10)
            cloud[i] = cloud[i]@rotation+shift
            transforms.append(dict(rotation=rotation.tolist(),translation=shift.tolist()))
        digest = rgb.prior.tensor_digest(cloud)
        rgb.prior.write_arrays(DATA/"points"/(digest+".npz"),dict(points=cloud))
        out[key] = dict(hash=digest,direct=base,transforms=transforms)
    return out


def score(item):
    key,row = item
    _,v = rgb.prior.evaluate_points(key,{row["hash"]},DATA)
    return key,v[row["hash"]]


def summarize(obs):
    controls = {"diffused":read(rgb.diffusion.RESULT)["observed"], "rgb_rigid":read(rgb.RESULT)["observed"]}
    scores = rgb.prior.summarize(read(rgb.prior.RESULT)["observed"],obs,extra=controls)
    return dict(references=scores["references"],uses_reference=True,eligible_for_transfer=False)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action = p.parse_args().action
    deps = dict(moments=sha(alignment.RESULT),baselines=sha(alignment.temporal.SEAL),
                protocol=hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal = read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Control inputs differ")
        deps = [sha(p) for p in (SEAL,rgb.RESULT,rgb.diffusion.RESULT,rgb.prior.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs = dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        data = read(RESULT)
        if deps!=data["inputs"] or summarize(data["observed"])!=data["results"]:
            raise ValueError("Control scores differ")
        if action=="verify":
            rgb.prior.old.verify_map_archive(PUBLIC/"rigid_reference_archive.json",
                                             {k:{"oracle":v["hash"]} for k,v in seal["rows"].items()})
        print("Reference-assisted surface evaluation complete")
