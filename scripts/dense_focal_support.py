"""Recompute any geometric support on the exact FP32 priors staged for the GPU."""
import argparse
import time
import zipfile

import numpy as np

from evaluation_stability import dump, sha
from focal_metadata_control import focal_proposal
from prepare_dense_focal_pilot import INPUT, PUBLIC, ROOT, jobs
from replay_focal_track_decisions import independent_gain, load_folds
from score_saved_geometry import read


def verify():
    parent=ROOT/"results/relative_focal_windows"
    studied=read(parent/"summary.json")
    obs={(o["camera"]["scene"],o["camera"]["batch"]):o for o in read(parent/"observations.json")}
    inputs=read(ROOT/"manifests/dense_focal_inputs.json")
    cases={}
    with zipfile.ZipFile(parent/"redetected_tracks.zip") as z:
        for job in jobs():
            key=job["prior_key"]+"_"+job["case"]
            if key in cases:
                continue
            image=inputs["inputs"][job["input_key"]]
            tracks=obs[job["scene"],image["batch"]]["tracks"]
            if tracks["rgb_sha256"]!=image["rgb_sha256"]:
                raise ValueError("Matching used different pixels")
            prior=inputs["priors"][job["prior_key"]]
            if sha(INPUT/prior["file"])!=prior["sha256"]:
                raise ValueError("GPU camera prior changed")
            with np.load(INPUT/prior["file"],allow_pickle=False) as f:
                k=f["intrinsics"].astype(float)
                e=np.repeat(np.eye(4)[None],5,axis=0)
                e[:,:3]=f["extrinsics"]
            folds=load_folds(z,tracks)
            v,new=focal_proposal(k)
            if v!=job["view"]:
                raise ValueError("Could not identify supplied candidate")
            start=time.perf_counter()
            gains=[independent_gain(fold,e,k,new,v) for fold in folds]
            seconds=time.perf_counter()-start
            use=all(g is not None and g>=.25 for g in gains)
            r,=[r for r in studied["rows"] if (r["scene"],r["case"],r["view"],r["direction"],r["noise"],r["seed"])
                    ==(job["scene"],job["case"],v,job["direction"],"mild",137)]
            cases[key]=dict(supplied_prior_sha256=prior["sha256"],rgb_sha256=image["rgb_sha256"],
                track_sha256=tracks["matches_sha256"],gain_pixels=gains,two_folds=use,
                published_float64_agreement=bool(use==r["edited"]["both_folds"]),matcher_seconds=tracks["elapsed_seconds"],
                fold_scoring_seconds=seconds)
    return dict(replication_sha256=sha(parent/"summary.json"),cases=cases)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write",action="store_true")
    args=p.parse_args()
    value=verify()
    path=PUBLIC/"geometric_support.json"
    if args.write:
        if path.exists():
            raise FileExistsError(path)
        dump(path,value)
    else:
        prior=read(path)
        for name,one in value["cases"].items():
            archived=prior["cases"][name]
            for key in ("fold_scoring_seconds","gain_pixels"):
                original=archived.pop(key)
                actual=one.pop(key)
                if key=="gain_pixels":
                    for a,b in zip(actual,original):
                        if a is None or b is None:
                            if a is not b:
                                raise ValueError("Changed support")
                        else:
                            np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
        if value!=prior:
            raise ValueError("Published matching support differs")
    print("Recomputed support for all",len(value["cases"]),"paired dense conditions.")
