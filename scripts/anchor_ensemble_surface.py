"""Seal order-ensembled depths on fixed cameras and compare canonical surfaces."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import numpy as np

import anchor_order_capture as capture
import cross_view_surface as cross
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, RAW, SCENES
from score_saved_geometry import read

prior = cross.prior
DATA = RAW/"anchor_ensemble"
SEAL = PUBLIC/"anchor_ensemble_predictions.json"
RESULT = PUBLIC/"anchor_ensemble_scores.json"
SINGLES = tuple("shift"+str(i) for i in range(1,5))
ARMS = SINGLES + ("half","median")


def infer():
    rows = read(capture.RESULT)["rows"]
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    records = {}
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        _,depth,k,e,direct = prior.inputs(key,read(prior.PARENT/"predictions.json")["rows"],read(prior.PLAN))
        fields = []
        offsets = []
        for i in range(5):
            name = f"{key}_{i}"
            path = capture.DATA/(name+".npz")
            row = rows[name]
            if sha(path)!=row["sha256"] or row["identity"]["order"]!=np.roll(np.arange(5),-i).tolist():
                raise ValueError("Invalid rotation capture")
            with np.load(path) as f:
                alternative = f["depth"][...,0].astype(float)
            if i==0:
                np.testing.assert_array_equal(alternative,depth)
            shift = np.log(alternative/depth)
            offset = np.median(shift)
            offsets.append(float(offset))
            fields.append(shift-offset)
        middle = np.median(fields,axis=0)
        updates = dict(zip(ARMS,fields[1:]+[.5*middle,middle]))
        hashes = {}
        for arm,delta in updates.items():
            points = prior.point_map(depth*np.exp(delta),k,e,np.ones(len(depth)))
            digest = prior.tensor_digest(points)
            prior.write_arrays(DATA/"points"/(digest+".npz"),dict(points=points))
            hashes[arm] = digest
        records[key] = dict(hashes=hashes,direct=direct,log_offsets=offsets,
                            median_absolute_log_step={arm:float(np.median(np.abs(v))) for arm,v in updates.items()})
    return records


def score(item):
    key,r = item
    _,by_hash = prior.evaluate_points(key,set(r["hashes"].values()),DATA)
    return key,{arm:by_hash[h] for arm,h in r["hashes"].items()}


def summarize(obs):
    if set(obs)!={s.split("/")[0]+"_base" for s in SCENES} or any(set(v)!=set(ARMS) for v in obs.values()):
        raise ValueError("Missing anchor comparison")
    controls = {"diffused":read(cross.diffusion.RESULT)["observed"],"rgb_rigid":read(cross.rgb.RESULT)["observed"],
                "two_planes":read(cross.RESULT)["observed"]}
    controls.update({a:{k:v[a] for k,v in obs.items()} for a in SINGLES+("half",)})
    joint = prior.summarize(read(prior.RESULT)["observed"],{k:v["median"] for k,v in obs.items()},extra=controls)
    oracle = {}
    for ref,g in joint["references"].items():
        best = {s:min(("direct",*SINGLES),key=lambda arm:vals[arm]["score"]) for s,vals in g["scenes"].items()}
        oracle[ref] = dict(choices=best,score=mean(g["scenes"][s][arm]["score"] for s,arm in best.items()))
    return dict(median=joint,reference_selected_single=oracle)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("infer","evaluate","verify"))
    action = parser.parse_args().action
    deps = [sha(p) for p in (capture.RESULT,prior.PLAN,prior.PARENT/"predictions.json")]
    deps.append(hashlib.sha256(capture.DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal = read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Anchor dependencies differ")
        deps = [sha(p) for p in (SEAL,cross.RESULT,cross.rgb.RESULT,cross.diffusion.RESULT,prior.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                obs = dict(pool.map(score,seal["rows"].items()))
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report = read(RESULT)
        if report["inputs"]!=deps or summarize(report["observed"])!=report["results"]:
            raise ValueError("Anchor surface replay failed")
        if action=="verify":
            rows = read(capture.RESULT)["rows"]
            if set(rows)!={f"{k}_{i}" for k in seal["rows"] for i in range(5)}:
                raise ValueError("Expected all 15 anchor captures")
            archive = read(PUBLIC/"anchor_order_archive.json")
            files = {r["file"]:r["sha256"] for r in archive["files"]}
            if (files!={k+".npz":v["sha256"] for k,v in rows.items()} or archive["verified_physical_drives"]!=2
                    or archive["total_bytes"]!=sum(r["bytes"] for r in archive["files"])):
                raise ValueError("Captured heads archive differs")
            for k,v in rows.items():
                if k.endswith("_0") and any(v["max_abs_difference"].values()):
                    raise ValueError("Original order failed capture reproduction")
            prior.old.verify_map_archive(PUBLIC/"anchor_ensemble_archive.json",{k:v["hashes"] for k,v in seal["rows"].items()})
        print(report["results"]["median"]["checks"])
