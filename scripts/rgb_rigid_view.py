"""Fit bounded per-view rigid corrections using disjoint RGB track folds."""
import argparse
import hashlib
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import depth_correction_diffusion as diffusion
import run_depth_scale_study as tracks
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers, sample
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

prior = diffusion.prior
DATA = RAW/"rgb_rigid_view"
SEAL = PUBLIC/"rgb_rigid_view_predictions.json"
RESULT = PUBLIC/"rgb_rigid_view_scores.json"
DESIGN = ROOT/"docs/RGB_RIGID_VIEW_FIT.md"
LIMITS = np.array([np.deg2rad(3.)]*3+[.03]*3)


def lift(pairs, depth, k, e, scale):
    points, views = [], []
    for i,j,x,y in pairs:
        a, va = sample(x, depth[i], k[i], e[i,:3,:3])
        b, vb = sample(y, depth[j], k[j], e[j,:3,:3])
        valid = va & vb
        if valid.sum() >= 6:
            points.append(np.stack((a[valid],b[valid]),axis=1)/scale)
            views.append(np.tile((i,j),(valid.sum(),1)))
    if not points:
        return np.empty((0,2,3)), np.empty((0,2),dtype=int)
    return np.concatenate(points), np.concatenate(views)


def fit(folds, origin):
    zero = np.zeros((5,6))
    def residual(params, data):
        p, idx = data
        rotations = Rotation.from_rotvec(params[:,:3]).as_matrix()
        transformed = np.einsum("nmij,nmj->nmi",rotations[idx],p) + (origin+params[:,3:])[idx]
        return transformed[:,0]-transformed[:,1]
    def cost(params, data):
        r = residual(params,data)/.02
        return float(np.mean(np.sqrt(1+r*r)-1))
    edges = set(map(tuple,folds[0][1]))
    connected = {0}
    for _ in range(4):
        connected |= {n for edge in edges if connected.intersection(edge) for n in edge}
    report = dict(counts=[len(v) for _,v in folds], edges=sorted([list(map(int,e)) for e in edges]))
    if len(connected)<5 or not len(folds[1][1]):
        return zero, dict(report, decision="insufficient_support")
    def unpack(x):
        return np.vstack((zero[:1],x.reshape(4,6)))
    bound = np.tile(LIMITS,4)
    solution = least_squares(lambda x:residual(unpack(x),folds[0]).ravel(), np.zeros(24),
                             bounds=(-bound,bound), loss="soft_l1", f_scale=.02)
    if not solution.success:
        raise RuntimeError(solution.message)
    params = unpack(solution.x)
    for fields,limit in ((slice(0,3),LIMITS[0]),(slice(3,6),LIMITS[-1])):
        params[:,fields] /= np.maximum(1,np.linalg.norm(params[:,fields],axis=1)/limit)[:,None]
    before = [cost(zero,data) for data in folds]
    after = [cost(params,data) for data in folds]
    accepted = before[1]>0 and after[1]<=.99*before[1]
    return params if accepted else zero, dict(report, before=before, after=after,
                  proposal=params.tolist(), decision="accepted" if accepted else "validation_veto")


def infer():
    plan = read(tracks.PLAN)
    observations = {(r["camera"]["scene"],r["camera"]["batch"]):r["tracks"] for r in read(tracks.TRACKS/"observations.json")}
    records = {}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(tracks.TRACKS/"redetected_tracks.zip") as archive:
        for scene in SCENES:
            key = scene.split("/")[0]+"_base"
            _, depth,k,e,base = prior.inputs(key,read(prior.PARENT/"predictions.json")["rows"],plan)
            rgb = plan["inputs"][key]
            meta = observations[scene,rgb["batch"]]
            if meta["rgb_sha256"]!=rgb["rgb_sha256"]:
                raise ValueError("RGB identity differs")
            scale = np.median(depth)
            origin = centers(e)
            folds = [lift(fold,depth,k,e,scale) for fold in tracks.load_folds(archive,meta)]
            params,log = fit(folds,origin/scale)
            cloud = prior.temporal.load_points(prior.temporal.OLD_POINTS,base)
            for i,p in enumerate(params):
                if np.any(p):
                    cloud[i] = (cloud[i]-origin[i])@Rotation.from_rotvec(p[:3]).as_matrix().T+origin[i]+scale*p[3:]
            digest = prior.tensor_digest(cloud)
            prior.write_arrays(DATA/"points"/(digest+".npz"),dict(points=cloud))
            records[key] = dict(hash=digest, direct=base, match_hash=meta["matches_sha256"], scale=float(scale), **log)
    return records


def score(item):
    key, row = item
    _, values = prior.evaluate_points(key,{row["hash"]},DATA)
    return key,values[row["hash"]]


def summarize(obs):
    fitted = read(prior.RESULT)["observed"]
    smooth = read(diffusion.RESULT)["observed"]
    return prior.summarize(fitted,obs,extra={"diffused":smooth})


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("infer","evaluate","verify"))
    action = p.parse_args().action
    deps = dict(protocol=hashlib.sha256(DESIGN.read_text().encode()).hexdigest(), plan=sha(tracks.PLAN),
                tracks=sha(tracks.TRACKS/"observations.json"), direct=sha(prior.PARENT/"predictions.json"))
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal = read(SEAL)
        if deps!=seal["inputs"]: raise ValueError("Inputs changed")
        rows = seal["rows"]
        if set(rows)!={s.split("/")[0]+"_base" for s in SCENES}: raise ValueError("Missing scene")
        for row in rows.values():
            if (row["decision"]=="accepted")!=(row["hash"]!=row["direct"]): raise ValueError("Unexpected map")
            if "after" in row and (row["after"][1]<=.99*row["before"][1])!=(row["decision"]=="accepted"):
                raise ValueError("Incorrect validation decision")
        deps = [sha(path) for path in (SEAL,prior.RESULT,diffusion.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                observed = dict(pool.map(score,rows.items()))
            dump(RESULT,dict(inputs=deps,observed=observed,results=summarize(observed)))
        report = read(RESULT)
        if report["inputs"]!=deps or summarize(report["observed"])!=report["results"]: raise ValueError("Scores differ")
        if action=="verify":
            prior.old.verify_map_archive(PUBLIC/"rgb_rigid_view_archive.json",{k:{"rigid":v["hash"]} for k,v in rows.items()})
        print(report["results"]["checks"])
