"""Measure independent-view surface support and correct depth along the original ray."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

import rgb_rigid_view as rgb
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

prior = rgb.prior
temporal = prior.temporal
weighted = prior.old
diffusion = rgb.diffusion
ARMS = ("direct", "weighted", "diffused", "rgb_rigid")
DATA = RAW/"cross_view_surface"
SEAL = PUBLIC/"cross_view_surface_predictions.json"
RESULT = PUBLIC/"cross_view_surface_scores.json"
DESIGN = ROOT/"docs/FROZEN_CROSS_VIEW_SUPPORT.md"


def sources(points):
    lattices = []
    for cloud in points:
        cloud = cloud.astype(float)
        normals = np.zeros_like(cloud)
        normals[1:-1,1:-1] = np.cross(cloud[1:-1,2:]-cloud[1:-1,:-2], cloud[2:,1:-1]-cloud[:-2,1:-1])
        normals = normals[::4,::4].reshape(-1,3)
        norm = np.linalg.norm(normals,axis=1,keepdims=True)
        normals = np.divide(normals,norm,out=np.zeros_like(normals),where=norm>1e-12)
        pts = cloud[::4,::4].reshape(-1,3)
        lattices.append((cKDTree(pts),pts,normals))
    return lattices


def correct_view(i, maps, lattices, scale, center):
    original = maps["direct"][i]
    query = original.reshape(-1,3).astype(float)
    count = len(query)
    views = [v for v in range(len(lattices)) if v!=i]
    distances, matches = [], []
    for v in views:
        d,idx = lattices[v][0].query(query, distance_upper_bound=.02*scale,workers=2)
        distances.append(d)
        matches.append(idx)
    distances = np.array(distances).T
    closest = np.argsort(distances,axis=1)[:,:2]
    d = np.take_along_axis(distances,closest,axis=1)
    both = np.isfinite(d[:,1])
    ids = np.flatnonzero(both)
    supported_views = np.array(views)[closest[ids]]
    matched = np.take_along_axis(np.array(matches).T[ids],closest[ids],axis=1)
    ray = query[ids]-center
    ray /= np.linalg.norm(ray,axis=1,keepdims=True)
    shifts = np.full((len(ids),2),np.nan)
    grid = np.zeros(original.shape[:2],bool)
    grid[::4,::4] = True
    sampled = grid.ravel()[ids]
    error = dict.fromkeys(ARMS,0.)
    for v in views:
        tree,pts,normals = lattices[v]
        rows,col = np.where(supported_views==v)
        source = matched[rows,col]
        n = normals[source]
        denom = np.sum(n*ray[rows],axis=1)
        use = np.abs(denom)>=.4
        diff = pts[source]-query[ids[rows]]
        shifts[rows[use],col[use]] = np.sum(n[use]*diff[use],axis=1)/denom[use]
        test = ids[rows[sampled[rows]]]
        for arm in ARMS:
            if len(test):
                error[arm] += float(tree.query(maps[arm][i].reshape(-1,3)[test],workers=2)[0].sum()/scale)
    accept = np.isfinite(shifts).all(axis=1) & (np.abs(shifts[:,0]-shifts[:,1])<=.01*scale)
    delta = np.clip(.5*shifts[accept].mean(axis=1),-.02*scale,.02*scale)
    updated = query.copy()
    updated[ids[accept]] += delta[:,None]*ray[accept]
    metrics = dict(pixels=count,accepted=int(accept.sum()),grid=int(grid.sum()),
                   one=int(np.isfinite(d[grid.ravel(),0]).sum()),two=int(sampled.sum()),distance_sums=error)
    return updated.reshape(original.shape).astype(np.float32),metrics


def choose(rows):
    two = sum(r["two"] for r in rows)
    sums = {arm:sum(r["distance_sums"][arm] for r in rows) for arm in ARMS}
    best = min(sums,key=sums.get)
    if two<.05*sum(r["grid"] for r in rows) or sums[best]>=sums["direct"] or sums[best]>.99*sums["direct"]:
        return "direct"
    return best


def infer():
    out = {}
    (DATA/"points").mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        _,depth,_,e,base = prior.inputs(key,read(prior.PARENT/"predictions.json")["rows"],read(prior.PLAN))
        locations = {"direct":(temporal.OLD_POINTS,base),"weighted":(weighted.DATA,read(weighted.SEAL)["hashes"][key]["weighted"]),
                     "diffused":(diffusion.DATA,read(diffusion.SEAL)["hashes"][key]),
                     "rgb_rigid":(rgb.DATA,read(rgb.SEAL)["rows"][key]["hash"])}
        maps = {arm:temporal.load_points(*loc) for arm,loc in locations.items()}
        lattices = sources(maps["direct"])
        views,stats = zip(*(correct_view(i,maps,lattices,float(np.median(depth)),c) for i,c in enumerate(centers(e))))
        points = np.stack(views)
        digest = prior.tensor_digest(points)
        prior.write_arrays(DATA/"points"/(digest+".npz"),dict(points=points))
        out[key] = dict(hash=digest,source_hashes={a:v[1] for a,v in locations.items()},views=stats,choice=choose(stats))
        print(key,out[key]["choice"],sum(r["accepted"] for r in stats),flush=True)
    return out


def score(item):
    key,row = item
    _,v = prior.evaluate_points(key,{row["hash"]},DATA)
    return key,v[row["hash"]]


def summarize(rows,obs):
    raw = {name:read(module.RESULT)["observed"] for name,module in (("diffused",diffusion),("rgb_rigid",rgb))}
    chosen = {}
    previous = temporal.controls()
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        choice = choose(rows[key]["views"])
        if choice!=rows[key]["choice"]: raise ValueError("Selector decision changed")
        if choice=="direct":
            chosen[key] = {ref:previous[scene][ref]["direct"] for ref in temporal.REFS}
        elif choice=="weighted":
            chosen[key] = read(weighted.RESULT)["observed"][key]["weighted"]
        else:
            chosen[key] = raw[choice][key]
    historical = read(prior.RESULT)["observed"]
    return dict(selector=prior.summarize(historical,chosen,extra=raw),
                plane=prior.summarize(historical,obs,extra=dict(raw,selector=chosen)))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("infer","evaluate","verify"))
    action = parser.parse_args().action
    deps = [sha(p) for p in (weighted.SEAL,diffusion.SEAL,rgb.SEAL,prior.PLAN,prior.PARENT/"predictions.json")]
    deps.append(hashlib.sha256(DESIGN.read_text().encode()).hexdigest())
    if action=="infer":
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,rows=infer()))
    else:
        seal = read(SEAL)
        if seal["inputs"]!=deps: raise ValueError("Sealed inputs differ")
        rows = seal["rows"]
        deps = [sha(p) for p in (SEAL,rgb.RESULT,diffusion.RESULT,prior.RESULT,weighted.RESULT)]
        if action=="evaluate":
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                observed = dict(pool.map(score,rows.items()))
            dump(RESULT,dict(inputs=deps,observed=observed,results=summarize(rows,observed)))
        data = read(RESULT)
        if data["inputs"]!=deps or summarize(rows,data["observed"])!=data["results"]:
            raise ValueError("Surface replay failed")
        if action=="verify":
            weighted.verify_map_archive(PUBLIC/"cross_view_surface_archive.json",{k:{"plane":v["hash"]} for k,v in rows.items()})
        print({k:v["checks"] for k,v in data["results"].items()})
