"""Check visual-calibration decisions and errors from compact public evidence."""
import argparse
import hashlib
import zipfile

import numpy as np

from calibration_audit import resized_k
from evaluation_stability import sha
from focal_zoom_observations import PUBLIC as PARENT, SCALE
from score_saved_geometry import read
from verify_masked_camera_gradients import compare
from visual_focal_baseline import METHODS, PLAN, PUBLIC, estimate, fundamental_pairs, keys, pairs, protocol, summaries


def matrices(r):
    k=np.repeat(resized_k(525.,518,392,uniform=False)[None],5,axis=0)
    k[:,[0,1],[0,1]]*=SCALE
    truth=k.copy()
    v=r["view"]
    if r["case"] in ("metadata_fault","valid_zoom"):
        k[v,[0,1],[0,1]]*=SCALE**r["direction"]
    if r["case"] in ("hidden_zoom","valid_zoom"):
        truth[v,[0,1],[0,1]]*=SCALE**r["direction"]
    return k,truth


def verify(replay):
    if protocol()!=read(PLAN):
        raise ValueError("Solver protocol or source changed")
    record=read(PUBLIC/"summary.json")
    for name,path in (("manifest_sha256",PLAN),("parent_summary_sha256",PARENT/"summary.json"),
                      ("solver_sha256",PUBLIC/"solver_outputs.json"),("track_sha256",PARENT/"redetected_tracks.zip"),
                      ("gradients_sha256",PUBLIC/"gradient_arrays.zip")):
        if record[name]!=sha(path):
            raise ValueError("Solver evidence hash mismatch: "+name)
    obs=read(PARENT/"observations.json")
    parent=read(PARENT/"summary.json")
    first={}
    for r in parent["rows"]:
        first.setdefault((r["scene"],r["batch"]),r)
    solver=read(PUBLIC/"solver_outputs.json")
    adjusted={(r["scene"],r["case"],r["view"],r["direction"]):r for r in solver["calibrations"]}
    if len(solver["calibrations"])!=93 or len(adjusted)!=93 or len(solver["fundamentals"])!=33:
        raise ValueError("Missing camera configuration")
    with zipfile.ZipFile(PARENT/"redetected_tracks.zip") as z:
        if len(z.namelist())!=33 or set(z.namelist())!=set(solver["fundamentals"]):
            raise ValueError("Missing scene track files")
        for o in obs:
            t=o["tracks"]
            key=keys(t["scene"],t["batch"])
            if hashlib.sha256(z.read(key)).hexdigest()!=t["matches_sha256"]:
                raise ValueError("Raw track member changed")
            if replay:
                # Reproduce the exact floating-point coordinates supplied for this batch's first fit.
                k,_=matrices(first[t["scene"],t["batch"]])
                fresh=fundamental_pairs(pairs(z,o),k)
                archived=solver["fundamentals"][key]
                if len(fresh)!=len(archived):
                    raise ValueError("Mismatched observed pair count")
                for a,b in zip(fresh,archived):
                    if any(a[k]!=b[k] for k in ("i","j","matches")) or (a["f"] is None)!=(b["f"] is None):
                        raise ValueError("Visual fundamental estimation inputs differ")
                    if a["f"] is not None:
                        # F is homogeneous. Compare its action after normalization and sign alignment.
                        fa,fb=np.asarray(a["f"]),np.asarray(b["f"])
                        fa=fa/np.linalg.norm(fa)
                        fb=fb/np.linalg.norm(fb)
                        if min(np.linalg.norm(fa-fb),np.linalg.norm(fa+fb))>1e-6:
                            raise ValueError("Refitted fundamental matrix differs")
    if replay:
        for (scene,case,view,direction),meta in adjusted.items():
            k,_=matrices(meta)
            tag="base" if case in ("metadata_fault","uniform_clean") else f"v{view}_"+("tight" if direction==1 else "wide")
            a,b,detail=estimate(solver["fundamentals"][keys(scene,tag)],k)
            for predicted,n in ((a,"continuous_xy"),(b,"snapped_xy")):
                np.testing.assert_allclose(predicted[:,[0,1],[0,1]],meta[n],rtol=1e-8,atol=1e-8)
            if detail["snapped_edit"]!=meta["snapped_edit"] or detail["valid_neighbors"]!=meta["valid_neighbors"]:
                raise ValueError("External solver decision differs")
    with np.load(PARENT/"gradient_arrays.zip",allow_pickle=False) as z:
        original=z["variants"]
    with np.load(PUBLIC/"gradient_arrays.zip",allow_pickle=False) as z:
        gradients=z["variants"]
    if gradients.shape!=(465,2,5,9) or original.shape!=(465,4,5,9) or len(record["rows"])!=465:
        raise ValueError("Missing gradient arrays")
    for r,p,gs,ref in zip(record["rows"],parent["rows"],gradients,original):
        if any(r[k]!=p[k] for k in ("scene","case","view","direction","noise","seed","batch")):
            raise ValueError("Unpaired solver records")
        meta=adjusted[p["scene"],p["case"],p["view"],p["direction"]]
        if meta["snapped_edit"]!=r["solver_snapped_edit"]:
            raise ValueError("Solver edit record changed")
        k,truth=matrices(p)
        target=truth[:,[0,1],[0,1]]
        mean_error=lambda a:float(np.abs(np.log(a/target)).mean())
        for method,existing in p["errors"].items():
            origin=k[:,[0,1],[0,1]].copy()
            if p["edits"][method]:
                v=p["view"]
                origin[v]=np.median(np.delete(origin,v,axis=0),axis=0)
            compare(r["errors"][method],dict(focal_gradient_relative=existing["focal_gradient_relative"],
                    whole_gradient_relative=existing["whole_gradient_relative"],all_camera_log_error=mean_error(origin)))
        conditional,ideal=ref[2:].astype(np.float64)
        for index,method in enumerate(METHODS):
            a=np.asarray(meta[("continuous_xy","snapped_xy")[index]])
            g=gs[index].astype(np.float64)
            np.testing.assert_array_equal(gs[index,:,:7],ref[0,:,:7])
            compare(r["errors"][method],dict(focal_gradient_relative=float(np.linalg.norm(g[:,7:]-conditional[:,7:]))/max(1e-6,float(np.linalg.norm(conditional[:,7:]))),
                    whole_gradient_relative=float(np.linalg.norm(g-ideal))/max(1e-6,float(np.linalg.norm(ideal))),
                    all_camera_log_error=mean_error(a)))
    compare(record["results"],summaries(record["rows"]))
    print("Verified 93 visual solver configurations, 33 sets of image matches and 930 candidate gradients.")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rerun-solver",action="store_true")
    verify(p.parse_args().rerun_solver)
