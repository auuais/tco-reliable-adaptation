"""Check public offset-25 focal decisions, raw feature isolation and analytical gradients."""
import argparse
from pathlib import Path
import sys
import zipfile

import numpy as np

from calibration_audit import resized_k
from collect_relative_focal_windows import OUT, PLAN, SCENES, UPSTREAM, check
from evaluation_stability import sha
from focal_zoom_gradient import LEVELS, errors, noisy_poses
from focal_zoom_observations import SCALE
from measure_focal_crop_response import focal_xy
from relative_focal_confirmation import confirmation
from replay_focal_track_decisions import independent_gain, load_folds
from score_relative_focal_windows import CONFIGS, assess, camera_rig, tag_for
from score_saved_geometry import read
from verify_focal_zoom_derivatives import field_of_view_gradient
from verify_masked_camera_gradients import compare
from visual_focal_baseline import METHODS, estimate, fundamental_pairs, keys, pairs


def targets(case, v, sign):
    base = np.repeat(resized_k(525.,518,392,uniform=False)[None],5,axis=0).astype(float)
    base[:,[0,1],[0,1]] *= SCALE
    supplied, truth = base.copy(), base.copy()
    if case in ("metadata_fault","valid_zoom"):
        supplied[v,[0,1],[0,1]] *= SCALE**sign
    if case in ("hidden_zoom","valid_zoom"):
        truth[v,[0,1],[0,1]] *= SCALE**sign
    return base, supplied, truth


def verify(refit, archives):
    spec = check()
    record = read(OUT/"summary.json")
    if record["manifest_sha256"] != sha(PLAN) or any(sha(OUT/k) != v for k,v in record["inputs"].items()):
        raise ValueError("Changed offset-25 evidence")
    obs = {(o["camera"]["scene"], o["camera"]["batch"]):o for o in read(OUT/"observations.json")}
    solver = read(OUT/"solver_outputs.json")
    fits = {(s["scene"], s["case"], s["view"], s["direction"]):s for s in solver["calibrations"]}
    expected = {(scene, case, v, d, level, seed) for scene in SCENES for case,v,d in CONFIGS
                for level,seed,_,_ in LEVELS}
    axis = [0,1]
    levels = {(n,s):(a,t) for n,s,a,t in LEVELS}
    native = {s:camera_rig(archives,s,spec["inputs"][s]) for s in SCENES} if archives else {}
    with np.load(OUT/"gradient_arrays.zip", allow_pickle=False) as data:
        traces = data["variants"]
    if (traces.shape != (465,6,5,9) or not np.isfinite(traces).all() or len(expected) != 465
            or len(record["rows"]) != 465 or len(obs) != 33 or len(fits) != 93):
        raise ValueError("Incomplete camera, solver or gradient grid")
    first = {}
    for r in record["rows"]:
        first.setdefault((r["scene"],r["batch"]), r)
    support = {}
    with zipfile.ZipFile(OUT/"redetected_tracks.zip") as z:
        if len(z.namelist()) != 33:
            raise ValueError("Expected 33 sets of match coordinates")
        for (scene,tag), o in obs.items():
            support[scene,tag] = load_folds(z, o["tracks"])
            if refit:
                sample = first[scene,tag]
                _, k, _ = targets(sample["case"], sample["view"], sample["direction"])
                fresh = fundamental_pairs(pairs(z,o), k)
                saved = solver["fundamentals"][keys(scene,tag)]
                if len(fresh) != len(saved):
                    raise ValueError("Incorrect pair count")
                for a,b in zip(fresh,saved):
                    if any(a[n] != b[n] for n in ("i","j","matches")) or (a["f"] is None) != (b["f"] is None):
                        raise ValueError("Fundamental matrix fit used different data")
                    if a["f"] is not None:
                        x,y = np.asarray(a["f"]),np.asarray(b["f"])
                        x,y = x/np.linalg.norm(x),y/np.linalg.norm(y)
                        if min(np.linalg.norm(x-y),np.linalg.norm(x+y)) > 1e-6:
                            raise ValueError("Refitted F differs")
    if refit:
        for (scene,case,v,d), s in fits.items():
            _,k,_ = targets(case,v,d)
            a,b,details = estimate(solver["fundamentals"][keys(scene,s["batch"])],k)
            for intr, name in ((a,"continuous"),(b,"snapped")):
                np.testing.assert_allclose(intr[:,axis,axis],s[name],atol=1e-8,rtol=1e-8)
            if details["snapped_edit"] != s["snapped_edit"] or details["valid_neighbors"] != s["valid_neighbors"]:
                raise ValueError("External solver changed decision")
    max_derivative_error = 0.
    for row,g in zip(record["rows"], traces):
        scene,case,v,d,noise,seed = key = tuple(row[n] for n in ("scene","case","view","direction","noise","seed"))
        expected.remove(key)
        tag = tag_for(case,v,d)
        cam = obs[scene,tag]["camera"]
        if tag != row["batch"] or cam["rgb_sha256"] != row["rgb_sha256"]:
            raise ValueError("Wrong image-conditioned encoding")
        base,supplied,truth = targets(case,v,d)
        if archives and case in ("metadata_fault","valid_zoom"):
            e = noisy_poses(native[scene],scene,seed,*levels[noise,seed])
            gains = [independent_gain(p,e,supplied,base[v],v) for p in support[scene,tag]]
            for a,b in zip(gains,row["fold_gains"]):
                if a is None or b is None:
                    if a is not b:
                        raise ValueError("Unexpected missing fold")
                else:
                    np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
        matched = all(x is not None and x >= .25 for x in row["fold_gains"])
        suspect, decisions = confirmation(supplied[:,axis,axis],focal_xy(cam),matched)
        decisions.update(retain=False,consensus=suspect is not None,both_folds=matched)
        if decisions != row["edited"]:
            raise ValueError("Relative focal decision mismatch")
        idx = 0 if v is None else v
        log_errors = [float(np.abs(np.log(k[idx,axis,axis] / truth[idx,axis,axis])).mean()) for k in (supplied,base)]
        compare(row["errors"],errors(g[:4],log_errors,decisions))
        stored = fits[scene,case,v,d]
        choices = [k[:,axis,axis] for k in (supplied,base,truth,truth)] + [stored["continuous"],stored["snapped"]]
        enc = np.asarray(cam["encoding"])[0]
        for grad,focals in zip(g,choices):
            deriv = field_of_view_gradient(enc,focals)
            np.testing.assert_allclose(grad[:,7:],deriv,atol=1e-4,rtol=2e-5)
            max_derivative_error = max(max_derivative_error,float(np.abs(grad[:,7:]-deriv).max()))
        for j in (1,2,4,5):
            np.testing.assert_array_equal(g[0,:,:7],g[j,:,:7])
        correct = g[2,:,7:].astype(float)
        norm = max(1e-6,float(np.linalg.norm(correct)))
        compare(row["classical_gradients"], {m:float(np.linalg.norm(vv[:,7:].astype(float)-correct))/norm
                                             for m,vv in zip(METHODS,g[4:])})
    if expected:
        raise ValueError("Incomplete replication")
    compare(record["results"],assess(record["rows"]))
    print(f"Verified 465 conditions, 2,790 gradient arrays and 27,900 focal derivatives. Max discrepancy {max_derivative_error:.3g}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refit-solver",action="store_true")
    p.add_argument("--archives",type=Path)
    args = p.parse_args()
    sys.path.insert(0,str(UPSTREAM))
    verify(args.refit_solver,args.archives)
