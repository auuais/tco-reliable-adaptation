"""Discriminate single-view focal metadata faults from matched valid image zooms."""
import argparse
import hashlib
from pathlib import Path
from statistics import mean

import numpy as np

from evaluation_stability import dump, sha
from repair_one_camera import prepare_edges, validation_score
from score_saved_geometry import read
from screen_single_camera_repair import CALIBRATION, ROOT, TRACKS, WINDOWS, dataset


NAME = "focal_metadata_control"
PLAN = ROOT / "manifests" / (NAME + ".json")
OUT = ROOT / "results" / NAME / "summary.json"
METHODS = ("retain", "consensus", "first_fold", "both_folds")
FACTORS = (1.05, 1.12, 1.25)
MARGIN = 0.25


def protocol():
    sources = [Path(__file__).resolve(), ROOT / "docs/FOCAL_METADATA_CONTROL.md", TRACKS,
               WINDOWS, CALIBRATION, ROOT / "scripts/screen_single_camera_repair.py",
               ROOT / "scripts/localize_camera_prior_faults.py", ROOT / "scripts/repair_one_camera.py",
               ROOT / "scripts/replay_sensor_windows_from_zips.py", ROOT / "scripts/calibration_audit.py"]
    return dict(factors=list(FACTORS), include_reciprocals=True, margin_pixels=MARGIN,
                windows=9, views=5, cases_per_configuration=2, unchanged_controls=9,
                raster=[518,392], methods=list(METHODS), additional_cloud_usd=0,
                sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources})


def focal_proposal(intrinsics):
    focal = intrinsics[:, [0,1], [0,1]]
    if np.any(focal <= 0) or not np.isfinite(focal).all():
        raise ValueError("Invalid focal length")
    # Both axes share the injected scale. Average their deviations in log space.
    log_focal = np.log(focal)
    deviations = np.abs(log_focal - np.median(log_focal,axis=0)).mean(axis=1)
    winners = np.flatnonzero(np.isclose(deviations,deviations.max(),rtol=0,atol=1e-12))
    if len(winners) != 1:
        return None, None
    view = int(winners[0])
    candidate = intrinsics[view].copy()
    candidate[[0,1],[0,1]] = np.median(np.delete(focal,view,axis=0),axis=0)
    return view, candidate


def matched_tracks(folds, view, center, scale):
    cases = {"metadata_error":[[],[]], "valid_zoom":[[],[]]}
    identity = hashlib.sha256()
    counts = [0,0]
    for side, pairs in enumerate(folds):
        for i,j,a,b in pairs:
            if view not in (i,j):
                continue
            original = a if i == view else b
            zoomed = center + scale * (original.astype(float) - center)
            bounds = np.array([517.,391.])
            keep = ((original >= 0) & (original <= bounds) & (zoomed >= 0) & (zoomed <= bounds)).all(axis=1)
            identity.update(np.array([side,i,j,len(keep)],dtype="<i8").tobytes())
            identity.update(keep.tobytes())
            if keep.sum() < 6:
                continue
            counts[side] += int(keep.sum())
            cases["metadata_error"][side].append((i,j,a[keep],b[keep]))
            za,zb = (zoomed[keep],b[keep]) if i == view else (a[keep],zoomed[keep])
            cases["valid_zoom"][side].append((i,j,za,zb))
    return cases, counts, identity.hexdigest()


def fold_gain(e, k, proposal, view, pairs):
    edges = prepare_edges(e,k,pairs,view)
    if len(edges) < 2:
        return None
    original = validation_score((e[view],k[view]),edges)
    corrected = validation_score((e[view],proposal),edges)
    return float(original-corrected)


def decisions(gains):
    agreed = [v is not None and v >= MARGIN for v in gains]
    return dict(retain=False,consensus=True,first_fold=agreed[0],both_folds=all(agreed))


def run(archives):
    rows, controls = [], []
    for scene,tag,e,k,folds in dataset(archives):
        view, _ = focal_proposal(k)
        controls.append(dict(scene=scene,window=tag,edit=view is not None))
        for injected in range(len(k)):
            for magnitude in FACTORS:
                for scale in (magnitude,1/magnitude):
                    observed = k.copy()
                    observed[injected,[0,1],[0,1]] *= scale
                    view, proposed = focal_proposal(observed)
                    if view != injected:
                        raise ValueError("The unique observed focal outlier was not identified")
                    cases,counts,track_sha = matched_tracks(folds,view,k[view,:2,2],scale)
                    for case, observations in cases.items():
                        gains = [fold_gain(e,observed,proposed,view,pairs) for pairs in observations]
                        rows.append(dict(scene=scene,window=tag,view=view,scale=scale,severity=magnitude,
                                         case=case,matches_per_fold=counts,track_subset_sha256=track_sha,
                                         residual_gain_pixels=gains,edits=decisions(gains)))
    return rows, controls


def group(rows):
    def error(row,method):
        wrong = row["edits"][method] != (row["case"] == "metadata_error")
        return abs(float(np.log(row["scale"]))) if wrong else 0.

    errors = {m:mean(error(r,m) for r in rows) for m in METHODS}
    expected = mean(abs(float(np.log(r["scale"]))) / 2 for r in rows)
    if not all(np.isclose(errors[m],expected,rtol=0,atol=1e-12) for m in ("retain","consensus")):
        raise ValueError("Matched focal-only baseline identity failed")
    cases = {key:[r for r in rows if r["case"] == key] for key in ("metadata_error","valid_zoom")}
    return dict(configurations=len(rows)//2,mean_abs_log_error=errors,
                relative_gain=1-errors["both_folds"]/min(errors["retain"],errors["consensus"]),
                metadata_correction_rate=mean(r["edits"]["both_folds"] for r in cases["metadata_error"]),
                valid_zoom_edit_rate=mean(r["edits"]["both_folds"] for r in cases["valid_zoom"]),
                two_fold_support=mean(all(v is not None for v in r["residual_gain_pixels"]) for r in rows))


def summarize(rows, controls):
    windows = {(r["scene"],r["window"]) for r in controls}
    if len(windows) != 9 or len(controls) != 9:
        raise ValueError("Expected nine unchanged controls")
    expected = {(s,w,i,a,scale,case) for s,w in windows for i in range(5)
                for a in FACTORS for scale in (a,1/a) for case in ("metadata_error","valid_zoom")}
    keys = [(r["scene"],r["window"],r["view"],r["severity"],r["scale"],r["case"]) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("Incomplete matched focal grid")
    supports = {}
    for key,r in zip(keys,rows):
        signature = (r["matches_per_fold"],r["track_subset_sha256"])
        gains = r["residual_gain_pixels"]
        if (len(gains) != 2 or not all(v is None or np.isfinite(v) for v in gains)
                or r["edits"] != decisions(gains) or supports.setdefault(key[:-1],signature) != signature):
            raise ValueError("A decision or matched track subset changed")
    overall = group(rows)
    scenes = {s:group([r for r in rows if r["scene"]==s]) for s in sorted({s for s,_ in windows})}
    severities = {str(a):group([r for r in rows if r["severity"]==a]) for a in FACTORS}
    clean_edits = sum(c["edit"] for c in controls)
    checks = dict(gain=overall["relative_gain"]>=.2, scenes=all(g["relative_gain"]>=0 for g in scenes.values()),
                  valid_zooms=overall["valid_zoom_edit_rate"]<=.05, unchanged=clean_edits==0)
    return dict(overall=overall,scenes=scenes,severities=severities,clean_edits=clean_edits,
                worst_scene_gain=min(g["relative_gain"] for g in scenes.values()),checks=checks,
                motivates_gradient_test=all(checks.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","run","verify"))
    parser.add_argument("--archives",type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN,protocol())
        return
    if read(PLAN) != protocol():
        raise ValueError("Frozen inputs or code changed")
    if args.action == "verify":
        record = read(OUT)
        if record["manifest_sha256"] != sha(PLAN) or record["results"] != summarize(record["rows"],record["controls"]):
            raise ValueError("Published focal-control decision changed")
        print("Verified 270 matched pairs and nine unchanged windows")
        return
    if OUT.exists() or args.archives is None:
        raise ValueError("Run once in a fresh result directory with --archives")
    rows,controls = run(args.archives)
    dump(OUT,dict(manifest_sha256=sha(PLAN),rows=rows,controls=controls,results=summarize(rows,controls)))


if __name__ == "__main__":
    main()
