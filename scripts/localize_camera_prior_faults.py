"""Can fixed RGB correspondences identify which of five supplied cameras is corrupted?"""
import argparse
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np

from audit_sensor_frame import effective_camera
from calibration_audit import NOMINAL_FOCAL, resized_k
from evaluation_stability import dump, sha
from fit_sensor_extrinsic import edge, signed_distances
from prior_corruptions import axis_rotation
from score_saved_geometry import read


ROOT = Path(__file__).resolve().parents[1]
STUDY = "single_camera_fault_localization"
PLAN = ROOT / "manifests" / (STUDY + ".json")
PUBLIC = ROOT / "results" / STUDY
WINDOWS = ROOT / "results/sensor_windows_v16/summary.json"
TRACKS = ROOT / "results/sensor_public_replay/rgb_matches.zip"
CALIBRATION = ROOT / "manifests/sensor_frame_v14.json"
FAULTS = ("rotation", "translation", "focal")
CONVENTIONS = ("released", "fixed_rgb")
SEEDS = (17, 43, 101)


def protocol():
    sources = [Path(__file__).resolve(), WINDOWS, TRACKS, CALIBRATION, ROOT / "scripts/fit_sensor_extrinsic.py",
               ROOT / "scripts/prior_corruptions.py", ROOT / "scripts/calibration_audit.py",
               ROOT / "scripts/audit_sensor_frame.py", ROOT / "scripts/replay_sensor_windows_from_zips.py"]
    return dict(study=STUDY, sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                seeds=list(SEEDS), faults=list(FAULTS), conventions=list(CONVENTIONS), views=5, windows=9,
                rotation_deg=5, center_offset_fraction_of_window_diameter=.10, focal_factor=1.10,
                design="One corrupted camera at a time, including the anchor. For each of 9 previously used development windows, each of five cameras, three independent seeds and three fault types, evaluate both camera conventions. Use the same deterministic directions for both conventions. Add one clean observation per window and convention. Read only poses and previously frozen 2D RGB matches. No depth, learned model output, SfM reference or 3D metric enters the detector.",
                detector="Median over neighbors of per-edge median absolute Sampson distance in both directions, equal weight per neighbor. A camera requires at least two retained edges. Give equal credit across tied highest-scoring eligible cameras. Unsupported faults count as misses. All five possible fault locations are included, so uniform blind top-one credit is 0.2.",
                alarm="In each convention use the largest clean window score from the other two development scenes as a strict alert threshold. A correct detection requires both the alert and localization. Report clean false alarms, localization, detection, abstention and eligibility, with equal window weight. These cutoffs are descriptive given six calibration windows. The three focal seeds vary the sign only and are not independent images.",
                control="Check noiseless synthetic projections and global translation-scale invariance. Report every fault type, convention and location separately. This is a diagnostic of selective camera supervision, not a new matching algorithm or a reconstruction benchmark. It cannot release transfer funds.", cloud_usd=0)


def inject(cameras, k, view, kind, seed, window):
    e, intr = cameras.copy(), k.copy()
    if kind == "clean":
        return e, intr
    digest = hashlib.sha256(f"{window}:{view}:{kind}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    centers = np.linalg.inv(e)[:, :3, 3]
    center = centers[view].copy()
    if kind == "rotation":
        e[view,:3,:3] = axis_rotation(direction, np.deg2rad(5)) @ e[view,:3,:3]
    elif kind == "translation":
        diameter = np.linalg.norm(centers[:,None] - centers[None,:], axis=-1).max()
        center += .10 * diameter * direction
    elif kind == "focal":
        intr[view,[0,1],[0,1]] *= 1.10 if rng.integers(2) else 1/1.10
        return e, intr
    else:
        raise ValueError(kind)
    e[view,:3,3] = -e[view,:3,:3] @ center
    return e, intr


def score(cameras, k, pairs):
    incident = [[] for _ in cameras]
    pooled = []
    for i,j,xy,target in pairs:
        directions = [edge(cameras,k,i,j,xy,target), edge(cameras,k,j,i,target,xy)]
        distances = np.abs(np.concatenate(signed_distances(np.eye(4), directions)))
        pooled.append(distances)
        median = float(np.median(distances))
        incident[i].append(median)
        incident[j].append(median)
    scores = [float(np.median(v)) if len(v) >= 2 else None for v in incident]
    best = max((v for v in scores if v is not None), default=None)
    winners = [i for i,v in enumerate(scores) if v is not None and np.isclose(v,best,rtol=1e-10,atol=1e-10)]
    return dict(scores=scores, best=best, winners=winners, degree=[len(v) for v in incident],
                pooled_sampson=float(np.median(np.concatenate(pooled))) if pooled else None)


def load_inputs(z, packed, rec):
    scene, seq = rec["scene"].split("/")
    tag = f"{scene}_{seq}_{rec['frames'][0]}"
    raw = packed.read(tag + ".npz")
    if hashlib.sha256(raw).hexdigest() != rec["matches_sha256"]:
        raise ValueError("Image matches changed")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        pairs = [(i,j,data[f"{n}_xy_i"],data[f"{n}_xy_j"]) for n,(i,j) in enumerate(rec["pair_order"])]
    cameras = []
    for frame, checks in zip(rec["frames"], rec["input_file_hashes"]):
        pose = z.read(seq + f"/frame-{frame:06d}.pose.txt")
        if hashlib.sha256(pose).hexdigest() != checks["pose.txt"]:
            raise ValueError("Reference pose file changed")
        c2w = np.loadtxt(io.BytesIO(pose))
        r = c2w[:3,:3].T
        cameras.append(effective_camera(np.column_stack((r, -r @ c2w[:3,3]))))
    return tag, np.stack(cameras), pairs


def summarize(rows):
    clean = [r for r in rows if r["fault"] == "clean"]
    if len(clean) != 18 or len(rows) != 828:
        raise ValueError("Incomplete camera-fault grid")
    keys = {(r["window"],r["convention"],r["fault"],r["view"],r["seed"]) for r in rows}
    tags = {r["scene"].replace("/","_") + "_" + str(r["frames"][0]) for r in read(WINDOWS)["rows"]}
    expected = {(w,c,f,v,s) for w in tags for c in CONVENTIONS for f in FAULTS for v in range(5) for s in SEEDS}
    expected.update((w,c,"clean",None,None) for w in tags for c in CONVENTIONS)
    if keys != expected or len(keys) != len(rows):
        raise ValueError("Missing, unexpected or duplicate camera fault")
    for row in rows:
        calibration = [r["best"] for r in clean if r["convention"]==row["convention"] and r["scene"]!=row["scene"] and r["best"] is not None]
        if len(calibration) != 6:
            raise ValueError("Missing clean calibration window")
        row["threshold_from_other_scenes"] = max(calibration)
        row["alert"] = row["best"] is not None and row["best"] > max(calibration)
        winners, view = row["winners"], row["view"]
        row["credit"] = (1/len(winners) if view in winners else 0.) if view is not None else None
    def rates(group):
        return dict(cases=len(group),top_one=float(np.mean([r["credit"] for r in group])),
                    alerted_and_correct=float(np.mean([r["alert"]*r["credit"] for r in group])),
                    eligible=float(np.mean([r["scores"][r["view"]] is not None for r in group])),
                    alert_rate=float(np.mean([r["alert"] for r in group])))
    tables = {}
    for c in CONVENTIONS:
        cases = [r for r in rows if r["convention"]==c and r["fault"]!="clean"]
        tables[c] = dict(clean_alarms=sum(r["alert"] for r in clean if r["convention"]==c),
                         by_fault={f:rates([r for r in cases if r["fault"]==f]) for f in FAULTS},
                         by_view={str(i):rates([r for r in cases if r["view"]==i]) for i in range(5)},
                         by_fault_and_view={f:{str(i):rates([r for r in cases if r["fault"]==f and r["view"]==i])
                                               for i in range(5)} for f in FAULTS},
                         by_scene={s:rates([r for r in cases if r["scene"]==s]) for s in sorted({r["scene"] for r in cases})})
    return tables


def run(archives):
    from replay_sensor_windows_from_zips import sequence_archive

    windows = read(WINDOWS)["rows"]
    physical = np.array(read(CALIBRATION)["calibration"])
    rows = []
    with zipfile.ZipFile(TRACKS) as packed:
        for label in sorted({r["scene"] for r in windows}):
            scene, seq = label.split("/")
            with sequence_archive(archives, scene, seq) as z:
                for rec in [r for r in windows if r["scene"]==label]:
                    tag, cameras, pairs = load_inputs(z, packed, rec)
                    for convention in CONVENTIONS:
                        released = convention == "released"
                        matrix = resized_k(NOMINAL_FOCAL if released else 525., 518, 392, uniform=released).astype(np.float32)
                        k = np.repeat(matrix[None], len(cameras), axis=0)
                        cases = [("clean", None, None)] + [(f,v,s) for f in FAULTS for v in range(5) for s in SEEDS]
                        for fault, view, seed in cases:
                            e, intr = inject(cameras, k, view, fault, seed, tag)
                            result = score(e if released else physical @ e, intr, pairs)
                            if fault=="clean" and abs(result["pooled_sampson"]-rec["metrics"][convention]["sampson_median_px"])>0.002:
                                raise ValueError("Clean Sampson score disagrees with the sensor audit")
                            rows.append(dict(scene=label, window=tag, convention=convention, fault=fault,
                                             view=view, seed=seed, **result))
    tables = summarize(rows)
    dump(PUBLIC / "summary.json", dict(manifest_sha256=sha(PLAN),rows=rows,tables=tables,additional_cloud_usd=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","run","verify"))
    parser.add_argument("--archives",type=Path)
    args=parser.parse_args()
    if args.action=="freeze":
        if PLAN.exists(): raise FileExistsError(PLAN)
        dump(PLAN,protocol())
    elif read(PLAN)!=protocol():
        raise ValueError("Camera-fault protocol changed")
    elif args.action=="run":
        if (PUBLIC/"summary.json").exists(): raise FileExistsError("Existing results")
        run(args.archives)
    else:
        result=read(PUBLIC/"summary.json")
        rebuilt=[dict(row) for row in result["rows"]]
        if (result["manifest_sha256"]!=sha(PLAN) or result["tables"]!=summarize(rebuilt)
                or result["rows"]!=rebuilt):
            raise ValueError("Published localization summaries changed")
        print("Verified 810 corrupted and 18 clean window conditions.")


if __name__=="__main__":
    main()
