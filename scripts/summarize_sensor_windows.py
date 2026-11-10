"""Equal-window and equal-scene summary of frozen camera transforms."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from evaluation_stability import dump,sha
from score_saved_geometry import read

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/"results/sensor_windows_v16"
PLAN = ROOT/"manifests/sensor_windows_v16.json"
KEYS = ("sampson_median_px","positive_depth_reprojection_median_px")


def summarize():
    p, raw = read(PLAN), read(PUBLIC/"summary.json")
    if raw["manifest_sha256"]!=sha(PLAN) or any(sha(ROOT/f)!=v for f,v in p["sources"].items()):
        raise ValueError("Window manifest changed")
    windows = {(s,tuple(w)) for s in p["scenes"] for w in p["windows"]}
    rows = raw["rows"]
    if len(rows)!=len(windows) or {(r["scene"],tuple(r["frames"])) for r in rows}!=windows:
        raise ValueError("Incomplete window grid")
    attempts,retained,matches = 0,0,0
    for r in rows:
        pairs = r["matcher"]["attempted_pairs"]
        flags = [v for v in pairs if v["retained"]]
        count = sum(v["ransac_inliers"] for v in flags)
        if (len(pairs)!=10 or len(r["input_file_hashes"])!=5 or r["matcher"]["reference_inputs_used"]
                or set(r["metrics"])!=set(p["methods"]) or r["pair_order"]!=[[v["i"],v["j"]] for v in flags]):
            raise ValueError("Image support provenance incomplete")
        for metrics in r["metrics"].values():
            if count:
                if metrics["track_observations"]!=2*count or metrics["positive_depth"]>metrics["valid_depth"]:
                    raise ValueError("Invalid directed counts")
            elif metrics is not None:
                raise ValueError("Empty support received a score")
        attempts+=10
        retained+=len(flags)
        matches+=count
    scores,contrasts = {},{}
    for method in p["methods"]:
        per_scene = {}
        for scene in p["scenes"]:
            group = [r["metrics"][method] for r in rows if r["scene"]==scene]
            per_scene[scene] = {k:mean(m[k] for m in group) if all(m and m[k] is not None for m in group) else None for k in KEYS}
        scores[method] = dict(scenes=per_scene,overall={k:mean(v[k] for v in per_scene.values())
                              if all(v[k] is not None for v in per_scene.values()) else None for k in KEYS})
    for m,b in (("fixed_rgb","released"),("fixed_rgb","intrinsics_only"),("fitted_rgb","fixed_rgb"),("inverse_control","fixed_rgb")):
        contrasts[m+"_vs_"+b] = {}
        for k in KEYS:
            counts = Counter()
            for r in rows:
                left,right = r["metrics"][m],r["metrics"][b]
                if not left or not right or left[k] is None or right[k] is None:
                    counts["missing"]+=1
                else:
                    counts["lower" if left[k]<right[k] else "equal" if left[k]==right[k] else "higher"]+=1
            contrasts[m+"_vs_"+b][k] = dict(counts)
    return dict(summary_sha256=sha(PUBLIC/"summary.json"),attempted_pairs=attempts,retained_pairs=retained,
                matched_correspondences=matches,windows=len(rows),scores=scores,paired_window_counts=contrasts)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--write",action="store_true")
    args=p.parse_args()
    calculated=summarize()
    path=PUBLIC/"aggregates.json"
    if args.write and not path.exists():
        dump(path,calculated)
    if calculated!=read(path):
        raise ValueError("Published window aggregate differs")
    print("Verified",calculated["windows"],"windows and",calculated["matched_correspondences"],"matches")
