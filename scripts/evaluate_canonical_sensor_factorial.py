"""Seal all full-GPU trajectories before scoring the calibration contrasts locally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys

import evaluate_sensor_tco as scoring
from evaluation_stability import dump, sha
from run_canonical_sensor_factorial import PLAN, PUBLIC, RAW, ROOT, STUDY, check_record, read


def summary(rows, seal):
    result = scoring.summarize(rows, seal)
    result["local_scoring_cloud_usd"] = result.pop("additional_cloud_usd")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare","score","group","components","collect","verify"))
    parser.add_argument("--scene", choices=scoring.SCENES)
    parser.add_argument("--reference", choices=scoring.REFS)
    args = parser.parse_args()
    scoring.NAME, scoring.MANIFEST, scoring.RAW, scoring.PUBLIC = STUDY, PLAN, RAW, PUBLIC
    spec = read(PLAN)
    if any(sha(ROOT / name) != digest for name, digest in spec["sources"].items()):
        raise ValueError("Canonical evaluation or protocol source changed")
    if args.action == "prepare":
        identities = set()
        for job in spec["jobs"]:
            folder, _ = scoring.completed(job)
            record = check_record(folder,spec,job)
            identities.add((record["gpu_identity"],record["research_commit"],record["gsplat_backend"]["sha256"]))
        if len(identities) != 1:
            raise ValueError("All 12 conditions must share hardware, renderer and research revision")
        scoring.prepare()
    elif args.action == "group":
        scoring.score(args.scene, args.reference)
    elif args.action in ("score","components","collect"):
        # Only score after the seal is present in the published origin/main.
        rel = (PUBLIC / "seal.json").relative_to(ROOT).as_posix()
        published = subprocess.check_output(["git","-C",str(ROOT),"show","origin/main:" + rel])
        if published != (PUBLIC / "seal.json").read_bytes():
            raise ValueError("Commit and push the exact prediction seal before scoring")
        if args.action == "score":
            def score_pair(pair):
                scene, ref = pair
                with (RAW / (scene.replace("/","_") + "_" + ref + ".log")).open("a") as log:
                    subprocess.run([sys.executable,__file__,"group","--scene",scene,"--reference",ref],
                                   check=True,timeout=3600,stdout=log,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2"))
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(score_pair,[(s,r) for r in scoring.REFS for s in scoring.SCENES]))
        elif args.action == "components":
            from diagnose_sensor_components import score
            subset = ROOT / "reports/sensor_tco_v17/component_rgb_poses.json"
            if sha(subset) != read(ROOT / "manifests/sensor_tco_readout_replication_evaluation.json")["pose_subset_sha256"]:
                raise ValueError("RGB SfM pose subset changed")
            score(STUDY,subset,None)
        else:
            path = PUBLIC / "summary.json"
            if path.exists():
                raise FileExistsError(path)
            rows = [read(f) for f in sorted((RAW / "scores").rglob("evaluation.json"))]
            dump(path, summary(rows, read(PUBLIC / "seal.json")))
    else:
        result,seal = (read(PUBLIC / name) for name in ("summary.json","seal.json"))
        if (len(result["rows"]) != 66 or len(seal["runs"]) != 12 or seal["manifest_sha256"] != sha(PLAN)
                or result != summary(result["rows"],seal)):
            raise ValueError("Published decision does not reproduce")
        print("Verified the sealed 12-run, 66-score calibration experiment.")


if __name__ == "__main__":
    main()
