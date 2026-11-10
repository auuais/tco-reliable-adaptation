"""Score the sealed staged calibration factorial with the original common references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys

import evaluate_sensor_tco as evaluation
from evaluation_stability import dump, sha
from run_staged_sensor_factorial import PLAN, PUBLIC, RAW, ROOT, STUDY, check_inputs
from score_saved_geometry import read


def group(pair):
    scene, reference = pair
    with (RAW / ("evaluation_" + scene.replace("/", "_") + "_" + reference + ".log")).open("a") as log:
        subprocess.run([sys.executable, __file__, "group", "--scene", scene, "--reference", reference],
                       check=True, timeout=3600, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "score", "group", "components", "collect", "verify"))
    parser.add_argument("--scene", choices=evaluation.SCENES)
    parser.add_argument("--reference", choices=evaluation.REFS)
    args = parser.parse_args()
    evaluation.NAME, evaluation.MANIFEST, evaluation.PUBLIC, evaluation.RAW = STUDY, PLAN, PUBLIC, RAW
    if args.action != "verify":
        check_inputs(read(PLAN))
    if args.action == "prepare":
        evaluation.prepare()
    elif args.action == "group":
        evaluation.score(args.scene, args.reference)
    elif args.action == "score":
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(group, [(s, r) for r in evaluation.REFS for s in evaluation.SCENES]))
    elif args.action == "components":
        from diagnose_sensor_components import score
        subset = ROOT / "reports/sensor_tco_v17/component_rgb_poses.json"
        if sha(subset) != read(ROOT / "manifests/sensor_tco_readout_replication_evaluation.json")["pose_subset_sha256"]:
            raise ValueError("RGB SfM pose subset changed")
        score(STUDY, subset, None)
    elif args.action == "collect":
        rows = [read(p) for p in sorted((RAW / "scores").rglob("evaluation.json"))]
        result = evaluation.summarize(rows, read(PUBLIC / "seal.json"))
        path = PUBLIC / "summary.json"
        if path.exists():
            raise FileExistsError(path)
        dump(path, result)
    else:
        seal, result = (read(PUBLIC / n) for n in ("seal.json", "summary.json"))
        if (seal["manifest_sha256"] != sha(PLAN) or len(seal["runs"]) != 12
                or len(result["rows"]) != 66 or result != evaluation.summarize(result["rows"], seal)):
            raise ValueError("Published factorial did not reproduce")
        print("Verified all 12 full trajectories and 66 dense evaluations")


if __name__ == "__main__":
    main()
