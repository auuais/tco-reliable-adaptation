"""Apply the original paired calibration evaluator to the separate readout replication."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import evaluate_sensor_tco as experiment
from evaluation_stability import dump,sha
from run_sensor_readout_replication import ROOT,NAME,RAW,MANIFEST
from score_saved_geometry import read

PUBLIC=ROOT/"results"/NAME
ANALYSIS=ROOT/"manifests"/(NAME+"_evaluation.json")


def plan():
    pose_subset=ROOT/"reports/sensor_tco_v17/component_rgb_poses.json"
    return dict(replication_sha256=sha(MANIFEST),paired_evaluator_sha256=sha(ROOT/"scripts/evaluate_sensor_tco.py"),
                component_evaluator_sha256=sha(ROOT/"scripts/diagnose_sensor_components.py"),
                component_protocol_sha256=sha(ROOT/"manifests/sensor_component_metrics.json"),
                pose_subset_sha256=sha(pose_subset),analysis_source_sha256=sha(Path(__file__)),
                protocol="Apply the exact original initial FP32, direct-camera, unchanged TCO and half-depth evaluations to the homogeneous readout replication, with the original two common references. Reuse the predeclared secondary axial-depth and RGB SfM pose diagnostics on the same offset-50 frames. Archive every initialization and engineering bridge. Keep the original interrupted run outside the twelve-cell table.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","prepare","score","group","components","collect","verify"))
    p.add_argument("--scene",choices=experiment.SCENES)
    p.add_argument("--reference",choices=experiment.REFS)
    args=p.parse_args()
    if args.action=="freeze":
        if ANALYSIS.exists():raise FileExistsError(ANALYSIS)
        dump(ANALYSIS,plan())
        return
    if args.action!="verify" and read(ANALYSIS)!=plan():raise ValueError("Evaluation protocol changed")
    # The historical evaluator resolves its manifest, sealed predictions and output
    # directory through these module paths. Use the replication's own provenance.
    experiment.NAME,experiment.MANIFEST,experiment.RAW,experiment.PUBLIC=NAME,MANIFEST,RAW,PUBLIC
    if args.action=="prepare":
        experiment.prepare()
    elif args.action=="group":
        experiment.score(args.scene,args.reference)
    elif args.action=="score":
        def run(pair):
            s,r=pair
            with (RAW/("evaluation_"+s.replace("/","_")+"_"+r+".log")).open("a") as log:
                subprocess.run([sys.executable,__file__,"group","--scene",s,"--reference",r],check=True,timeout=3600,
                               stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run,[(s,r) for r in experiment.REFS for s in experiment.SCENES]))
    elif args.action=="components":
        from diagnose_sensor_components import score
        score(NAME,ROOT/"reports/sensor_tco_v17/component_rgb_poses.json",None)
    elif args.action=="collect":
        result=experiment.summarize([read(f) for f in sorted((RAW/"scores").rglob("evaluation.json"))],read(PUBLIC/"seal.json"))
        if (PUBLIC/"summary.json").exists():raise FileExistsError("Published summary exists")
        dump(PUBLIC/"summary.json",result)
    else:
        seal=read(PUBLIC/"seal.json")
        result=read(PUBLIC/"summary.json")
        if (seal["manifest_sha256"]!=sha(MANIFEST) or len(seal["runs"])!=12
                or result!=experiment.summarize(result["rows"],seal)):raise ValueError("Incomplete verification")
        print("Verified all 12 adaptations and 66 dense evaluations in the separate replication")


if __name__=="__main__":main()
