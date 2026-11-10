"""Freeze a stronger image-only support baseline before inspecting its outcomes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import numpy as np
from evaluation_stability import dump,sha
from fixed_match_validation import image_digest

ROOT=Path(__file__).resolve().parents[1]
STUDY="learned_support_v8"
SOURCES=["scripts/prepare_learned_support_rgb.py","scripts/learned_fixed_support.py","scripts/cache_learned_support_v8.py",
         "scripts/fixed_match_validation.py","scripts/evaluate_fixed_matches.py","scripts/evaluation_stability.py"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--execution-commit");args=parser.parse_args()
    manifest=ROOT/"manifests"/(STUDY+".json")
    rgb=ROOT/"reports/learned_support_v8_rgb";upstream=ROOT/"third_party/LightGlue"
    weights=ROOT/"reports/learned_support_v8_weights/hub/checkpoints"
    if args.freeze:
        assert not manifest.exists()
        plan=dict(study_id=STUDY,upstream_commit="5a650c444718caffc2669dc0094cde908f42dd73",
            upstream_url="https://github.com/cvg/LightGlue",input_metadata=json.loads((rgb/"inputs.json").read_text()),
            sources={name:sha(ROOT/name) for name in SOURCES},
            upstream_python_sha256_lf={p.relative_to(upstream).as_posix():hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in sorted((upstream/"lightglue").glob("*.py"))},
            weight_sha256={p.name:sha(p) for p in sorted(weights.glob("*.pth"))},
            runtime=dict(torch="2.7.1+cu128",numpy="2.1.2",kornia="0.8.2",kornia_rs="0.1.10",opencv="4.11.0"),
            extractor=dict(name="DISK",weights="depth",max_num_keypoints=2000,origin="center",resize=None),
            matcher=dict(name="LightGlue",features="disk",flash=False,mp=False,depth_confidence=-1,width_confidence=-1,filter_threshold=.1),
            support="All 10 pairs per scene; same F-RANSAC 1.5px/.999/5000 and >=20 inliers as v6. Reject out-of-image coordinates without clipping.",
            folds="Same connected-component hash parity with half-pixel location identity; no shared component between folds",
            eligibility="Both folds connect all 5 views with >=100 matches each and >=10 matches per counted edge; unchanged from v6",
            next_analysis="Freeze all proxy scores before joining all 40 v6 reference outcomes; unchanged 2% both-fold improvement rule; no threshold fitting",
            limitations="A conventional matcher baseline, not a proposed contribution. v6 labels were already inspected before this redesign; no held-out validity claim.",
            device="cpu",cpu_threads=4,runpod_spend_usd=0)
        assert len(plan["weight_sha256"])==2;dump(manifest,plan)
    plan=json.loads(manifest.read_text())
    for name,digest in plan["sources"].items():assert sha(ROOT/name)==digest,name
    if args.freeze or args.dry_run:
        print(json.dumps(dict(study_id=STUDY,manifest_sha256=sha(manifest),scenes=3,pairs=30)));return
    assert re.fullmatch("[0-9a-f]{40}",args.execution_commit or ""),"Provide the clean host checkout commit"
    for name,digest in plan["upstream_python_sha256_lf"].items():
        assert hashlib.sha256((upstream/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==digest,name
    for name,digest in plan["weight_sha256"].items():assert sha(weights/name)==digest,name
    os.environ["TORCH_HOME"]=str(weights.parent.parent)
    import torch
    import cv2
    from lightglue import DISK,LightGlue
    from learned_fixed_support import build
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20261111);np.random.seed(20261111)
    versions={key:importlib.metadata.version(key) for key in ["torch","numpy","kornia","kornia_rs"]};versions["opencv"]=cv2.__version__
    assert versions==plan["runtime"],versions
    output=ROOT/"reports"/STUDY;output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();extractor=DISK(max_num_keypoints=2000,origin="center").eval()
    matcher=LightGlue(features="disk",flash=False,mp=False,depth_confidence=-1,width_confidence=-1,filter_threshold=.1).eval()
    load_seconds=time.perf_counter()-started;summaries=[]
    dump(output/"plan.json",dict(manifest_sha256=sha(manifest),execution_commit=args.execution_commit,versions=versions))
    for item in plan["input_metadata"]["rows"]:
        path=rgb/item["file"];assert sha(path)==item["file_sha256"]
        images=np.load(path,allow_pickle=False);assert image_digest(images)==item["image_sha256"]
        before=time.perf_counter();pairs,meta=build(images,extractor,matcher);entries=[];key=item["scene"].replace("/","_")
        for index,pair in enumerate(pairs):
            name=f"{key}_pair{index}.npz";np.savez_compressed(output/name,xy_i=pair["xy_i"],xy_j=pair["xy_j"],fold=pair["fold"])
            entries.append(dict(i=pair["i"],j=pair["j"],file=name,sha256=sha(output/name),matches=len(pair["fold"]),fold_counts=[int(np.sum(pair["fold"]==f)) for f in [0,1]]))
        meta.update(scene=item["scene"],image_sha256=item["image_sha256"],pairs=entries,matching_seconds=time.perf_counter()-before)
        dump(output/(key+".json"),meta);summaries.append(meta)
        print(json.dumps(dict(scene=item["scene"],pairs=len(pairs),folds=meta["folds"],seconds=meta["matching_seconds"])),flush=True)
    result=dict(study_id=STUDY,execution_commit=args.execution_commit,manifest_sha256=sha(manifest),scenes=summaries,
                model_load_seconds=load_seconds,total_seconds=time.perf_counter()-started,device="cpu",runtime=versions,
                reference_geometry_used=False,runpod_spend_usd=0,timing_note="Concurrent local work; no controlled speed comparison")
    destination=ROOT/"results"/STUDY;destination.mkdir(parents=True,exist_ok=False);dump(destination/"support.json",result)


if __name__=="__main__":main()
