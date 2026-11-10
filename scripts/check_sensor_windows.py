"""Test locked sensor transforms on disjoint frame indices in the same development sequences."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path

import numpy as np

from audit_sensor_frame import SCENES, effective_camera, stats, track_errors
from calibration_variants import calibration_prior
from evaluation_stability import dump, sha
from fixed_match_validation import image_digest, sample_points
from score_saved_geometry import read

ROOT = Path(__file__).resolve().parents[1]
STUDY = "sensor_windows_v16"
PUBLIC = ROOT/"results"/STUDY
RAW = ROOT/"reports"/STUDY
MANIFEST = ROOT/"manifests"/(STUDY+".json")
OFFSETS = (50,100,150)
METHODS = ("released","intrinsics_only","fixed_rgb","fitted_rgb","inverse_control")


def plan():
    sources = ["scripts/check_sensor_windows.py","scripts/learned_fixed_support.py","scripts/fixed_match_validation.py",
               "scripts/audit_sensor_frame.py","scripts/calibration_variants.py","manifests/learned_support_v8.json",
               "results/sensor_extrinsic_v15/seal.json","manifests/sensor_frame_v14.json"]
    return dict(study=STUDY,sources={p:sha(ROOT/p) for p in sources},scenes=list(SCENES),methods=list(METHODS),
                windows=[[i+offset for i in (0,200,400,600,800)] for offset in OFFSETS],
                analysis="Transfer the frozen two-other-scene transform, physical transform and its inverse to nine five-frame windows with entirely new frame IDs on the same development sequences. No fitting. Include identity with both released and registration K. Use frozen CPU DISK/LightGlue parameters. Record all ten attempted pairs, including insufficient matches; score each retained pair in both directions. Report per-window median Sampson and registered-depth reprojection, positive-depth counts and pair coverage. Average window medians within scenes, then across scenes. Missing windows stay visible. Compare physical vs original and fitted vs physical. This estimates within-sequence consistency, without an independent-scene or dense-accuracy claim.",
                additional_cloud_usd=0)


def evaluate_window(data, pairs, transforms):
    old = np.stack([effective_camera(e) for e in data["extrs"]])
    z = np.einsum("nhwj,nj->nhw",data["pointclouds"],old[:,2,:3])+old[:,2,3,None,None]
    original_k = data["intrs"]
    rgb_k = calibration_prior(original_k,z.shape[2],z.shape[1],"axis_registration")
    results = {}
    for label,h in transforms.items():
        # Match v14's physical reference projection exactly.
        e = h @ old
        k = original_k if label=="released" else rgb_k
        distances, errors = [], []
        for pair in pairs:
            for a,b in (("i","j"),("j","i")):
                i,j = pair[a],pair[b]
                xy,target = pair["xy_"+a],pair["xy_"+b]
                depth = sample_points(z[i,...,None],xy)[:,0]
                valid = sample_points(data["valid_mask"][i,...,None].astype(float),xy)[:,0]>=1-1e-8
                sd,re = track_errors(xy,target,valid,depth,k[i],k[j],e[j]@np.linalg.inv(e[i]))
                distances.append(sd)
                errors.append(re)
        results[label] = stats(np.concatenate(distances),np.concatenate(errors)) if pairs else None
    return results


def run(args):
    import torch
    import cv2
    from run_repeat_calibration import SevenScenes
    from lightglue import DISK,LightGlue
    from learned_fixed_support import build

    spec = read(ROOT/"manifests/learned_support_v8.json")
    versions = {k:importlib.metadata.version(k) for k in ("torch","numpy","kornia","kornia_rs")}
    versions["opencv"] = cv2.__version__
    if versions != spec["runtime"]:
        raise ValueError("Matching runtime changed")
    for name,digest in spec["upstream_python_sha256_lf"].items():
        if hashlib.sha256((args.lightglue/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() != digest:
            raise ValueError("Matcher code changed")
    for name,digest in spec["weight_sha256"].items():
        if sha(args.weights/"hub/checkpoints"/name)!=digest:
            raise ValueError("Matcher weights changed")
    os.environ["TORCH_HOME"] = str(args.weights)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261111)
    np.random.seed(20261111)
    extractor = DISK(max_num_keypoints=2000,origin="center").eval()
    matcher = LightGlue(features="disk",flash=False,mp=False,depth_confidence=-1,width_confidence=-1,filter_threshold=.1).eval()
    dataset = SevenScenes(SEVENSCENES_DIR=str(args.data),split="test",load_img_size=518,cache_file=str(args.cache))
    physical = np.array(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    fits = read(ROOT/"results/sensor_extrinsic_v15/seal.json")["fits"]
    RAW.mkdir(parents=True,exist_ok=True)
    rows = []
    for scene in SCENES:
        for frames in plan()["windows"]:
            directory = RAW/(scene.replace("/","_")+"_"+str(frames[0]))
            directory.mkdir(exist_ok=True)
            output = directory/"window.json"
            if output.exists():
                record = read(output)
                if (record["manifest_sha256"]!=sha(MANIFEST) or record["scene"]!=scene or record["frames"]!=frames
                        or sha(directory/"matches.npz")!=record["matches_sha256"]):
                    raise ValueError("Checkpoint changed")
                rows.append(record)
                continue
            data = dataset.get_data(sequence_name=scene, ids=frames)
            images = data["images"].numpy()
            pairs, metadata = build(images,extractor,matcher)
            matches = {str(n)+"_"+k:p[k] for n,p in enumerate(pairs) for k in ("xy_i","xy_j","fold")}
            np.savez_compressed(directory/"matches.npz",**matches)
            transforms = dict(released=np.eye(4),intrinsics_only=np.eye(4),fixed_rgb=physical,
                              fitted_rgb=np.array(fits[scene]["matrix"]),inverse_control=np.linalg.inv(physical))
            identities = [{ext:sha(args.data/scene/f"frame-{i:06d}.{ext}") for ext in ("color.png","depth.proj.png","pose.txt")}
                          for i in frames]
            record = dict(manifest_sha256=sha(MANIFEST),scene=scene,frames=frames,image_sha256=image_digest(images),
                          input_file_hashes=identities,matches_sha256=sha(directory/"matches.npz"),matcher=metadata,
                          pair_order=[[p["i"],p["j"]] for p in pairs],metrics=evaluate_window(data,pairs,transforms))
            dump(output,record)
            print("Completed",scene,frames,flush=True)
            rows.append(record)
    if len(rows)!=9:
        raise ValueError("Window grid incomplete")
    output = PUBLIC/"summary.json"
    if output.exists():
        raise FileExistsError(output)
    dump(output,dict(manifest_sha256=sha(MANIFEST),runtime=versions,rows=rows,additional_cloud_usd=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","run"))
    for key in ("data","cache","weights","lightglue"):
        p.add_argument("--"+key,type=Path)
    args = p.parse_args()
    if args.action=="freeze":
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        dump(MANIFEST,plan())
    elif read(MANIFEST)!=plan():
        raise ValueError("Protocol changed")
    else:
        run(args)


if __name__=="__main__":
    main()
