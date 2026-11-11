"""Collect fresh image-conditioned VGGT camera outputs and disjoint RGB tracks."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from calibration_audit import resized_k
from evaluation_stability import dump, sha
from fixed_match_validation import image_digest
from score_saved_geometry import read


ROOT = Path(__file__).resolve().parents[1]
NAME = "focal_zoom_gradient"
PLAN = ROOT / "manifests" / (NAME + ".json")
RAW = ROOT / "reports" / NAME
PUBLIC = ROOT / "results" / NAME
UPSTREAM = ROOT / "third_party/TCO"
SUPPORT = ROOT / "manifests/learned_support_v8.json"
SCALE = 1.12
FRAMES = (0,200,400,600,800)
MODEL_SHA = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
UPSTREAM_REV = "65387333877723cf99086f7f29089379dd50c59a"


def digest_large(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol():
    sources = ["docs/FOCAL_ZOOM_GRADIENT.md", "scripts/focal_zoom_observations.py",
               "scripts/focal_zoom_gradient.py", "scripts/focal_metadata_control.py",
               "scripts/repair_one_camera.py", "scripts/learned_fixed_support.py", "scripts/masked_camera_supervision.py",
               "scripts/audit_masked_camera_gradients.py", "scripts/calibration_audit.py", "scripts/replay_sensor_windows_from_zips.py",
               "manifests/learned_support_v8.json", "manifests/sensor_frame_v14.json", "results/masked_camera_gradients/inputs.json"]
    return dict(study=NAME,sources={n:sha(ROOT/n) for n in sources},scale=SCALE,frames=list(FRAMES),
                model_sha256=MODEL_SHA,tco_commit=UPSTREAM_REV,noise_seeds=[137,941],
                image_batches=33,gradient_reference="original TCO external camera loss, FP32 on CPU",cloud_usd=0)


def check_plan():
    if read(PLAN) != protocol():
        raise ValueError("Frozen experiment changed")
    if subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip() != UPSTREAM_REV:
        raise ValueError("Wrong TCO revision")
    paths = ["constraints.py","tco_loss.py","utils/geometry.py","datasets/utils/cropping.py","base_models/vggt/vggt"]
    diff = subprocess.check_output(["git","-c","core.fileMode=false","-C",str(UPSTREAM),"diff",
                                    "--ignore-cr-at-eol","HEAD","--",*paths])
    if diff:
        raise ValueError("Research-relevant upstream source was modified")


def images_from_zip(archives, scene, expected):
    from PIL import Image
    from replay_sensor_windows_from_zips import sequence_archive
    sys.path.insert(0,str(UPSTREAM))
    from datasets.utils.cropping import resize_image
    label,seq = scene.split("/")
    with sequence_archive(archives,label,seq) as z:
        pixels = []
        for frame in FRAMES:
            with Image.open(io.BytesIO(z.read(f"{seq}/frame-{frame:06d}.color.png"))) as img:
                image = np.asarray(resize_image(img.convert("RGB"),(518,392)),dtype=np.float32) / 255
            pixels.append(image.transpose(2,0,1))
    images = np.stack(pixels)
    if image_digest(images) != expected:
        raise ValueError("RGB pixels differ from the original five-view input: "+scene)
    return images


def batches(images):
    import cv2
    center = resized_k(525.,518,392,uniform=False)[:2,2]
    native = np.ascontiguousarray(images)
    def crop(image, factor):
        m = np.column_stack((np.eye(2)*factor,center*(1-factor)))
        return cv2.warpAffine(image.transpose(1,2,0),m,(518,392),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).transpose(2,0,1)
    base = np.stack([crop(img,SCALE) for img in native])
    yield "base",base
    for view in range(5):
        for label,factor in (("wide",1.),("tight",SCALE**2)):
            batch = base.copy()
            batch[view] = native[view] if factor == 1 else crop(native[view],factor)
            yield f"v{view}_{label}",batch


def camera_model(checkpoint):
    from safetensors import safe_open
    sys.path.insert(0,str(UPSTREAM / "base_models/vggt"))
    from vggt.models.vggt import VGGT
    if digest_large(checkpoint)!=MODEL_SHA:
        raise ValueError("Pretrained weight hash differs")
    model = VGGT(enable_point=False,enable_depth=False,enable_track=False).eval()
    with safe_open(str(checkpoint),framework="pt",device="cpu") as f:
        needed=set(model.state_dict())
        if needed != {k for k in f.keys() if k.startswith(("aggregator.","camera_head."))}:
            raise ValueError("Camera model parameter names differ")
        model.load_state_dict({k:f.get_tensor(k) for k in needed},strict=True)
    return model


def lightglue(folder):
    import cv2
    import importlib.metadata
    spec = read(SUPPORT)
    code = folder / "third_party/LightGlue"
    extra = folder / "reports/learned_support_v8_python"
    weights = folder / "reports/learned_support_v8_weights"
    for name,h in spec["upstream_python_sha256_lf"].items():
        if hashlib.sha256((code/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() != h:
            raise ValueError("Unrecognized LightGlue source")
    for name,h in spec["weight_sha256"].items():
        if sha(weights/"hub/checkpoints"/name)!=h:
            raise ValueError("Unrecognized matcher weights")
    os.environ["TORCH_HOME"] = str(weights)
    sys.path[:0] = [str(code),str(extra)]
    from lightglue import DISK, LightGlue
    versions={k:importlib.metadata.version(k) for k in ("torch","numpy","kornia","kornia_rs")}
    if dict(versions,opencv=cv2.__version__) != spec["runtime"]:
        raise ValueError("Matcher runtime changed")
    return DISK(max_num_keypoints=2000,origin="center").eval(), LightGlue(
        features="disk",flash=False,mp=False,depth_confidence=-1,width_confidence=-1,filter_threshold=.1).eval()


def collect(args):
    import cv2
    import torch
    from learned_fixed_support import build
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261111)
    np.random.seed(20261111)
    cv2.setNumThreads(1)
    scenes = read(SUPPORT)["input_metadata"]["rows"]
    plan_hash = sha(PLAN)
    model = camera_model(args.checkpoint) if args.action == "camera" else None
    matcher = lightglue(args.support) if args.action == "tracks" else None
    for scene in scenes:
        images = images_from_zip(args.archives,scene["scene"],scene["image_sha256"])
        for tag,batch in batches(images):
            folder = RAW / scene["scene"].replace("/","_") / tag
            folder.mkdir(exist_ok=True,parents=True)
            path = folder / (args.action+".json")
            identity=dict(scene=scene["scene"],batch=tag,rgb_sha256=image_digest(batch),manifest_sha256=plan_hash)
            if path.exists():
                rec=read(path)
                if any(rec[k]!=v for k,v in identity.items()) or (matcher and sha(folder/"tracks.npz")!=rec["matches_sha256"]):
                    raise ValueError("Existing observation differs")
                continue
            start = time.monotonic()
            if model is not None:
                with torch.inference_mode():
                    tokens,_ = model.aggregator(torch.from_numpy(batch)[None])
                    enc = model.camera_head(tokens)[-1].cpu().numpy()
                    del tokens
                if enc.shape!=(1,5,9) or not np.isfinite(enc).all():
                    raise ValueError("Invalid image-conditioned camera encoding")
                value = dict(encoding=enc.tolist(),dtype=str(enc.dtype),device="cpu")
            else:
                pairs,metadata=build(batch,*matcher)
                data={f"{n}_{k}":p[k] for n,p in enumerate(pairs) for k in ("xy_i","xy_j","fold")}
                partial=folder/"tracks.partial.npz"
                np.savez_compressed(partial,**data)
                partial.replace(folder/"tracks.npz")
                value=dict(matches_sha256=sha(folder/"tracks.npz"),metadata=metadata,
                           pair_order=[[p["i"],p["j"]] for p in pairs])
            elapsed=time.monotonic()-start
            tmp=path.with_suffix(".partial.json")
            dump(tmp,dict(**identity,**value,elapsed_seconds=elapsed))
            tmp.replace(path)
            print(args.action,scene["scene"],tag,round(elapsed,2),"seconds",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","camera","tracks"))
    p.add_argument("--archives",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--support",type=Path)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    if args.action=="freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN,protocol())
    else:
        check_plan()
        collect(args)


if __name__=="__main__":
    main()
