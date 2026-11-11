"""Prepare paired zoom inputs and offline point-map references on the local machine."""
import argparse
from pathlib import Path
import sys

import numpy as np

from audit_sensor_frame import effective_camera
from calibration_audit import resized_k
from collect_relative_focal_windows import FRAMES, PLAN as REPLICATION, ROOT, SCENES, UPSTREAM, check
from evaluation_stability import dump, sha
from fit_sensor_extrinsic import homogeneous
from focal_zoom_gradient import noisy_poses
from focal_zoom_observations import SCALE, batches, image_digest
from score_saved_geometry import read
from zoom_dense_reference import zoom_reference


RAW = ROOT/"reports/dense_focal_pilot"
INPUT = RAW/"inference"
REF = RAW/"evaluation"
VIEWS = dict(zip(SCENES,(0,2,4)))
DESIGN = ROOT/"docs/DENSE_FOCAL_PILOT.md"
PUBLIC = ROOT/"results/dense_focal_pilot"


def write_arrays(path, values):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(values):
                raise ValueError("Existing artifact fields differ: " + str(path))
            for k,v in values.items():
                np.testing.assert_array_equal(z[k],v)
        return
    temp = path.with_suffix(".partial")
    with temp.open("wb") as f:
        np.savez_compressed(f,**values)
    temp.replace(path)


def jobs():
    rows = []
    for seed in (42,123):
        for case in ("metadata_fault","valid_zoom"):
            for direction in (-1,1):
                for round_index in range(3):
                    for j,scene in enumerate(SCENES):
                        arm = ("retain","consensus","adaptive")[(j+round_index+(seed==123))%3]
                        key = scene.split("/")[0] + ("_wide" if direction < 0 else "_tight")
                        view = VIEWS[scene]
                        batch = "base" if case == "metadata_fault" else f"v{view}_" + ("wide" if direction < 0 else "tight")
                        rows.append(dict(scene=scene, case=case, direction=direction, view=view, seed=seed, arm=arm,
                                         input_key=scene.split("/")[0]+"_"+batch, prior_key=key,
                                         run_id=f"{key}_{case}_{arm}_s{seed}"))
    if len(rows) != 72 or len({r["run_id"] for r in rows}) != 72:
        raise ValueError("Incomplete dense grid")
    return rows


def prepare(assets):
    from datasets.sevenscenes import SevenScenes
    spec = check()
    canonical = read(ROOT/"manifests/canonical_sensor_factorial.json")
    cache = ROOT/"environment/development_dataset_cache_v7.npy"
    if sha(cache) != canonical["dataset_cache_sha256"]:
        raise ValueError("Unexpected data cache")
    dataset = SevenScenes(SEVENSCENES_DIR=str(assets/"data/7scenes"), split="test", load_img_size=518,cache_file=str(cache))
    h = np.asarray(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    for path in (INPUT, REF):
        path.mkdir(parents=True,exist_ok=True)
    grids, priors, files, reference_files = {}, {}, {}, {}
    for scene in SCENES:
        name,seq = scene.split("/")
        source_files = spec["inputs"][scene]["files"]
        depth_files = {}
        for frame in FRAMES:
            stem = f"frame-{frame:06d}."
            for ext in ("color.png","pose.txt"):
                if sha(assets/"data/7scenes"/scene/(stem+ext)) != source_files[seq+"/"+stem+ext]:
                    raise ValueError("Prepared RGB or pose differs from public original")
            depth_files[stem+"depth.proj.png"] = sha(assets/"data/7scenes"/scene/(stem+"depth.proj.png"))
        reference_files[scene] = depth_files
        data = dataset.get_data(sequence_name=scene,ids=list(FRAMES))
        native = data["images"].numpy()
        if image_digest(native) != spec["inputs"][scene]["input_rgb_sha256"]:
            raise ValueError("Prepared loader and archive RGB differ")
        released_e = np.stack([effective_camera(x) for x in data["extrs"]])
        old = data["pointclouds"]
        depth = np.einsum("nhwj,nj->nhw",old,released_e[:,2,:3]) + released_e[:,2,3,None,None]
        reg_k = np.repeat(resized_k(525.,518,392,uniform=False)[None],5,axis=0)
        center = reg_k[0,:2,2].copy()
        v = VIEWS[scene]
        noisy = noisy_poses(h@homogeneous(data["extrs"]),scene,137,1.,.02)[:,:3].astype(np.float32)
        for sign, suffix in ((-1,"wide"),(1,"tight")):
            observed = reg_k.copy()
            observed[:,[0,1],[0,1]] *= SCALE
            observed[v,[0,1],[0,1]] *= SCALE**sign
            prior = name+"_"+suffix
            path = INPUT/(prior+"_prior.npz")
            write_arrays(path, dict(extrinsics=noisy,intrinsics=observed.astype(np.float32)))
            priors[prior] = dict(file=path.name,sha256=sha(path))
        for tag,images in batches(native):
            if tag not in ("base",f"v{v}_wide",f"v{v}_tight"):
                continue
            key = name+"_"+tag
            file = INPUT/(key+"_rgb.npz")
            write_arrays(file,dict(images=images))
            grids[key] = dict(file=file.name,sha256=sha(file),rgb_sha256=image_digest(images),scene=scene,batch=tag)
            factors = np.full(5,SCALE)
            if tag != "base":
                factors[v] = 1. if tag.endswith("wide") else SCALE**2
            targets, shared_mask = {}, None
            for label,k,e in (("released", data["intrs"],released_e),("registered_rgb",reg_k,h@released_e)):
                xyz,masks=[],[]
                for i in range(5):
                    points,mask,_ = zoom_reference(depth[i],data["valid_mask"][i],k[i],e[i],center,factors[i])
                    xyz.append(points.astype(np.float32))
                    masks.append(mask)
                masks=np.stack(masks)
                if shared_mask is None:
                    shared_mask=masks
                else:
                    np.testing.assert_array_equal(shared_mask,masks)
                targets[label]=np.stack(xyz)
            targets["valid_mask"]=shared_mask
            if np.any(shared_mask.reshape(5,-1).sum(axis=1) == 0):
                raise ValueError("Empty reference view")
            target=REF/(key+".npz")
            write_arrays(target,targets)
            files[key]=dict(file=target.name,sha256=sha(target),valid_pixels=int(shared_mask.sum()))
    index=dict(protocol_sha256=sha(DESIGN),replication_manifest_sha256=sha(REPLICATION),
               dataset_cache_sha256=sha(cache),inputs=grids,priors=priors,registered_depth_sources=reference_files,
               references=files,jobs=jobs())
    path=RAW/"inputs.json"
    if path.exists() and read(path)!=index:
        raise ValueError("Existing dense input manifest differs")
    dump(path,index)
    print("Verified",len(grids),"image batches,",len(priors),"paired prior rigs and",2*len(files),"dense references.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(UPSTREAM))
    prepare(args.assets)
