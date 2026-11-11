"""Independently project the new dense references back to their RGB pixels."""
import argparse
from pathlib import Path
import sys

import numpy as np

from audit_sensor_frame import effective_camera
from calibration_audit import NOMINAL_FOCAL, resized_k
from collect_relative_focal_windows import FRAMES, SCENES, UPSTREAM
from evaluation_stability import dump, sha
from focal_zoom_observations import ROOT, SCALE, image_digest
from prepare_dense_focal_pilot import INPUT, PUBLIC, RAW, REF, VIEWS, jobs
from score_saved_geometry import read


def check(assets):
    from utils.geometry import closed_form_inverse_se3
    spec=read(RAW/"inputs.json")
    if spec["jobs"]!=jobs():
        raise ValueError("Unexpected experimental grid")
    h=np.asarray(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    report=[]
    for scene in SCENES:
        label=scene.split("/")[0]
        native=np.stack([np.loadtxt(assets/"data/7scenes"/scene/f"frame-{f:06d}.pose.txt") for f in FRAMES])
        rigid=closed_form_inverse_se3(native)
        old=np.stack([effective_camera(e[:3]) for e in rigid])
        released=resized_k(NOMINAL_FOCAL,518,392,uniform=True).astype(np.float32).astype(float)
        registered=resized_k(525.,518,392,uniform=False)
        center=registered[:2,2]
        priors=[]
        for direction,suffix in ((-1,"wide"),(1,"tight")):
            item=spec["priors"][label+"_"+suffix]
            if sha(INPUT/item["file"])!=item["sha256"]:
                raise ValueError("Prior file changed")
            with np.load(INPUT/item["file"],allow_pickle=False) as f:
                if set(f.files)!={"intrinsics","extrinsics"}:
                    raise ValueError("Unexpected inference priors")
                k,e=f["intrinsics"],f["extrinsics"]
            base=np.repeat(registered[None],5,axis=0)
            base[:,[0,1],[0,1]]*=SCALE
            base[VIEWS[scene],[0,1],[0,1]]*=SCALE**direction
            np.testing.assert_array_equal(k,base.astype(np.float32))
            priors.append(e)
        np.testing.assert_array_equal(*priors)
        for key,item in spec["inputs"].items():
            if item["scene"]!=scene:
                continue
            tag=item["batch"]
            if sha(INPUT/item["file"])!=item["sha256"]:
                raise ValueError("RGB file changed")
            with np.load(INPUT/item["file"],allow_pickle=False) as z:
                if z.files!=["images"] or image_digest(z["images"])!=item["rgb_sha256"]:
                    raise ValueError("Unrecognized input")
            ref=spec["references"][key]
            if sha(REF/ref["file"])!=ref["sha256"]:
                raise ValueError("Reference file changed")
            with np.load(REF/ref["file"],allow_pickle=False) as z:
                mask=z["valid_mask"]
                if int(mask.sum())!=ref["valid_pixels"]:
                    raise ValueError("Unexpected reference mask")
                axes=[]
                max_ray=0.
                for name,k0,e0 in (("registered_rgb",registered,h@old),("released",released,old)):
                    targets=z[name]
                    axial=[]
                    for i in range(5):
                        factor=SCALE
                        if i==VIEWS[scene] and tag!="base":
                            factor=1. if tag.endswith("wide") else SCALE**2
                        y,x=np.nonzero(mask[i])
                        world=targets[i][mask[i]].astype(float)
                        cam=world@e0[i,:3,:3].T+e0[i,:3,3]
                        if np.min(cam[:,2])<=0:
                            raise ValueError("Valid target behind camera")
                        pix=cam@k0.T
                        xy=(pix[:,:2]/pix[:,2:]-center)*factor+center
                        max_ray=max(max_ray,float(np.abs(xy-np.column_stack((x,y))).max()))
                        axial.append(cam[:,2])
                    axes.append(np.concatenate(axial))
                zerr=float(np.abs(axes[0]-axes[1]).max())
                if zerr>1e-5 or max_ray>.005:
                    raise ValueError("Ray or depth correspondence failed")
                report.append(dict(input=key,valid_pixels=int(mask.sum()),max_reprojection_px=max_ray,max_axial_depth_gap_m=zerr))
    if len(report)!=9:
        raise ValueError("Incomplete dense reference check")
    return dict(inputs_sha256=sha(RAW/"inputs.json"),rows=report,reference_clouds=18,
                individual_valid_points_checked=2*sum(r["valid_pixels"] for r in report),
                predictions_or_matching_used=False,additional_cloud_usd=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--write",action="store_true")
    args=p.parse_args()
    sys.path.insert(0,str(UPSTREAM))
    results=check(args.assets)
    dest=PUBLIC/"input_geometry_checks.json"
    if args.write:
        if dest.exists():
            raise FileExistsError(dest)
        dump(dest,results)
    elif read(dest)!=results:
        raise ValueError("Reference geometry evidence differs")
    print("Reprojected",results["individual_valid_points_checked"],"valid world points over both dense references.")
