"""Check crop evaluation geometry against both dense references and an independent sampler."""
import argparse
from pathlib import Path

import numpy as np

from audit_sensor_frame import effective_camera
from calibration_variants import calibration_prior
from evaluation_stability import dump, sha
from focal_zoom_observations import ROOT, SCALE
from score_saved_geometry import read
from screen_sensor_extrinsic import make_reference
from zoom_dense_reference import zoom_reference


def run():
    import cv2
    from repeatability_capture import array_digest
    sealpath=ROOT/"results/canonical_sensor_factorial/seal.json"
    physical=np.array(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    rows=[]
    for scene,group in read(sealpath)["common"].items():
        with np.load(ROOT/group["inputs"],allow_pickle=False) as f:
            arrays=dict(f)
        if array_digest(arrays)!=group["reference"]:
            raise ValueError("Unexpected registered-depth source")
        old=np.stack([effective_camera(x) for x in arrays["extrinsics"]])
        z=np.einsum("nhwj,nj->nhw",arrays["reference_points"],old[:,2,:3])+old[:,2,3,None,None]
        _,h,w=z.shape
        reg=calibration_prior(arrays["intrinsics"],w,h,"axis_registration")
        for name,extr,intr in (("released",old,arrays["intrinsics"]),("registered_rgb",physical@old,reg)):
            original=make_reference(arrays,scene,name,None)
            for i,(depth,valid,k,e) in enumerate(zip(z,arrays["valid_mask"],intr,extr)):
                center=reg[i,:2,2].astype(float)
                for factor in (1.,SCALE,SCALE**2):
                    out,mask,k2=zoom_reference(depth,valid,k,e,center,factor)
                    # A separate OpenCV warp determines which original depth samples are valid.
                    # At exact half-pixel ties OpenCV's fixed-point nearest rounding can differ.
                    a=np.column_stack((np.eye(2)*factor,(1-factor)*center))
                    cv_mask=cv2.warpAffine(valid.astype(np.uint8),a,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
                    cv_depth=cv2.warpAffine(depth,a,(w,h),flags=cv2.INTER_NEAREST)
                    yy,xx=np.indices((h,w))
                    src=(np.stack((xx,yy),axis=-1)-center)/factor+center
                    stable=np.all(abs(src-np.floor(src)-.5)>.003,axis=-1)
                    if not np.array_equal(mask[stable],cv_mask[stable]):
                        raise ValueError("Mask differs away from nearest-sample rounding boundaries")
                    keep=stable&mask
                    cam=out[keep]@e[:3,:3].T+e[:3,3]
                    pix=cam@k2.T
                    uv=pix[:,:2]/pix[:,2:]
                    pos=np.stack((xx[keep],yy[keep]),axis=1)
                    ray=float(np.max(abs(uv-pos)))
                    axial=float(np.max(abs(cam[:,2]-cv_depth[keep])))
                    identity=float(np.max(abs(out[mask]-original[i][mask]))) if factor==1 else None
                    if ray>1e-6 or axial>1e-6 or (identity is not None and identity>1e-6):
                        raise ValueError("Reprojection, axial depth or identity geometry differs")
                    rows.append(dict(scene=scene,reference=name,view=i,factor=factor,valid_pixels=int(mask.sum()),
                                     independently_checked_pixels=int(keep.sum()),rounding_sensitive_pixels=int((~stable).sum()),
                                     mask_difference_pixels=int((mask!=cv_mask).sum()),max_ray_error_px=ray,
                                     max_axial_error_m=axial,identity_error_m=identity))
    if len(rows)!=90:
        raise ValueError("Incomplete dense-geometry controls")
    return dict(source_seal_sha256=sha(sealpath),implementation_sha256=sha(ROOT/"scripts/zoom_dense_reference.py"),
                probe_sha256=sha(Path(__file__)),rows=rows,cloud_usd=0)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=run()
    dump(args.output,result)
    print("Verified 90 camera/raster configurations for the offline dense evaluator.")
