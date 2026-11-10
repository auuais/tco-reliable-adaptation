"""Audit released 7-Scenes calibration/conventions without changing model inputs.

Physical calibration cannot be certified by projection round-trips. Report the
registration-vs-loader focal discrepancy and the resize mapping separately.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/"third_party/TCO"
SCENES = ["heads/seq-01", "chess/seq-03", "fire/seq-03"]
FRAMES = [0, 200, 400, 600, 800]
NOMINAL_FOCAL = 554.2562584220408
REGISTRATION_RGB_FOCAL = 525.0


def resized_k(focal, width=518, height=392, uniform=False):
    """Pixel-center-aware pinhole K under a specified image affine transform."""
    sx, sy = width/640., height/480.
    if uniform: sx = sy = max(sx,sy)
    return np.array([[focal*sx,0,(320+.5)*sx-.5],
                     [0,focal*sy,(240+.5)*sy-.5],[0,0,1]],dtype=np.float64)


def projection(points, extrinsic, intrinsic):
    camera = points @ extrinsic[:3,:3].T + extrinsic[:3,3]
    uv = camera @ intrinsic.T
    return uv[:,:2]/uv[:,2:]


def mapping_findings(width=518,height=392):
    released=resized_k(NOMINAL_FOCAL,width,height,True)
    axis=resized_k(NOMINAL_FOCAL,width,height)
    registered=resized_k(REGISTRATION_RGB_FOCAL,width,height)
    xy=np.array([[0,0],[width-1,0],[0,height-1],[width-1,height-1],[(width-1)/2,(height-1)/2]],dtype=float)
    rays=np.c_[xy,np.ones(len(xy))] @ np.linalg.inv(axis).T
    remapped=rays @ released.T
    delta=remapped[:,:2]/remapped[:,2:]-xy
    return dict(original_size=[480,640],processed_size=[height,width],
        registration_depth_focal_px=585.,registration_rgb_focal_px=REGISTRATION_RGB_FOCAL,
        loader_nominal_focal_px=NOMINAL_FOCAL,
        nominal_focal_excess_percent=100*(NOMINAL_FOCAL/REGISTRATION_RGB_FOCAL-1),
        actual_image_scale_xy=[width/640.,height/480.],
        released_intrinsics_scale=max(width/640.,height/480.),
        released_k=released.tolist(),axis_aware_nominal_k=axis.tolist(),axis_aware_registration_k=registered.tolist(),
        released_fx_excess_vs_axis_percent=100*(released[0,0]/axis[0,0]-1),
        released_cx_shift_vs_axis_pixels=float(released[0,2]-axis[0,2]),
        image_mapping_probe_pixels=xy.tolist(),mapping_displacement_pixels=delta.tolist(),
        max_probe_mapping_displacement_pixels=float(np.linalg.norm(delta,axis=1).max()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=pathlib.Path,default=pathlib.Path("/data/data/7scenes"))
    parser.add_argument("--dataset-cache",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(), "Preserve audit attempts"
    sys.path.insert(0,str(UPSTREAM))
    from datasets.sevenscenes import SevenScenes
    from utils.geometry import closed_form_inverse_se3
    from PIL import Image
    dataset=SevenScenes(SEVENSCENES_DIR=str(args.data_root),split="test",load_img_size=518,cache_file=str(args.dataset_cache))
    records=[]
    for scene in SCENES:
        data=dataset.get_data(sequence_name=scene,ids=FRAMES)
        assert list(data["images"].shape)==[5,3,392,518]
        before=hashlib.sha256(data["pointclouds"].tobytes()+data["intrs"].tobytes()+data["extrs"].tobytes()).hexdigest()
        for i,frame in enumerate(FRAMES):
            k,e=data["intrs"][i],data["extrs"][i]
            np.testing.assert_allclose(k,resized_k(NOMINAL_FOCAL,uniform=True),atol=5e-5,rtol=0)
            path=args.data_root/scene/f"frame-{frame:06d}.pose.txt"
            pose=np.loadtxt(path)
            np.testing.assert_allclose(e,closed_form_inverse_se3(pose[None])[0][:3,:],atol=1e-10,rtol=0)
            assert .99 < np.linalg.det(pose[:3,:3]) < 1.01
            valid=data["valid_mask"][i];y,x=np.nonzero(valid)
            select=np.linspace(0,len(y)-1,min(2000,len(y)),dtype=int);x,y=x[select],y[select]
            world=data["pointclouds"][i,y,x]
            used_c2w=closed_form_inverse_se3(e[None])[0]
            exact_w2c=np.linalg.inv(used_c2w)
            expected=np.stack([x,y],axis=1)
            exact_error=np.linalg.norm(projection(world,exact_w2c,k)-expected,axis=1)
            nominal_error=np.linalg.norm(projection(world,e,k)-expected,axis=1)
            assert exact_error.max()<.001
            raw_depth=np.asarray(Image.open(args.data_root/scene/f"frame-{frame:06d}.depth.proj.png"))
            assert raw_depth.shape==(480,640) and raw_depth.dtype==np.uint16
            np.testing.assert_allclose(pose[3],[0,0,0,1],atol=1e-8)
            records.append(dict(scene=scene,frame=frame,
                exact_inverse_roundtrip_max_pixels=float(exact_error.max()),
                transpose_inverse_roundtrip_max_pixels=float(nominal_error.max()),
                rotation_orthonormality_max=float(np.max(np.abs(pose[:3,:3].T@pose[:3,:3]-np.eye(3)))),
                rotation_determinant=float(np.linalg.det(pose[:3,:3])),valid_fraction=float(valid.mean()),
                pose_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                registered_depth_sha256=hashlib.sha256((args.data_root/scene/f"frame-{frame:06d}.depth.proj.png").read_bytes()).hexdigest()))
        assert before==hashlib.sha256(data["pointclouds"].tobytes()+data["intrs"].tobytes()+data["extrs"].tobytes()).hexdigest()
    report=dict(status="completed_with_calibration_findings",scope="15 fixed frames in three development sequences; no parameter tuning",
        findings=mapping_findings(),checks=dict(frames=15,loader_k_matches_release=True,pose_convention_matches_release=True,
        exact_inverse_projection_roundtrip_passed=True,evaluation_targets_unchanged=True),frames=records,
        source_sha256={f:hashlib.sha256((UPSTREAM/f).read_bytes()).hexdigest() for f in
            ["datasets/sevenscenes.py","datasets/utils/cropping.py","utils/geometry.py"]},
        audit_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        upstream_commit=subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip(),
        reference_sources=["https://github.com/cvlab-stonybrook/TCO/blob/65387333877723cf99086f7f29089379dd50c59a/datasets/sevenscenes.py",
            "https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py"],
        interpretation=["Registration uses focal 525 but the released loader uses nominal focal 554.256; this is not independently calibrated ground truth.",
            "Image resize is anisotropic after height snaps to a multiple of 14; the released K uses one maximum scale for both axes.",
            "Round-trip consistency verifies mathematical conventions, not physical calibration or absolute depth accuracy.",
            "Keep original calibration and evaluation for this factorial ablation; any corrected-K study needs separate matched controls and reference accounting."])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=report["status"],checks=report["checks"],findings=report["findings"])))


if __name__=="__main__": main()
