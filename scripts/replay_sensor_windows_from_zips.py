"""Recompute nine RGB calibration comparisons from original public 7-Scenes ZIPs on a CPU."""
import argparse
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from PIL import Image

from audit_sensor_frame import effective_camera, stats, track_errors
from calibration_audit import NOMINAL_FOCAL, resized_k
from evaluation_stability import dump,sha
from fixed_match_validation import sample_points
from score_saved_geometry import read

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/"results/sensor_public_replay"


@contextmanager
def sequence_archive(folder,scene,sequence):
    small=folder/(scene+"-"+sequence+".zip")
    if small.is_file():
        with zipfile.ZipFile(small) as z:yield z
    else:
        with zipfile.ZipFile(folder/(scene+".zip")) as outer,tempfile.TemporaryFile() as inner:
            with outer.open(scene+"/"+sequence+".zip") as source:
                shutil.copyfileobj(source,inner,1024*1024)
            inner.seek(0)
            with zipfile.ZipFile(inner) as z:yield z


def registered_depth(raw,h):
    if raw.shape!=(480,640):raise ValueError("Expected original Kinect depth raster")
    d=raw.astype(np.float32)/1000.
    y,x=np.where((d>0)&(d<100))
    z=d[y,x]
    sensor=np.stack(((x+.5-320)*z/585.,(y+.5-240)*z/585.,z),axis=1)
    rgb=sensor@h[:3,:3].T+h[:3,3]
    xy=np.rint(rgb[:,:2]/rgb[:,2:3]*525.+[320,240]).astype(int)
    ok=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<480)
    buffer=np.full((480,640),2000.,dtype=np.float32)
    np.minimum.at(buffer,(xy[ok,1],xy[ok,0]),rgb[ok,2].astype(np.float32))
    buffer[buffer>1000.]=0
    # Reproduce the integer-mm intermediate saved by the registration preprocessing.
    mm=(buffer*1000).astype(np.uint16)
    depth=mm.astype(np.float32)/1000.
    depth[(mm==65535)|(depth>10)|(depth<.001)]=0.
    return cv2.resize(depth,(518,392),interpolation=cv2.INTER_NEAREST)


def replay(z,scene,seq,record,transforms,packed):
    prefix=scene+"_"+seq+"_"+str(record["frames"][0])
    content=packed.read(prefix+".npz")
    if hashlib.sha256(content).hexdigest()!=record["matches_sha256"]:raise ValueError("Match file changed")
    with np.load(io.BytesIO(content),allow_pickle=False) as source:
        pairs=[(i,j,source[f"{n}_xy_i"],source[f"{n}_xy_j"]) for n,(i,j) in enumerate(record["pair_order"])]
    cameras,depths=[],[]
    for frame,identities in zip(record["frames"],record["input_file_hashes"]):
        stem=seq+f"/frame-{frame:06d}."
        pose=z.read(stem+"pose.txt")
        if (hashlib.sha256(pose).hexdigest()!=identities["pose.txt"]
                or hashlib.sha256(z.read(stem+"color.png")).hexdigest()!=identities["color.png"]):
            raise ValueError("Public image or pose differs from recorded input")
        c2w=np.loadtxt(io.BytesIO(pose))
        e=np.column_stack((c2w[:3,:3].T,-c2w[:3,:3].T@c2w[:3,3]))
        cameras.append(effective_camera(e))
        with Image.open(io.BytesIO(z.read(stem+"depth.png"))) as im:
            depths.append(registered_depth(np.array(im),transforms["fixed_rgb"]))
    observations=[]
    for i,j,a,b in pairs:
        for src,dst,xy,target in ((i,j,a,b),(j,i,b,a)):
            d=depths[src]
            valid=sample_points((d>1e-4).astype(float)[...,None],xy)[:,0]>=1-1e-8
            observations.append((src,dst,xy,target,valid,sample_points(d[...,None],xy)[:,0]))
    out={}
    for method,h in transforms.items():
        e=h@np.stack(cameras)
        k=resized_k(NOMINAL_FOCAL if method=="released" else 525.,518,392,uniform=method=="released").astype(np.float32)
        sd,errors=[],[]
        for i,j,xy,target,valid,depth in observations:
            s,r=track_errors(xy,target,valid,depth,k,k,e[j]@np.linalg.inv(e[i]))
            sd.append(s)
            errors.append(r)
        out[method]=stats(np.concatenate(sd),np.concatenate(errors))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archives",type=Path,required=True,help="Directory with the three original scene ZIPs or their extracted sequence ZIPs named scene-seq-NN.zip")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    study=read(ROOT/"results/sensor_windows_v16/summary.json")
    physical=np.array(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
    fits=read(ROOT/"results/sensor_extrinsic_v15/seal.json")["fits"]
    rows,errors,counts=[],[],[]
    with zipfile.ZipFile(PUBLIC/"rgb_matches.zip") as packed:
        for label,fit in fits.items():
            scene,seq=label.split("/")
            transforms=dict(released=np.eye(4),intrinsics_only=np.eye(4),fixed_rgb=physical,
                            fitted_rgb=np.array(fit["matrix"]),inverse_control=np.linalg.inv(physical))
            with sequence_archive(args.archives,scene,seq) as z:
                for rec in [r for r in study["rows"] if r["scene"]==label]:
                    metrics=replay(z,scene,seq,rec,transforms,packed)
                    for method,m in metrics.items():
                        primary=rec["metrics"][method]
                        for k in ("track_observations","valid_depth","positive_depth","within_five_pixels"):
                            counts.append(abs(m[k]-primary[k]))
                        for k in ("sampson_median_px","positive_depth_reprojection_median_px"):
                            errors.append(abs(m[k]-primary[k]))
                    rows.append(dict(scene=label,frames=rec["frames"],metrics=metrics))
    if len(rows)!=9 or max(counts)>0 or max(errors)>2e-4:
        raise ValueError(f"Replay mismatch: {len(rows)} windows, maximum absolute median error {max(errors)}, count difference {max(counts)}")
    dump(args.output,dict(rows=rows,maximum_absolute_median_difference_px=max(errors),max_integer_count_difference=max(counts),
                          rgb_matches_archive_sha256=sha(PUBLIC/"rgb_matches.zip"),source_sha256=sha(Path(__file__))))
    print(f"Reproduced {len(rows)*len(transforms)} conditions; maximum median error difference {max(errors):.8f} px")


if __name__=="__main__":main()
