"""Prospective, gated Hypersim bridge and independent scene intake for the two-track study."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

import numpy as np
from PIL import Image
from tco_hypersim_geometry import BridgeSupportError, camera_from_metadata, convert_frame, world_to_camera
from tco_hypersim_range import SceneArchive
from tco_two_track_protocol import MANIFEST, PUBLIC, ROOT, STUDY, read, seal


POLICY=MANIFEST.with_name(STUDY+"_hypersim.json")
PREP=Path("E:/ResearchData/TCO/preparation")/STUDY
CACHE=PREP/"hypersim_publisher"
SPLIT="evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
CALIB="contrib/mikeroberts3000/metadata_camera_parameters.csv"
URLS="code/python/tools/dataset_download_images.py"


def sha(file):
    h=hashlib.sha256()
    with Path(file).open("rb") as f:
        for data in iter(lambda:f.read(1024**2),b""):
            h.update(data)
    return h.hexdigest()


def array_hash(arrays):
    h=hashlib.sha256()
    for name,value in sorted(arrays.items()):
        a=np.ascontiguousarray(value)
        h.update(repr((name,a.shape,a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def metadata(policy,folder):
    folder.mkdir(parents=True,exist_ok=True)
    files={}
    for key,expected in policy["metadata"].items():
        path=folder/key.replace("/","__")
        if not path.exists():
            url="https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/"+policy["revision"]+"/"+key
            with urlopen(url,timeout=35) as r:
                payload=r.read(4*1024**2+1)
            if hashlib.sha256(payload).hexdigest()!=expected:
                raise ValueError("Unexpected pinned metadata: "+key)
            tmp=path.with_suffix(".partial")
            tmp.write_bytes(payload)
            tmp.replace(path)
        if sha(path)!=expected:
            raise ValueError("Cached metadata hash mismatch")
        files[key]=path
    with files[CALIB].open() as f:
        cameras={r["scene_name"]:r for r in csv.DictReader(f)}
    partitions={}
    available=defaultdict(lambda:defaultdict(list))
    with files[SPLIT].open() as f:
        for row in csv.DictReader(f):
            if row["included_in_public_release"]!="True":
                continue
            scene,trajectory,split=row["scene_name"],row["camera_name"],row["split_partition_name"]
            if split not in ("train","val","test") or partitions.setdefault(scene,split)!=split:
                raise ValueError("Scene leaks into another publisher split")
            available[scene][trajectory].append(int(row["frame_id"]))
    links=re.findall(r"https://docs-assets\.developer\.apple\.com/ml-research/datasets/hypersim/v1/scenes/(ai_\d{3}_\d{3})\.zip",files[URLS].read_text())
    return cameras,partitions,available,{s:"https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/"+s+".zip" for s in links}


def contrast_gate():
    gate=PUBLIC/"a_arkitscenes_fresh_comparator_decision.json"
    if not gate.is_file():
        raise ValueError("Awaiting supported fresh ARKit gate; Hypersim data access is closed")
    analysis=read(gate)["analysis"]
    nomination=read(PUBLIC/"a_nomination.json")["candidate"]
    cov=analysis["coverage"]
    if (not nomination or analysis["point_only"] or not analysis["candidates"][nomination]["equal_step"]["passed"]
            or cov["expected"]!=cov["scored"] or len(analysis["candidates"][nomination]["comparisons"]["Full10"]["complete_rooms"])!=6):
        raise ValueError("No supported six-visit primary improvement")
    return dict(candidate=nomination,primary_gate_sha256=sha(gate))


def check_committed(path):
    key=path.resolve().relative_to(ROOT).as_posix()
    result=subprocess.run(["git","show","HEAD:"+key],cwd=ROOT,capture_output=True)
    if result.returncode or result.stdout.replace(b"\r\n",b"\n")!=path.read_bytes().replace(b"\r\n",b"\n"):
        raise ValueError("Commit the prospective lock before opening dense geometry: "+key)


def exposures(policy):
    q=subprocess.run(["git","grep","-I","-h","-o","-E","ai_[0-9]{3}_[0-9]{3}",
         policy["study_exposure_anchor"],"--","manifests","results","docs"],cwd=ROOT,text=True,capture_output=True)
    if q.returncode not in (0,1):
        raise ValueError("Unable to verify prior research exposures: "+q.stderr)
    return set(q.stdout.split())


def hdf(payload):
    import h5py
    with h5py.File(io.BytesIO(payload),"r") as f:
        return f["dataset"][:]


def member_names(scene,trajectory,frame,training):
    name=f"frame.{frame:04d}"
    root=scene+"/images/scene_"+trajectory
    geo=root+"_geometry_hdf5/"+name
    paths={"rgb":root+"_final_preview/"+name+".tonemap.jpg","range":geo+".depth_meters.hdf5"}
    if training:
        paths["position"]=geo+".position.hdf5"
    return paths


def camera_track(archive,scene,trajectory):
    root=scene+"/_detail/"
    sc,scale_hash=archive.member(root+"metadata_scene.csv")
    rows=csv.DictReader(io.StringIO(sc.decode("utf-8-sig")))
    unit=float(next(r["parameter_value"] for r in rows if r["parameter_name"]=="meters_per_asset_unit"))
    arrays={}
    hashes={"metadata_scene":scale_hash}
    for key in ("frame_indices","positions","orientations"):
        data,h=archive.member(root+trajectory+"/camera_keyframe_"+key+".hdf5")
        arrays[key]=hdf(data)
        hashes[key]=h
    ids=arrays["frame_indices"]
    if (ids.dtype.kind not in "iu" or ids.ndim!=1 or len(np.unique(ids))!=len(ids)
            or len(arrays["positions"])!=len(ids) or len(arrays["orientations"])!=len(ids)
            or not np.isfinite(unit) or unit<=0):
        raise ValueError("Publisher pose arrays disagree")
    poses={int(n):world_to_camera(R,C,unit) for n,C,R in zip(ids,arrays["positions"],arrays["orientations"])}
    return poses,hashes,unit


def window(frame_ids,poses):
    valid=set(frame_ids)&set(poses)
    for first in sorted(valid):
        for stride in (1,2,4,8):
            frames=[first+i*stride for i in range(5)]
            if not all(f in valid for f in frames):
                continue
            centres=np.stack([-poses[f][:,:3].T@poses[f][:,3] for f in frames])
            baseline=float(np.linalg.norm(centres[-1]-centres[0]))
            if .05<=baseline<=.75 and (np.linalg.norm(np.diff(centres,axis=0),axis=1)>1e-6).all():
                return frames,baseline
    return None,None


def lock(split):
    gate=contrast_gate()
    check_committed(POLICY)
    meta=read(POLICY)
    cameras,partitions,available,urls=metadata(meta,CACHE/"metadata")
    denied=exposures(meta)
    extra=PUBLIC/"hypersim_disclosed_exposures.json"
    if extra.is_file():
        check_committed(extra)
        denied.update(read(extra)["scenes"])
    if split=="test":
        bridge=PUBLIC/"hypersim_bridge.json"
        if not read(bridge)["passed"] or read(bridge)["primary_gate_sha256"]!=gate["primary_gate_sha256"]:
            raise ValueError("Fixed training bridge did not pass")
        denied.add(read(bridge)["scene"])
    target=MANIFEST.with_name(STUDY+"_hypersim_"+split+"_windows.json")
    if target.exists():
        old=read(target)
        if old["primary_gate_sha256"]!=gate["primary_gate_sha256"] or old["policy_sha256"]!=sha(POLICY):
            raise ValueError("Existing lock uses a different gate")
        return
    choices,skipped=[],[]
    for scene in sorted(s for s in available if partitions[s]==split):
        if scene in denied or scene not in urls or scene not in cameras:
            skipped.append(dict(scene=scene,reason="prior exposure or missing source metadata"))
            continue
        try:
            calibration=camera_from_metadata(cameras[scene])
        except ValueError as exc:
            skipped.append(dict(scene=scene,reason="unsupported calibration: "+str(exc)))
            continue
        # Acquiring a ZIP index and camera calibration is sufficient to fix the
        # window. No range/position/image member is read during membership selection.
        with SceneArchive(urls[scene],CACHE/"members") as archive:
            for trajectory,public_ids in sorted(available[scene].items()):
                files={i:member_names(scene,trajectory,i,split=="train") for i in public_ids}
                eligible=[i for i,f in files.items() if set(f.values())<=archive.names]
                if len(eligible)<5:
                    skipped.append(dict(scene=scene,trajectory=trajectory,reason="fewer than five released complete member names"))
                    continue
                try:
                    poses,hashes,unit=camera_track(archive,scene,trajectory)
                except KeyError:
                    skipped.append(dict(scene=scene,trajectory=trajectory,reason="missing named camera metadata"))
                    continue
                if not np.isclose(unit,float(cameras[scene]["settings_units_info_meters_scale"]),rtol=1e-5,atol=1e-10):
                    raise ValueError("Scene and renderer disagree about metres per asset unit")
                frames,baseline=window(eligible,poses)
                if frames is None:
                    skipped.append(dict(scene=scene,trajectory=trajectory,reason="no accepted pose-only window"))
                    continue
                choices.append(dict(scene=scene,trajectory=trajectory,frames=frames,
                    members={str(i):files[i] for i in frames},endpoint_baseline_m=baseline,asset_to_m=unit,
                    poses={str(i):poses[i].tolist() for i in frames},camera_track_sha256=hashes,
                    scene_url=urls[scene],archive_length=archive.remote.length,etag=archive.remote.etag,
                    metadata_range_bytes=archive.remote.received,rectification_zoom=calibration["zoom"],
                    source_projection_max_px=calibration["projection_check_max_px"]))
                break
        if len(choices)==(1 if split=="train" else 6):
            break
    wanted=1 if split=="train" else 6
    if len(choices)!=wanted:
        raise ValueError("Exhausted publisher split with "+str(len(choices))+" eligible windows")
    seal(target,dict(study=STUDY,**gate,split=split,scenes=choices,excluded=sorted(denied),skipped=skipped,
         policy_sha256=sha(POLICY),publisher_revision=meta["revision"],rental_compute_usd=0))
    print(target)


def packet(path,arrays):
    if path.is_file():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError("Existing frozen packet differs")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".partial")
        with tmp.open("wb") as f:
            np.savez_compressed(f,**arrays)
        tmp.replace(path)
    return sha(path)


def materialize(split):
    gate=contrast_gate()
    locked=MANIFEST.with_name(STUDY+"_hypersim_"+split+"_windows.json")
    check_committed(locked)
    roster=read(locked)
    if roster["primary_gate_sha256"]!=gate["primary_gate_sha256"] or roster["policy_sha256"]!=sha(POLICY):
        raise ValueError("Stale scene lock")
    if split=="test" and not read(PUBLIC/"hypersim_bridge.json")["passed"]:
        raise ValueError("Bridge not established")
    cameras,_,_,_=metadata(read(POLICY),CACHE/"metadata")
    rooms=[]
    seen_rgb=set()
    for r in roster["scenes"]:
        scene,track=r["scene"],r["trajectory"]
        hashes,stats=[],[]
        images,clouds,masks,extrs=[],[],[],[]
        cm=camera_from_metadata(cameras[scene])
        with SceneArchive(r["scene_url"],CACHE/"members") as ar:
            if ar.remote.etag!=r["etag"] or ar.remote.length!=r["archive_length"]:
                raise ValueError("Scene archive changed after window lock")
            for uid in r["frames"]:
                ident=str(uid)
                raw={}
                digests={}
                for key,name in r["members"][ident].items():
                    raw[key],digests[key]=ar.member(name)
                with Image.open(io.BytesIO(raw["rgb"])) as im:
                    if im.size!=cm["size"] or im.mode!="RGB":
                        raise ValueError("Unexpected publisher tonemap orientation or channels")
                    rgb=np.array(im)
                extr=np.array(r["poses"][ident])
                try:
                    col,pc,mask,check=convert_frame(rgb,hdf(raw["range"]),cm,extr,
                                hdf(raw["position"]) if split=="train" else None,r["asset_to_m"])
                except BridgeSupportError as exc:
                    seal(PUBLIC/"hypersim_bridge.json",dict(**gate,scene=scene,frames=r["frames"],passed=False,
                        failed_frame=uid,reason=str(exc),checks=stats,source_hashes=hashes+[digests],
                        window_lock_sha256=sha(locked),publisher_revision=roster["publisher_revision"],rental_compute_usd=0))
                    raise
                images.append(col);clouds.append(pc);masks.append(mask);extrs.append(extr.astype(np.float32))
                hashes.append(digests);stats.append(check)
            payload=ar.remote.received
        five_hashes={h["rgb"] for h in hashes}
        if len(five_hashes)!=5 or five_hashes & seen_rgb:
            raise ValueError("Repeated RGB in frozen window")
        seen_rgb.update(five_hashes)
        room="hypersim/"+scene+"/"+track
        if split=="train":
            passed=all(c["bridge_passed"] for c in stats)
            seal(PUBLIC/"hypersim_bridge.json",dict(**gate,scene=scene,room=room,frames=r["frames"],passed=passed,
                 checks=stats,source_hashes=hashes,window_lock_sha256=sha(locked),archive_payload_bytes=payload,
                 publisher_revision=roster["publisher_revision"],rental_compute_usd=0))
            if not passed:
                raise ValueError("Training bridge failed its fixed thresholds")
            return
        slug=room.replace("/","__")
        model=dict(images=np.stack(images),extrs=np.stack(extrs),intrs=np.repeat(cm["K"][None],5,axis=0).astype(np.float32))
        ref=dict(pointclouds=np.stack(clouds),valid_mask=np.stack(masks))
        p=dict(room=room,scene=room,slug=slug,dataset="hypersim",split="test",publisher_scene=scene,
            trajectory=track,frames=r["frames"],raster=list(model["images"].shape[-2:]),
            input_sha256=array_hash(dict(rgb=model["images"],poses=model["extrs"],intrinsics=model["intrs"])),
            reference_sha256=array_hash(dict(surface=ref["pointclouds"],mask=ref["valid_mask"])),
            input_file_sha256=packet(PREP/"model"/(slug+".npz"),model),reference_file_sha256=packet(PREP/"reference"/(slug+".npz"),ref))
        rooms.append(p)
        seal(PUBLIC/("hypersim_"+scene+"_packet.json"),dict(room=p,source_files_sha256=hashes,geometry=stats,
             window_lock_sha256=sha(locked),archive_payload_bytes=payload,etag=r["etag"]))
    if len({r["publisher_scene"] for r in rooms})!=6 or len({r["input_sha256"] for r in rooms})!=6:
        raise ValueError("Expected six disjoint publisher test scenes")
    seal(PUBLIC/"hypersim_fresh_inputs.json",dict(**gate,rooms=rooms,bridge_sha256=sha(PUBLIC/"hypersim_bridge.json"),
         window_lock_sha256=sha(locked),publisher_revision=roster["publisher_revision"],rental_compute_usd=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("metadata","lock-train","bridge","lock-test","prepare-test"))
    args=p.parse_args()
    if args.action=="metadata":
        cm,parts,_,_=metadata(read(POLICY),CACHE/"metadata")
        print("Pinned metadata",len(cm),"camera records;",len(parts),"released scenes")
    elif args.action in ("lock-train","lock-test"):
        lock(args.action.split("-")[1])
    else:
        materialize("train" if args.action=="bridge" else "test")
