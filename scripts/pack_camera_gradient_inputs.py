"""Export three derived camera-head outputs and the verified RGB matches used by the audit."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import torch

from evaluation_stability import dump, sha
from prior_corruptions import tensor_hash
from repeatability_capture import array_digest
from score_saved_geometry import read


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/masked_camera_gradients"


def pack(evidence, archive):
    supports = read(ROOT / "results/learned_support_v8/support.json")["scenes"]
    known = {r["scene"]:r for r in read(ROOT/"results/pointmap_precision_v9/summary.json")["rows"]
             if r["mode"]=="upstream_cpu_float32"}
    if sha(archive) != read(ROOT/"results/learned_support_v8/archive.json")["sha256"]:
        raise ValueError("Learned-match archive changed")
    rows, payloads = [], {}
    with tarfile.open(archive,"r:gz") as tar:
        for scene in supports:
            label=scene["scene"]
            run=evidence/"runs"/("repeat6_"+label.replace("/","_")+"_both_released_seed42_r0")
            meta=read(run/"run.json")
            capture=read(run/"capture.json")["records"]["initial_predictions"]
            with np.load(run/capture["file"],allow_pickle=False) as src:
                heads=dict(src)
            if (sha(run/capture["file"])!=capture["file_sha256"] or array_digest(heads)!=capture["tensor_sha256"]
                    or capture["tensor_sha256"]!=known[label]["initial_heads_tensor_sha256"]):
                raise ValueError("Pretrained camera head capture changed")
            from utils.geometry import extrinsics_to_se3, invert_se3, pose_encoding_to_extri_intri
            raw=torch.from_numpy(heads["pose_enc"]).float()
            e,k=pose_encoding_to_extri_intri(raw,(392,518),build_intrinsics=True)
            decoded={"camera_poses":invert_se3(extrinsics_to_se3(e)).numpy(),"intrinsics":k.numpy()}
            differences={}
            for key,a in decoded.items():
                np.testing.assert_allclose(a,heads[key],atol=1e-4,rtol=2e-6)
                differences[key]=float(np.max(np.abs(a-heads[key])))
            prior=read(run/"prior_metadata.json")
            with np.load(run/"supplied_priors.npz",allow_pickle=False) as f:
                if any(tensor_hash(f[k])!=prior[k+"_sha256"] for k in ("extrinsics","intrinsics")):
                    raise ValueError("Unchanged supplied cameras missing")
                training_sha=tensor_hash(f["extrinsics"].astype(np.float32))
                calibrated=np.asarray(read(ROOT/"manifests/sensor_frame_v14.json")["calibration"])
                lifted=np.broadcast_to(np.eye(4),(5,4,4)).copy()
                lifted[:,:3]=f["extrinsics"]
                calibrated_sha=tensor_hash((calibrated@lifted).astype(np.float32))
            pairs=[]
            for pair in scene["pairs"]:
                name=pair["file"]
                content=tar.extractfile("reports/learned_support_v8/"+name).read()
                if hashlib.sha256(content).hexdigest()!=pair["sha256"]:raise ValueError("Changed match file")
                payloads[name]=content
                pairs.append({k:pair[k] for k in ("i","j","file","sha256")})
            refs={Path(k).name:v for k,v in meta["reference_sha256"].items() if k.endswith(".pose.txt")}
            rows.append(dict(scene=label,frames=meta["arguments"]["frames"],frame_pose_hashes=refs,
                             initial_heads_tensor_sha256=capture["tensor_sha256"],initial_encoding=raw.numpy().tolist(),
                             supplied_extrinsics_sha256=prior["extrinsics_sha256"],pairs=pairs,
                             supplied_extrinsics_fp32_sha256=training_sha,calibrated_extrinsics_fp32_sha256=calibrated_sha,
                             maximum_cpu_decoding_error=differences))
    PUBLIC.mkdir(parents=True,exist_ok=True)
    import io
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as out:
        for name in sorted(payloads):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            out.writestr(info,payloads[name],compress_type=zipfile.ZIP_DEFLATED)
    packed=PUBLIC/"rgb_matches.zip"
    if packed.exists():
        if packed.read_bytes()!=buffer.getvalue():raise ValueError("Existing feature archive differs")
    else:
        with packed.open("xb") as out:out.write(buffer.getvalue())
    dump(PUBLIC/"inputs.json",dict(scenes=rows,matches_sha256=sha(PUBLIC/"rgb_matches.zip"),
                                  description="Derived nine-parameter initial VGGT encodings and RGB match identities for three offset-zero windows. Obtain the supplied camera poses from the public 7-Scenes ZIPs."))


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(ROOT/"third_party/TCO"))
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--support",type=Path,required=True)
    args=p.parse_args()
    pack(args.evidence,args.support)
