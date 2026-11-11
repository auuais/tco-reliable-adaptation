"""Check corresponding depth, normals and camera accuracy on every new prior draw."""
import argparse
from statistics import mean

import numpy as np

import camera_prior_noise_repeats as repeat
import tangent_paired_geometry as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import FRAMES,INDEX,PUBLIC,RAW,SCENES
from score_saved_geometry import read

RESULT=PUBLIC/"prior_noise_paired.json"


def measure(study=repeat, seeds=repeat.SEEDS):
    rows=[]
    fusion=repeat.fusion
    load=fusion.depth.cross.temporal.load_points
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        spec=read(fusion.model.SEAL)["rows"][key]
        rig=np.array(spec["fused"])
        base=load(fusion.model.DATA,spec["hashes"]["joint"])
        scale=read(paired.axial.SCALES)["batches"][key]["scale_to_metres"]
        cameras=read(paired.poses.RESULT)["rows"][scene]
        ref=np.array(cameras["reference"])
        for seed in seeds:
            record=read(study.SEAL)["rows"][key][str(seed)]
            fixed,coupled=[load(study.DATA,record["hashes"][a]) for a in ("fixed","coupled")]
            updated=np.stack([paired.poses.recover_pose(p,q,e) for p,q,e in zip(base,fixed,rig)])
            pose_errors={}
            target_r,target_delta=paired.poses.relative_rays(ref)
            for label,estimate in (("joint",rig),("equal",updated)):
                r,d=paired.poses.relative_rays(estimate)
                angles=paired.poses.Rotation.from_matrix(r@target_r.transpose(0,2,1)).magnitude()
                factor=max(0.,np.sum(d*target_delta)/np.sum(d*d))
                pose_errors[label]=dict(rotation_deg=float(np.rad2deg(angles).mean()),
                                        center_relative_rmse=float(np.linalg.norm(factor*d-target_delta)/np.linalg.norm(target_delta)))
            for i,frame in enumerate(FRAMES):
                path=RAW/"reference"/slug/f"{frame:06d}.npz"
                if sha(path)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]: raise ValueError("Reference differs")
                xyz={a:p[i]@e[i,:3,:3].T+e[i,:3,3] for a,p,e in (("joint",base,rig),("equal",fixed,updated),("coupled",coupled,updated))}
                with np.load(path) as f:
                    stats=paired.frame_score(f["depth"].astype(float),f["k"],f["valid"],xyz,scale)
                rows.append(dict(scene=scene,frame=frame,seed=seed,camera={a:pose_errors[a] for a in ("joint","equal")},**stats))
    return rows


def summarize(rows,seeds=repeat.SEEDS):
    result={}
    for seed in seeds:
        group=[r for r in rows if r["seed"]==seed]
        if [(r["scene"],r["frame"]) for r in group]!=[(s,f) for s in SCENES for f in FRAMES]: raise ValueError("Missing targets")
        for r in group:
            m=r["metrics"]
            np.testing.assert_allclose(m["joint"]["depth_sse"],m["equal"]["depth_sse"],rtol=1e-6)
            np.testing.assert_allclose(m["coupled"]["depth_sse"],m["equal"]["depth_sse"]+2*r["cross"]+r["energy"],rtol=1e-9)
        def avg(items):
            return {a:dict(depth_mse=mean(r["metrics"][a]["depth_sse"]/r["depth_count"] for r in items),
                           normal_deg=mean(r["metrics"][a]["angular_sum"]/r["normal_count"] for r in items),
                           camera_deg=mean(r["camera"]["joint" if a=="joint" else "equal"]["rotation_deg"] for r in items))
                    for a in ("joint","equal","coupled")}
        result[str(seed)]=dict(overall=avg(group),scenes={s:avg([r for r in group if r["scene"]==s]) for s in SCENES})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (repeat.SEAL,paired.poses.RESULT,INDEX,paired.axial.SCALES)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    report=read(RESULT)
    if deps!=report["inputs"] or report["results"]!=summarize(report["rows"]): raise ValueError("Paired measures changed")
    print({s:v["overall"]["coupled"] for s,v in report["results"].items()})
