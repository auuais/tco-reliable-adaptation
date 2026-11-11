"""Score the transported mesh at matching reference pixels using the fixed paired masks."""
import argparse
from statistics import mean
import numpy as np

import camera_mesh_render as mesh
import tangent_paired_geometry as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import FRAMES,INDEX,PUBLIC,RAW,SCENES
from score_saved_geometry import read

RESULT=PUBLIC/"camera_mesh_paired.json"


def measure():
    rows=[]
    load=mesh.fusion.depth.cross.temporal.load_points
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        spec=read(mesh.fusion.model.SEAL)["rows"][key]
        rig=read(paired.poses.RESULT)["rows"][scene]
        scale=read(paired.axial.SCALES)["batches"][key]["scale_to_metres"]
        maps={"joint":(load(mesh.fusion.model.DATA,spec["hashes"]["joint"]),"joint")}
        for arm in mesh.fusion.ARMS:
            for prefix,study in (("fixed",mesh.fusion),("tangent",mesh.tangent),("render",mesh)):
                digest=read(study.SEAL)["rows"][key]["hashes"][arm]
                maps[prefix+"_"+arm]=(load(study.DATA,digest),arm)
        for i,frame in enumerate(FRAMES):
            path=RAW/"reference"/slug/f"{frame:06d}.npz"
            if sha(path)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]: raise ValueError("Reference differs")
            xyz={}
            for label,(cloud,arm) in maps.items():
                e=np.array(rig[arm][i])
                xyz[label]=cloud[i]@e[:3,:3].T+e[:3,3]
            with np.load(path) as ref:
                scores=paired.frame_score(ref["depth"].astype(float),ref["k"],ref["valid"],xyz,scale,
                                          parent="fixed_equal",candidate="render_equal")
            rows.append(dict(scene=scene,frame=frame,**scores))
    return rows


def summarize(rows):
    old=read(paired.RESULT)["rows"]
    if [(r["scene"],r["frame"]) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]: raise ValueError("Missing target")
    for r,v in zip(rows,old):
        if any(r[k]!=v[k] for k in ("scene","frame","depth_count","normal_count")): raise ValueError("Paired masks differ")
        for a,b in (("joint","joint"),("fixed_equal","equal"),("tangent_equal","coupled")):
            if r["metrics"][a]!=v["metrics"][b]: raise ValueError("Comparator differs")
    def group(items):
        return {a:dict(depth_mse=mean(r["metrics"][a]["depth_sse"]/r["depth_count"] for r in items),
                       angle_deg=mean(r["metrics"][a]["angular_sum"]/r["normal_count"] for r in items))
                for a in items[0]["metrics"]}
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r["scene"]==s]) for s in SCENES})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (paired.RESULT,mesh.SEAL,mesh.tangent.SEAL,mesh.fusion.SEAL,paired.poses.RESULT,INDEX,paired.axial.SCALES)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if deps!=data["inputs"] or summarize(data["rows"])!=data["results"]: raise ValueError("Paired replay differs")
    print(data["results"]["overall"])
