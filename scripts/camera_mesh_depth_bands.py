"""Attribute rendered log-depth loss changes to observable step-size bands."""
import argparse
import numpy as np
import camera_mesh_paired as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import FRAMES,INDEX,PUBLIC,RAW,SCENES
from score_saved_geometry import read

mesh=paired.mesh
RESULT=PUBLIC/"camera_mesh_depth_bands.json"


def measure():
    rows=[]
    for scene in SCENES:
        slug=scene.split("/")[0]
        key=slug+"_base"
        rigs=read(paired.paired.poses.RESULT)["rows"][scene]
        scale=read(paired.paired.axial.SCALES)["batches"][key]["scale_to_metres"]
        load=mesh.fusion.depth.cross.temporal.load_points
        for arm in mesh.fusion.ARMS:
            z_fields=[]
            for source in (mesh.fusion,mesh):
                h=read(source.SEAL)["rows"][key]["hashes"][arm]
                cloud=load(source.DATA,h)
                e=np.array(rigs[arm])
                z_fields.append(np.einsum("nhwj,nj->nhw",cloud,e[:,2,:3])+e[:,2,3,None,None])
            for i,frame in enumerate(FRAMES):
                path=RAW/"reference"/slug/f"{frame:06d}.npz"
                if sha(path)!=read(INDEX)["frames"][scene][str(frame)]["reference_sha256"]: raise ValueError("Reference differs")
                with np.load(path) as f:
                    valid=f["valid"]
                    truth=f["depth"][valid]
                fixed,render=(z[i][valid] for z in z_fields)
                error=np.log(scale*fixed/truth)
                u=np.log(render/fixed)
                delta=(2*error+u)*u
                bands=np.digitize(np.abs(render/fixed-1),[.01,.05])
                rows.append(dict(scene=scene,frame=frame,arm=arm,n=len(u),base=float(error@error),
                                 counts=np.bincount(bands,minlength=3).tolist(),
                                 changes=np.bincount(bands,weights=delta,minlength=3).tolist()))
    return rows


def summarize(rows):
    expected=read(paired.RESULT)["rows"]
    result={}
    for arm in mesh.fusion.ARMS:
        group=[r for r in rows if r["arm"]==arm]
        if len(group)!=len(expected): raise ValueError("Missing depth bands")
        for r,p in zip(group,expected):
            if (r["scene"],r["frame"],r["n"])!=(p["scene"],p["frame"],p["depth_count"]) or sum(r["counts"])!=r["n"]:
                raise ValueError("Mask or targets changed")
            np.testing.assert_allclose(r["base"]+sum(r["changes"]),p["metrics"]["render_"+arm]["depth_sse"],rtol=1e-9)
        baseline=np.mean([r["base"]/r["n"] for r in group])
        result[arm]=dict(percent_pixels=(100*np.mean([np.array(r["counts"])/r["n"] for r in group],axis=0)).tolist(),
                         percent_baseline_mse=(100*np.mean([np.array(r["changes"])/r["n"] for r in group],axis=0)/baseline).tolist())
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (mesh.SEAL,mesh.fusion.SEAL,paired.RESULT,INDEX)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if data["inputs"]!=deps: raise ValueError("Changed dependency")
    replay=summarize(data["rows"])
    for arm in mesh.fusion.ARMS:
        for metric,values in replay[arm].items(): np.testing.assert_allclose(values,data["results"][arm][metric],atol=1e-10)
    print(replay)
