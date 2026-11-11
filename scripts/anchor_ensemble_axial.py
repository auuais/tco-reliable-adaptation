"""Evaluate corresponding depth for each fixed view order and the median ensemble."""
import argparse
from statistics import mean
import numpy as np

import anchor_ensemble_surface as ensemble
import cross_view_axial as axial
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, PUBLIC, RAW, SCENES
from score_saved_geometry import read

RESULT = PUBLIC/"anchor_ensemble_axial.json"


def measure():
    rows = []
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        sealed = read(ensemble.SEAL)["rows"][key]
        _,depth,_,poses,_ = ensemble.prior.inputs(key,read(ensemble.prior.PARENT/"predictions.json")["rows"],read(ensemble.prior.PLAN))
        scale = read(axial.SCALES)["batches"][key]["scale_to_metres"]
        fields = dict(direct=depth)
        for arm,h in sealed["hashes"].items():
            points = ensemble.cross.temporal.load_points(ensemble.DATA,h)
            fields[arm] = np.einsum("nhwj,nj->nhw",points,poses[:,2,:3])+poses[:,2,3,None,None]
        for i,frame in enumerate(FRAMES):
            path = RAW/"reference"/scene.split("/")[0]/f"{frame:06d}.npz"
            if sha(path)!=read(axial.model.INDEX)["frames"][scene][str(frame)]["reference_sha256"]:
                raise ValueError("Reference changed")
            with np.load(path) as ref:
                mask=ref["valid"]
                truth=ref["depth"][mask]
            sse={arm:float(np.sum(np.log(scale*z[i][mask]/truth)**2)) for arm,z in fields.items()}
            rows.append(dict(scene=scene,frame=frame,sse=sse,valid=int(mask.sum())))
    return rows


def summarize(rows):
    if [(r["scene"],r["frame"]) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError("Wrong targets")
    def group(items):
        return {arm:mean(1-r["sse"][arm]/r["sse"]["direct"] for r in items) for arm in ensemble.ARMS}
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r["scene"]==s]) for s in SCENES})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(p) for p in (ensemble.SEAL,axial.SCALES,axial.model.INDEX)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data=read(RESULT)
    if deps!=data["inputs"] or data["results"]!=summarize(data["rows"]): raise ValueError("Axial evidence differs")
    print(data["results"]["overall"])
