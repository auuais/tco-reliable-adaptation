"""Quantify the axial error available along the sealed two-plane displacement."""
import argparse
from statistics import mean

import numpy as np

import cross_view_surface as surface
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, PUBLIC, RAW, SCENES
from score_saved_geometry import read

model = surface.weighted.model
RESULT = PUBLIC/"cross_view_axial.json"
SCALES = model.prior.PRIOR/"source_transplant.json"


def measure():
    rows = []
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        row = read(surface.SEAL)["rows"][key]
        before = surface.temporal.load_points(surface.temporal.OLD_POINTS,row["source_hashes"]["direct"])
        after = surface.temporal.load_points(surface.DATA,row["hash"])
        _,_,_,poses,_ = surface.prior.inputs(key,read(surface.prior.PARENT/"predictions.json")["rows"],read(surface.prior.PLAN))
        scale = read(SCALES)["batches"][key]["scale_to_metres"]
        for i,frame in enumerate(FRAMES):
            ref = RAW/"reference"/scene.split("/")[0]/f"{frame:06d}.npz"
            if sha(ref)!=read(model.INDEX)["frames"][scene][str(frame)]["reference_sha256"]:
                raise ValueError("Reference differs")
            with np.load(ref) as f:
                mask = f["valid"]
                target = f["depth"][mask]
            def axial(points):
                return points[i][mask]@poses[i,2,:3]+poses[i,2,3]
            z = axial(before)
            error = np.log(scale*z/target)
            step = np.log(axial(after)/z)
            rows.append(dict(scene=scene,frame=frame,base=float(error@error),cross=float(error@step),
                             energy=float(step@step),pixels=len(z),changed=int(np.count_nonzero(step))))
    return rows


def summarize(rows):
    if [(r["scene"],r["frame"]) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError("Wrong target set")
    def gain(r,alpha):
        return -(2*alpha*r["cross"]+alpha**2*r["energy"])/r["base"]
    def group(items):
        e = mean(r["energy"]/r["base"] for r in items)
        c = mean(r["cross"]/r["base"] for r in items)
        optimum = float(np.clip(-c/e,0,1)) if e>0 else 0.
        return dict(applied_gain=mean(gain(r,1.) for r in items),oracle_weight=optimum,
                    oracle_gain=mean(gain(r,optimum) for r in items))
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r["scene"]==s]) for s in SCENES})


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps = [sha(x) for x in (surface.SEAL,model.INDEX,SCALES)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows = measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data = read(RESULT)
    if data["inputs"]!=deps or data["results"]!=summarize(data["rows"]): raise ValueError("Axial comparison differs")
    print(data["results"])
