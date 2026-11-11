"""Compare paired geometry after equal-sized moves toward and away from the supplied rig."""
import argparse
from statistics import mean

import prior_direction_control as control
import prior_noise_paired as paired
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC
from score_saved_geometry import read

RESULT=PUBLIC/"prior_direction_paired.json"


def compare(opposite):
    r=read(paired.RESULT)["results"]
    original=read(paired.paired.RESULT)["results"]
    result={}
    for s,v in opposite.items():
        if s=="137":
            direct={k:dict(depth_mse=t["coupled"]["depth_mse"],normal_deg=t["coupled"]["angle_deg"])
                    for k,t in {"overall":original["overall"],**original["scenes"]}.items()}
        else:
            direct={k:t["coupled"] for k,t in {"overall":r[s]["overall"],**r[s]["scenes"]}.items()}
        result[s]={k:dict(correct={m:direct[k][m] for m in ("depth_mse","normal_deg")},
                          opposite={m:t["coupled"][m] for m in ("depth_mse","normal_deg")})
                   for k,t in {"overall":v["overall"],**v["scenes"]}.items()}
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=[sha(x) for x in (control.SEAL,paired.RESULT,paired.paired.RESULT)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows=paired.measure(control,control.SEEDS)
        dump(RESULT,dict(inputs=deps,rows=rows,results=compare(paired.summarize(rows,control.SEEDS))))
    data=read(RESULT)
    if data["inputs"]!=deps or data["results"]!=compare(paired.summarize(data["rows"],control.SEEDS)): raise ValueError("Direction pixel replay differs")
    print({s:v["overall"] for s,v in data["results"].items()})
