"""Compare the corresponding-point error of RGB-registered and original views."""
import argparse

import numpy as np

import per_view_alignment as alignment
import rgb_rigid_view as rigid
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, SCENES
from score_saved_geometry import read

RESULT = PUBLIC/"rgb_rigid_paired.json"


def summarize(moments):
    direct = read(alignment.RESULT)["results"]
    out = {}
    for ref in alignment.temporal.REFS:
        scenes = {}
        for scene in SCENES:
            obs = alignment.score(np.array(moments[scene]["rgb"][ref]), True)
            orig = direct[ref]["direct"]["opposite_blocks"]["scenes"][scene]
            scenes[scene] = {m:1-obs[m]/orig[m] for m in alignment.MODES}
        out[ref] = scenes
    return out


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("measure","verify"))
    deps = [sha(path) for path in (alignment.RESULT, rigid.SEAL)]
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows = read(rigid.SEAL)["rows"]
        moments = alignment.collect(lambda k:{"rgb":(rigid.DATA,rows[k]["hash"])})
        dump(RESULT, dict(inputs=deps,moments=moments,results=summarize(moments)))
    data = read(RESULT)
    if deps != data["inputs"]: raise ValueError("Paired sources differ")
    replay = summarize(data["moments"])
    for ref,scenes in replay.items():
        for scene,values in scenes.items():
            for m,v in values.items():
                np.testing.assert_allclose(v,data["results"][ref][scene][m],atol=1e-9)
        print(ref, np.mean([v["scene"] for v in scenes.values()]), [v["scene"] for v in scenes.values()])
