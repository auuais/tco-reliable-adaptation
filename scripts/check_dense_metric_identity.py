"""Send a known identical point map through the actual local reconstruction evaluator."""
import argparse

import numpy as np

from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import INPUT, PUBLIC, RAW, REF, ROOT
from score_saved_geometry import read


def run():
    import open3d as o3d
    import torch
    from run_repeat_calibration import evaluate
    spec=read(ROOT/"manifests/dense_focal_inputs.json")
    key="heads_base"
    image=spec["inputs"][key]
    ref=spec["references"][key]
    if sha(INPUT/image["file"])!=image["sha256"] or sha(REF/ref["file"])!=ref["sha256"]:
        raise ValueError("Unverified preflight input")
    with np.load(INPUT/image["file"],allow_pickle=False) as f:
        rgb=torch.from_numpy(f["images"])
    torch.set_num_threads(2)
    result={}
    with np.load(REF/ref["file"],allow_pickle=False) as f:
        for name in ("registered_rgb","released"):
            folder=RAW/"metric_identity"/name
            folder.mkdir(parents=True,exist_ok=True)
            np.random.seed(42)
            torch.manual_seed(42)
            o3d.utility.random.seed(42)
            reference=f[name]
            m=evaluate(reference.copy(),dict(images=rgb,pointclouds=reference,valid_mask=f["valid_mask"]),folder)
            values={k:m[k] for k in ("accuracy_mean_m","completeness_mean_m","normal_consistency_mean","valid_points")}
            if (values["valid_points"]!=ref["valid_pixels"] or max(values[k] for k in ("accuracy_mean_m","completeness_mean_m"))>1e-5
                    or not values["normal_consistency_mean"]>=.995):
                raise ValueError("Identity reconstruction does not receive identity metrics")
            result[name]=values
    return dict(input_key=key,reference_sha256=ref["sha256"],results=result,learned_model_scored=False,additional_cloud_usd=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write",action="store_true")
    args=p.parse_args()
    dest=PUBLIC/"metric_identity.json"
    if args.write and dest.exists():
        raise FileExistsError(dest)
    value=run()
    if args.write:
        dump(dest,value)
    else:
        saved=read(dest)
        for ref,metrics in value["results"].items():
            for k,v in metrics.items():
                np.testing.assert_allclose(v,saved["results"][ref][k],rtol=1e-6,atol=1e-7)
    print(value["results"])
