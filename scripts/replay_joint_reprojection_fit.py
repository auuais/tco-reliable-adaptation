"""Re-run all published joint reprojection optimizations from compact numeric inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from reprojection_fit import fit

PUBLIC = Path(__file__).resolve().parents[1]/"results/source_frame_spacing"
BUNDLE = PUBLIC/"joint_reprojection_fit_inputs.npz"
MANIFEST = PUBLIC/"joint_reprojection_fit_inputs.json"
PREDICTIONS = PUBLIC/"joint_reprojection_predictions.json"
ARMS = ("pose","depth","joint","joint_prior")
CASES = ("heads_base","chess_base","fire_base")
FIELDS = ("source_view","target_view","camera_ray","matched_pixel")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack():
    import joint_rgb_reprojection as source
    if BUNDLE.exists() or MANIFEST.exists():
        raise FileExistsError("Existing fit-input bundle")
    arrays = {}
    for key,_,_,_,scale,rot,c,k,target,_,folds in source.input_batches():
        for name,array in dict(rotation=rot,center=c/scale,intrinsic=k,prior=target).items():
            arrays[f"{key}_{name}"] = array
        for index,fold in enumerate(folds):
            for label,array in zip(FIELDS,fold):
                arrays[f"{key}_{index}_{label}"] = array
    np.savez_compressed(BUNDLE,**arrays)
    manifest = dict(bundle_sha256=digest(BUNDLE),prediction_sha256=digest(PREDICTIONS),cases=list(CASES))
    MANIFEST.write_bytes((json.dumps(manifest,indent=2)+"\n").encode())


def replay():
    manifest=json.loads(MANIFEST.read_bytes())
    if (manifest["cases"]!=list(CASES) or manifest["bundle_sha256"]!=digest(BUNDLE)
            or manifest["prediction_sha256"]!=digest(PREDICTIONS)):
        raise ValueError("Published fit inputs or expected outputs changed")
    sealed=json.loads(PREDICTIONS.read_bytes())["rows"]
    if set(sealed)!=set(CASES): raise ValueError("Missing scene")
    max_parameter_error=max_cost_error=0.
    with np.load(BUNDLE,allow_pickle=False) as z:
        for key in CASES:
            folds=[tuple(z[f"{key}_{i}_{label}"] for label in FIELDS) for i in (0,1)]
            if [len(f[0]) for f in folds]!=sealed[key]["fold_counts"]: raise ValueError("Fold sizes differ")
            rot,c,k,target=(z[f"{key}_{label}"] for label in ("rotation","center","intrinsic","prior"))
            for arm in ARMS:
                _,observed=fit(arm,folds,rot,c,k,target)
                expected=sealed[key]["arms"][arm]
                if observed["accepted"]!=expected["accepted"]: raise ValueError("Validation decision changed")
                delta=np.abs(np.array(observed["proposed"])-expected["proposed"])
                np.testing.assert_allclose(observed["proposed"],expected["proposed"],rtol=1e-4,atol=2e-6)
                max_parameter_error=max(max_parameter_error,float(delta.max()))
                for field in ("train_before","train_after","validation_before","validation_after"):
                    np.testing.assert_allclose(observed[field],expected[field],rtol=1e-5,atol=1e-7)
                    max_cost_error=max(max_cost_error,abs(observed[field]-expected[field]))
    print(f"Verified 12 fits; maximum absolute parameter difference {max_parameter_error:.3g}, cost difference {max_cost_error:.3g}.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--pack",action="store_true",help="Regenerate inputs from the local archived development maps")
    if parser.parse_args().pack: pack()
    replay()
