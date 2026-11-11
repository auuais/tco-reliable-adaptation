"""Reconstruct TCO's focal derivatives from its release equations and published inputs."""
import argparse

import numpy as np

from calibration_audit import resized_k
from evaluation_stability import dump, sha
from focal_zoom_observations import PUBLIC, SCALE
from score_saved_geometry import read


def field_of_view_gradient(encoding, target_xy):
    # The upstream encoding stores vertical then horizontal FoV. The K matrix stores fx, fy.
    half_angle=np.asarray(encoding,dtype=np.float64)[:,7:9]/2
    dims=np.array([392.,518.])
    focal=dims/(2*np.tan(half_angle))
    derivative=-dims/(4*np.sin(half_angle)**2)
    # Upstream casts the target to FP32; the pixel Smooth L1 changes branch at one pixel.
    residual=focal-np.asarray(target_xy,dtype=np.float32)[:,::-1]
    return (.01/(3*len(focal)))*np.clip(residual,-1,1)*derivative


def verify():
    record=read(PUBLIC/"summary.json")
    observations=read(PUBLIC/"observations.json")
    encodings={(o["camera"]["scene"],o["camera"]["batch"]):o["camera"]["encoding"][0] for o in observations}
    base=np.repeat(resized_k(525.,518,392,uniform=False)[[0,1],[0,1]][None],5,axis=0)*SCALE
    trace=PUBLIC/"gradient_arrays.zip"
    if sha(trace)!=record["trace_sha256"]:
        raise ValueError("Gradient archive differs")
    with np.load(trace,allow_pickle=False) as z:
        variants=z["variants"]
    if variants.shape!=(465,4,5,9) or len(encodings)!=33 or len(record["rows"])!=465:
        raise ValueError("Unexpected focal-gradient evidence")
    worst=0.
    for r,g in zip(record["rows"],variants):
        # Rebuild all 5 targets from the case specification, independently of the recorded errors.
        supplied,correct=base.copy(),base.copy()
        if r["case"] in ("metadata_fault","valid_zoom"):
            supplied[r["view"]] *= SCALE**r["direction"]
        if r["case"] in ("valid_zoom","hidden_zoom"):
            correct[r["view"]] *= SCALE**r["direction"]
        for actual,target in zip(g,(supplied,base,correct,correct)):
            predicted=field_of_view_gradient(encodings[r["scene"],r["batch"]],target)
            np.testing.assert_allclose(actual[:,7:9],predicted,atol=1e-4,rtol=2e-5)
            worst=max(worst,float(np.abs(actual[:,7:9]-predicted).max()))
        np.testing.assert_array_equal(g[0,:,:7],g[1,:,:7])
        np.testing.assert_array_equal(g[0,:,:7],g[2,:,:7])
        np.testing.assert_array_equal(g[2,:,7:],g[3,:,7:])
    return dict(trace_sha256=sha(trace),matched_focal_entries=int(variants[:,:,:,7:].size),
                compared_gradient_arrays=len(variants)*4,maximum_absolute_difference=worst,
                absolute_tolerance=1e-4,relative_tolerance=2e-5,
                pose_entries_unchanged_across_corrected_focals=True,focal_entries_unchanged_across_pose_noise=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    result=verify()
    path=PUBLIC/"analytic_gradient_check.json"
    if args.write:
        if path.exists():
            raise FileExistsError(path)
        dump(path,result)
    else:
        stored=read(path)
        for key,value in result.items():
            if key=="maximum_absolute_difference":
                if abs(value-stored[key])>1e-11:
                    raise ValueError("Analytic gradient discrepancy changed")
            elif value!=stored[key]:
                raise ValueError(key)
    print("Verified",result["matched_focal_entries"],"FoV derivatives against the analytic loss.")


if __name__=="__main__":
    main()
