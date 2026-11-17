"""Independently reconstruct the corrected surface from public native depth/cameras."""
import argparse
import hashlib
import numpy as np
from partner_bias_surface import PUBLIC, RAW, ARMS, read, sha
from replay_mast3r_transfer import independently_resize
from prepare_dense_focal_pilot import write_arrays


def replay(export):
    seal = read(PUBLIC/"seal.json")
    path = PUBLIC/"sampled_points.npz"
    samples = {}
    existing = {} if export else np.load(path)
    try:
        for name, row in seal["cases"].items():
            native = PUBLIC/"native"/(name+".npz")
            if sha(native) != row["native_sha256"]:
                raise ValueError("Native capture changed")
            with np.load(native) as capture:
                for arm in ARMS:
                    z = capture["depth"].astype(float)
                    if arm == "bias":
                        z += np.array(row["bias_m"])[:,None,None] / row["gauge"]
                    if arm == "scale":
                        z *= np.array(row["scale"])[:,None,None]
                    depth, k = independently_resize(z, capture["intrinsics"])
                    yy, xx = np.indices(depth.shape[1:])
                    homogeneous = np.stack([xx,yy,np.ones_like(xx)],axis=-1)
                    stride = 1 if export else 12
                    pixel = homogeneous[::stride,::stride]
                    camera = capture["c2w"].astype(float)
                    computed = np.stack([(pixel @ np.linalg.inv(k[i]).T * depth[i,::stride,::stride,None])
                        @ camera[i,:3,:3].T + camera[i,:3,3] for i in range(len(camera))])
                    tag = name+"_"+arm
                    if export:
                        h = row["arms"][arm]["points"]
                        with np.load(RAW/"points"/(h+".npz")) as output:
                            reference = output["points"]
                        if hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest() != h:
                            raise ValueError("Raw point map changed")
                        np.testing.assert_allclose(computed, reference, rtol=1e-5, atol=2e-5)
                        samples[tag] = reference[:,::12,::12]
                    else:
                        np.testing.assert_allclose(computed, existing[tag], rtol=1e-5, atol=2e-5)
        if export:
            write_arrays(path, samples)
        elif len(existing.files) != 18:
            raise ValueError("Expected eighteen surface samples")
    finally:
        if not export:
            existing.close()
    print("Independently verified 18", "complete point maps" if export else "sampled point maps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    replay(parser.parse_args().export)
