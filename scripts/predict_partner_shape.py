"""Predict full metric depth from the fitting side only. Seal before scoring."""
import argparse
import numpy as np
import torch
from mast3r.cloud_opt.sparse_ga import canonical_view

from partner_shape import make_modes, fit, native_rows
from prepare_partner_sources import RAW, PUBLIC, FRAMES, SIDES, write_arrays, read, sha, dump
from run_mast3r_neural_transfer import SCENES


def ratio(canonical):
    data = canonical.detach().double().cpu().numpy()
    centers = data[4::8, 4::8].repeat(8, axis=0).repeat(8, axis=1)
    if np.any(centers <= 0):
        raise ValueError("Nonpositive canonical anchor")
    return data / centers


def predict(scene, side):
    directory, meta = RAW/f"{scene}_{side}", PUBLIC/f"{scene}_{side}"
    trained = read(meta/"trained.json")
    if (meta/"prediction.json").exists():
        raise FileExistsError("Prediction already sealed")
    for name, expected in trained["files"].items():
        if sha(directory/name) != expected:
            raise ValueError(f"Training artifact changed: {name}")
    ids = trained["view_ids"]
    packet = PUBLIC/f"{side}.npz"
    if sha(packet) != trained["packet_sha256"]:
        raise ValueError("Training packet changed")
    cache = directory/"cache"
    full, sampled, summary = {}, {}, []
    with np.load(directory/"alignment.npz") as alignment, np.load(packet) as source:
        for i, (target, frame) in enumerate(zip(ids, FRAMES)):
            paths = list((cache/"canon_views").glob(target+"_*.pth"))
            if len(paths) != 1:
                raise ValueError(f"Expected one canonical view for {target}")
            (_, canonical, _), focal = torch.load(paths[0], map_location="cpu", weights_only=True)
            fused = ratio(canonical)
            single = []
            for other in ids:
                if other == target:
                    continue
                xyz, confidence, _, _ = torch.load(cache/"forward"/target/(other+".pth"), map_location="cpu", weights_only=True)
                _, canonical_single, _ = canonical_view(xyz[None], confidence[None], 8)
                single.append(ratio(canonical_single))
            real, shuffled, fractions = make_modes(np.stack(single), fused, 1701+i)
            z = alignment["depth"][i].astype(float)
            anchors = z[4::8, 4::8].repeat(8, axis=0).repeat(8, axis=1)
            beta = float(focal.item()) / float(alignment["intrinsics"][i, 0, 0])
            reconstructed = anchors*(1.+beta*(fused-1.))
            np.testing.assert_allclose(reconstructed, z, rtol=2e-5, atol=1e-6)
            scale = trained["metric_scale"]
            base = z*scale
            pixel, observed = native_rows(source[f"{scene}_{frame}"])
            x, y = pixel.T
            full[f"zero_{i}"] = base
            sampled[f"xy_{i}"] = pixel
            sampled[f"observed_{i}"] = observed
            sampled[f"base_{i}"] = base[y, x]
            row = dict(frame=frame, retained_fraction=fractions.tolist(), native_bridge_max=float(np.max(np.abs(reconstructed-z))),
                       fit_count=len(x), nonpositive_zero=int((base <= 0).sum()), arms={})
            for arm, modes in (("real", real), ("shuffled", shuffled)):
                design = modes * (scale * anchors * beta)
                fit_design = design[:, y, x].T
                a, singular, fallback = fit(base[y, x], fit_design, observed)
                predicted = base + np.einsum("j,jhw->hw", a, design)
                if not np.isfinite(predicted).all():
                    raise ValueError("Nonfinite predicted depth")
                full[f"{arm}_{i}"] = predicted
                # Seal predictors too, so post-seal sampled replay can reconstruct each depth.
                full[f"{arm}_design_{i}"] = design
                sampled[f"{arm}_design_{i}"] = fit_design
                row["arms"][arm] = dict(a=a.tolist(), singular=singular.tolist(), fallback=bool(fallback),
                                       nonpositive=int((predicted<=0).sum()))
            summary.append(row)
    write_arrays(directory/"fields.npz", full)
    write_arrays(meta/"fitting.npz", sampled)
    dump(meta/"prediction.json", dict(trained_sha256=sha(meta/"trained.json"), rows=summary,
         fields_sha256=sha(directory/"fields.npz"), fitting_sha256=sha(meta/"fitting.npz")))
    print("Sealed fitted maps", scene, side, flush=True)


def seal():
    from partner_bias_control import offsets
    records = {}
    for scene in SCENES:
        for side in SIDES:
            name = f"{scene}_{side}"
            p = PUBLIC/name/"prediction.json"
            proof = read(p)
            if sha(RAW/name/"fields.npz") != proof["fields_sha256"] or sha(PUBLIC/name/"fitting.npz") != proof["fitting_sha256"]:
                raise ValueError(f"Changed prediction: {name}")
            records[name] = sha(p)
    if (PUBLIC/"seal.json").exists() and read(PUBLIC/"seal.json") != records:
        raise ValueError("Previous prediction seal changed")
    values = offsets()
    bias_seal = PUBLIC/"bias_seal.json"
    if bias_seal.exists():
        if read(bias_seal) != values:
            raise ValueError("Training-only biases changed after sealing")
    elif (PUBLIC/"scores.json").exists():
        raise ValueError("Cannot add bias control after scoring")
    else:
        dump(bias_seal, values)
    dump(PUBLIC/"seal.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, metavar=("SCENE", "SIDE"))
    args = parser.parse_args()
    if args.case:
        if args.case[0] not in SCENES or args.case[1] not in SIDES:
            parser.error("Unknown scene or side")
        predict(*args.case)
    else:
        seal()
