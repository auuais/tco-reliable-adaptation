"""Score sealed predictions on the unused neighboring source and replay on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np

from evaluation_stability import sha, dump
from partner_shape import fit, native_rows


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/"results/mast3r_partner_shape"
RAW = ROOT/"reports/mast3r_partner_shape"
SCENES = ("heads", "chess", "fire")
SIDES = ("earlier", "later")
FRAMES = (25, 225, 425, 625, 825)
ARMS = ("zero", "real", "shuffled")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def disjoint_test(fit_pixels, test_rows):
    pixels, z = native_rows(test_rows)
    train = set(map(tuple, fit_pixels))
    keep = np.array([tuple(p) not in train for p in pixels], dtype=bool)
    if keep.sum() == 0:
        raise ValueError("Empty held-source view after overlap exclusion")
    return pixels[keep], z[keep], len(z)-int(keep.sum())


def check_seal(local):
    seal = read(PUBLIC/"seal.json")
    keys = {f"{s}_{d}" for s in SCENES for d in SIDES}
    if set(seal) != keys:
        raise ValueError("Incomplete global seal")
    inputs = read(PUBLIC/"inputs.json")
    for side in SIDES:
        if sha(PUBLIC/(side+".npz")) != inputs["packets"][side]:
            raise ValueError("Source packet changed")
    # Independent bridge to the pre-existing two-source range packet.
    parent = ROOT/"results/neighbor_sparse_depth/measurements.npz"
    with (np.load(PUBLIC/"earlier.npz") as earlier, np.load(PUBLIC/"later.npz") as later,
          np.load(parent) as archived):
        expected = {f"{s}_{f}" for s in SCENES for f in FRAMES}
        if any(set(z.files) != expected for z in (earlier, later, archived)):
            raise ValueError("Source window identities differ")
        for key in archived:
            nearest = {}
            for row in sorted([*earlier[key], *later[key]], key=lambda r:r[2]):
                nearest.setdefault(tuple(row[:2]), row)
            np.testing.assert_array_equal(np.stack(list(nearest.values())), archived[key])
    for name, digest in seal.items():
        directory = PUBLIC/name
        prediction = read(directory/"prediction.json")
        trained = read(directory/"trained.json")
        if (sha(directory/"prediction.json") != digest or sha(directory/"trained.json") != prediction["trained_sha256"]
                or sha(directory/"fitting.npz") != prediction["fitting_sha256"]
                or sha(directory/"training.json") != trained["training_sha256"]
                or sha(directory/"alignment.json") != trained["alignment_sha256"]
                or inputs["packets"][trained["side"]] != trained["packet_sha256"]):
            raise ValueError(f"Public prediction seal mismatch in {name}")
        if local and sha(RAW/name/"fields.npz") != prediction["fields_sha256"]:
            raise ValueError(f"Local field checksum mismatch in {name}")


def extract():
    from prepare_dense_focal_pilot import write_arrays
    check_seal(local=True)
    for scene in SCENES:
        for side, other in zip(SIDES, SIDES[::-1]):
            name = f"{scene}_{side}"
            directory = PUBLIC/name
            samples = {}
            with (np.load(RAW/name/"fields.npz") as fields, np.load(directory/"fitting.npz") as fitting,
                  np.load(PUBLIC/(other+".npz")) as unseen):
                for i, frame in enumerate(FRAMES):
                    xy, _, _ = disjoint_test(fitting[f"xy_{i}"], unseen[f"{scene}_{frame}"])
                    x, y = xy.T
                    samples[f"xy_{i}"] = xy
                    for arm in ARMS:
                        samples[f"{arm}_{i}"] = fields[f"{arm}_{i}"][y, x]
                        if arm != "zero":
                            samples[f"{arm}_design_{i}"] = fields[f"{arm}_design_{i}"][:, y, x].T
            write_arrays(directory/"held_predictions.npz", samples)


def score():
    check_seal(local=False)
    cases = {}
    fingerprints = {}
    for scene in SCENES:
        for side, other in zip(SIDES, SIDES[::-1]):
            name = f"{scene}_{side}"
            directory = PUBLIC/name
            rows = []
            fingerprints[name] = sha(directory/"held_predictions.npz")
            predicted = read(directory/"prediction.json")["rows"]
            with (np.load(directory/"held_predictions.npz") as held, np.load(directory/"fitting.npz") as fitting,
                  np.load(PUBLIC/(side+".npz")) as source, np.load(PUBLIC/(other+".npz")) as unseen):
                for i, frame in enumerate(FRAMES):
                    key = f"{scene}_{frame}"
                    xy, measured = native_rows(source[key])
                    np.testing.assert_array_equal(xy, fitting[f"xy_{i}"])
                    np.testing.assert_array_equal(measured, fitting[f"observed_{i}"])
                    held_xy, target, overlaps = disjoint_test(xy, unseen[key])
                    np.testing.assert_array_equal(held_xy, held[f"xy_{i}"])
                    expected_source = -4 if side == "earlier" else 4
                    if not np.all(source[key][:,3] == frame+expected_source) or not np.all(unseen[key][:,3] == frame-expected_source):
                        raise ValueError("Source identity overlap")
                    loss, training_loss, invalid = {}, {}, {}
                    for arm in ARMS:
                        trial = held[f"{arm}_{i}"]
                        base_fit = fitting[f"base_{i}"]
                        fit_trial = base_fit
                        if arm != "zero":
                            design = fitting[f"{arm}_design_{i}"]
                            coefficient, singular, fallback = fit(base_fit, design, measured)
                            reported = predicted[i]["arms"][arm]
                            np.testing.assert_allclose(coefficient, reported["a"], rtol=1e-5, atol=1e-7)
                            np.testing.assert_allclose(singular, reported["singular"], rtol=1e-8)
                            if fallback != reported["fallback"]:
                                raise ValueError("Identifiability decision changed")
                            np.testing.assert_allclose(held[f"zero_{i}"]+held[f"{arm}_design_{i}"]@coefficient,
                                                       trial, rtol=1e-6, atol=1e-8)
                            fit_trial = base_fit + design@coefficient
                        invalid[arm] = int(np.count_nonzero(trial <= 0))
                        loss[arm] = float(np.mean(np.abs(trial-target)))
                        training_loss[arm] = float(np.mean(np.abs(fit_trial-measured)))
                    rows.append(dict(frame=frame, fitted=len(xy), held=len(target), overlapping=overlaps,
                                     mae_m=loss, training_mae_m=training_loss, nonpositive=invalid))
            cases[name] = dict(rows=rows, mae_m={a:float(np.mean([r["mae_m"][a] for r in rows])) for a in ARMS})
    means, passes = {}, {}
    for side in SIDES:
        means[side] = {a:float(np.mean([cases[f"{s}_{side}"]["mae_m"][a] for s in SCENES])) for a in ARMS}
        passes[side] = [s for s in SCENES if beats_controls(cases[f"{s}_{side}"]["mae_m"])]
    both = sorted(set(passes["earlier"]) & set(passes["later"]))
    valid = all(all(r["nonpositive"][a] == 0 for a in ARMS) for case in cases.values() for r in case["rows"])
    passed = valid and len(both) >= 2 and all(beats_controls(v) for v in means.values())
    return dict(seal_sha256=sha(PUBLIC/"seal.json"), held_sha256=fingerprints, cases=cases, means=means,
                scenes_beating_controls=passes, scenes_passing_both=both, all_metric_samples_positive=valid,
                advance_to_dense=passed)


def beats_controls(errors):
    return errors["real"] < min(errors["shuffled"], errors["zero"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "score", "verify"))
    args = parser.parse_args()
    if args.action == "extract":
        extract()
    else:
        value = score()
        if args.action == "verify":
            if value != read(PUBLIC/"scores.json"):
                raise ValueError("Sparse screen replay differs")
        else:
            dump(PUBLIC/"scores.json", value)
        print("Held-source MAE in meters", value["means"], "advance", value["advance_to_dense"])
