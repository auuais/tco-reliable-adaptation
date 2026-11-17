"""Seal training-only median view biases as a secondary calibration diagnostic."""
import argparse
import numpy as np
from score_partner_shape import PUBLIC, SIDES, SCENES, FRAMES, read, sha, dump, disjoint_test


def offsets():
    values = {}
    for scene in SCENES:
        for side in SIDES:
            name = f"{scene}_{side}"
            fitting = PUBLIC/name/"fitting.npz"
            if sha(fitting) != read(PUBLIC/name/"prediction.json")["fitting_sha256"]:
                raise ValueError("Fitting inputs changed")
            with np.load(fitting) as data:
                rows = []
                for i in range(len(FRAMES)):
                    observed = data[f"observed_{i}"]
                    limit = .05*np.median(observed)
                    rows.append(float(np.clip(np.median(observed-data[f"base_{i}"]), -limit, limit)))
            values[name] = rows
    return values


def evaluate(values):
    primary = read(PUBLIC/"scores.json")
    cases = {}
    for scene in SCENES:
        for side, other in zip(SIDES, SIDES[::-1]):
            name = f"{scene}_{side}"
            directory = PUBLIC/name
            rows = []
            if sha(directory/"held_predictions.npz") != primary["held_sha256"][name]:
                raise ValueError("Held predictions changed")
            with (np.load(directory/"fitting.npz") as training, np.load(directory/"held_predictions.npz") as held,
                  np.load(PUBLIC/(other+".npz")) as packet):
                for i,f in enumerate(FRAMES):
                    xy, measured, _ = disjoint_test(training[f"xy_{i}"], packet[f"{scene}_{f}"])
                    np.testing.assert_array_equal(xy, held[f"xy_{i}"])
                    z = held[f"zero_{i}"] + values[name][i]
                    rows.append(dict(mae_m=float(np.mean(abs(z-measured))), nonpositive=int((z<=0).sum())))
            cases[name] = rows
    means = {side:float(np.mean([r["mae_m"] for scene in SCENES for r in cases[f"{scene}_{side}"]])) for side in SIDES}
    return dict(offsets_sha256=sha(PUBLIC/"bias_seal.json"), primary_sha256=sha(PUBLIC/"scores.json"), cases=cases, means=means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "score", "verify"))
    action = parser.parse_args().action
    values = offsets()
    seal = PUBLIC/"bias_seal.json"
    if action == "seal":
        if (PUBLIC/"scores.json").exists():
            raise ValueError("Bias control must be sealed before primary scoring")
        dump(seal, values)
    else:
        if values != read(seal):
            raise ValueError("Training-derived bias changed")
        result = evaluate(values)
        if action == "verify":
            if result != read(PUBLIC/"bias.json"):
                raise ValueError("Bias control replay differs")
        else:
            dump(PUBLIC/"bias.json", result)
        print("Classical held-source bias control MAE", result["means"])
