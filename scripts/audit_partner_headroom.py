"""Measure the best possible conditional two-mode sparse MAE after sealing."""
import argparse
import numpy as np
from scipy.optimize import linprog

from score_partner_shape import PUBLIC, SCENES, SIDES, FRAMES, check_seal, disjoint_test, read, sha, dump


def best_possible(base, design, target):
    n = len(base)
    # Minimize sum of auxiliary absolute residuals with fixed bounded coefficients.
    matrix = np.vstack([np.column_stack([design, -np.eye(n)]), np.column_stack([-design, -np.eye(n)])])
    rhs = np.r_[target-base, base-target]
    fit = linprog(np.r_[0., 0., np.ones(n)], A_ub=matrix, b_ub=rhs,
                  bounds=[(-1., 1.)]*2 + [(0., None)]*n, method="highs")
    if not fit.success:
        raise RuntimeError(f"Headroom LP failed: {fit.message}")
    a = fit.x[:2]
    mae = np.mean(np.abs(base+design@a-target))
    np.testing.assert_allclose(mae, fit.fun/n, atol=1e-8)
    return float(mae)


def diagnostic():
    check_seal(local=False)
    primary = read(PUBLIC/"scores.json")
    cases = {}
    for scene in SCENES:
        for side, other in zip(SIDES, SIDES[::-1]):
            name = f"{scene}_{side}"
            if sha(PUBLIC/name/"held_predictions.npz") != primary["held_sha256"][name]:
                raise ValueError("Held predictions changed after primary score")
            rows = []
            with (np.load(PUBLIC/name/"held_predictions.npz") as held,
                  np.load(PUBLIC/name/"fitting.npz") as fitting, np.load(PUBLIC/(other+".npz")) as other_packet):
                for i, frame in enumerate(FRAMES):
                    xy, z, _ = disjoint_test(fitting[f"xy_{i}"], other_packet[f"{scene}_{frame}"])
                    np.testing.assert_array_equal(xy, held[f"xy_{i}"])
                    base = held[f"zero_{i}"]
                    row = {arm:best_possible(base, held[f"{arm}_design_{i}"], z) for arm in ("real", "shuffled")}
                    for arm in row:
                        if row[arm] > min(np.mean(abs(base-z)), np.mean(abs(held[f"{arm}_{i}"]-z)))+1e-8:
                            raise ValueError("Optimizer failed to dominate feasible candidates")
                    rows.append(row)
            cases[name] = rows
    means = {side:{arm:float(np.mean([row[arm] for scene in SCENES for row in cases[f"{scene}_{side}"]]))
                   for arm in ("real", "shuffled")} for side in SIDES}
    return dict(primary_sha256=sha(PUBLIC/"scores.json"), cases=cases, means=means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = diagnostic()
    if args.verify:
        previous = read(PUBLIC/"headroom.json")
        if value["primary_sha256"] != previous["primary_sha256"]:
            raise ValueError("Changed primary score")
        for case, rows in value["cases"].items():
            for i, row in enumerate(rows):
                for a in row:
                    np.testing.assert_allclose(row[a], previous["cases"][case][i][a], atol=1e-8, rtol=1e-8)
        for side, means in value["means"].items():
            for arm in means:
                np.testing.assert_allclose(means[arm], previous["means"][side][arm], atol=1e-8, rtol=1e-8)
    else:
        dump(PUBLIC/"headroom.json", value)
    print("Diagnostic lower bound on attainable held-source MAE", value["means"])
