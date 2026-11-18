"""Independently verify D1 graph-zero responses by scalar ridge arithmetic."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from run_profiled_d0 import ROOT, now, read, sha, write_new

STATES = ("B0", "B1", "F0", "F1")
PATHS = (("B0", "B1"), ("F0", "F1"), ("B0", "F0"), ("B1", "F1"))
RTOL, ATOL = 1e-10, 1e-10


def scalar(values):
    return [float(v) for v in values]


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def fit(a, b, ridge):
    columns = [scalar(column) for column in a.T]
    b = scalar(b)
    diagonal = [dot(c, c) + ridge for c in columns]
    return [(dot(c, b) + ridge)/h for c, h in zip(columns, diagonal)], diagonal


def evaluate(matrix, scales):
    return np.array([dot(scalar(row), scales) for row in matrix])


def verify_matrix(a):
    if not (np.isfinite(a).all() and (np.count_nonzero(a, axis=1) == 1).all()):
        raise ValueError("Expected finite rows supported on one camera column")


def independent_replay(attempt, *, verify_only=False):
    if not re.fullmatch(r"d1_[0-9]{2}", attempt):
        raise ValueError("Invalid attempt ID")
    public = ROOT / "results/lora_profiled_d1_v1/attempts" / attempt
    private_matrix = ROOT / "reports/lora_profiled_d1_v1/attempts" / attempt / "matrices.npz"
    matrix = public / "matrices_replay.npz" if verify_only else private_matrix
    report_path = public / "report.json"
    report_hash = sha(report_path)
    seal_path = public / "model_seal.json"
    seal, report, replay = read(seal_path), read(report_path), read(public / "replay.json")
    if not (replay["pass_all"] and replay["report_sha256"] == report_hash
            and replay["model_seal_sha256"] == report["model_seal_sha256"] == sha(seal_path)):
        raise ValueError("Required exact CPU replay and seal do not agree")
    bound = seal["arrays"][0]
    if bound["file"] != private_matrix.relative_to(ROOT).as_posix() or sha(matrix) != bound["sha256"]:
        raise ValueError("Matrix bundle changed")
    with np.load(matrix, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    rows = report["scientific"]["responses"]
    expected = [(fold, f"{a}_to_{b}") for fold in (0, 1) for a, b in PATHS]
    if [(row["fold"], row["contrast"]) for row in rows] != expected:
        raise ValueError("Report fold or contrast membership changed")
    checks = []
    for row in rows:
        fold, contrast = row["fold"], row["contrast"]
        base, final = contrast.split("_to_")
        p = row["response"]
        def matrix_at(state, field):
            return arrays[f"fold{fold}_{state}_{field}"]
        a0, a1 = (matrix_at(s, "A") for s in (base, final))
        q0, q1 = (matrix_at(s, "Q") for s in (base, final))
        for a in (a0, a1, q0, q1):
            verify_matrix(a)
        b, d = (matrix_at(base, k) for k in ("b", "d"))
        ridge = float(matrix_at(base, "ridge"))
        s0, _ = fit(a0, b, ridge)
        s1, diagonal = fit(a1, b, ridge)
        e0, e1 = evaluate(q0, s0)-d, evaluate(q1, s1)-d
        t = evaluate(q1-q0, s0)
        c = evaluate(q1, [u-v for v, u in zip(s0, s1)])
        f = p["finite_response"]
        fields = dict(s0=s0, s1=s1, e0=e0, e1=e1, direct_t=t, compensation_c=c,
                      J0=.5*dot(scalar(e0), scalar(e0)), J1=.5*dot(scalar(e1), scalar(e1)))
        differences = {}
        for name, value in fields.items():
            ref = f[name]
            np.testing.assert_allclose(value, ref, rtol=RTOL, atol=ATOL, err_msg=name)
            differences[name] = float(np.max(np.abs(np.array(value)-ref)))

        direction = p["derivative"]
        if contrast == "B0_to_B1":
            a_dot, q_dot = a1-a0, q1-q0
            ds = [(dot(scalar(dc), scalar(b))-2*s*dot(scalar(ac), scalar(dc)))/h
                  for dc, ac, s, h in zip(a_dot.T, a1.T, s1, diagonal)]
            direct = evaluate(q_dot, s1)
            full = direct+evaluate(q1, ds)
            for name, value in (("full_outer_derivative", dot(scalar(e1), scalar(full))),
                                ("detached_at_same_state_outer_derivative", dot(scalar(e1), scalar(direct)))):
                np.testing.assert_allclose(value, direction[name], rtol=RTOL, atol=ATOL, err_msg=name)
                differences[name] = abs(value-direction[name])
        elif direction is not None:
            raise ValueError("Unexpected derivative contrast")
        checks.append(dict(fold=fold, contrast=contrast, maximum_abs_difference=differences))

    if sha(matrix) != bound["sha256"] or sha(report_path) != report_hash:
        raise ValueError("Inputs changed during independent replay")
    receipt = dict(schema="profiled_d1_independent_secant_v1", pass_all=True, attempt=attempt,
                   report_sha256=report_hash, matrices_sha256=bound["sha256"], checks=checks,
                   method="Diagonal ridge solve and quotient-rule derivative with scalar compensated sums",
                   tolerances=dict(rtol=RTOL, atol=ATOL), verified_at_utc=now(), rental_usd=0)
    receipt_path = public / "independent_arithmetic.json"
    if verify_only:
        earlier = read(receipt_path)
        fields = set(receipt) - {"verified_at_utc", "checks"}
        if any(earlier[field] != receipt[field] for field in fields):
            raise ValueError("Published independent receipt describes different evidence")
    else:
        write_new(receipt_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "checks"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Replay the published compact matrix bundle")
    args = parser.parse_args()
    independent_replay(args.attempt, verify_only=args.verify_only)
