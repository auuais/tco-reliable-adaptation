"""Independently check the published compact sampled-depth evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from profiled_compensation_protocol import ARMS, PUBLIC, ROOT, SCENES, SIDES
from run_profiled_d0 import now, read, sha, write_new

FRAMES = (25, 225, 425, 625, 825)
CONTROLS = ("P1", "P3", "R0", "R2")
RTOL, ATOL = 1e-12, 1e-12


def independent_verdict(rows, sample_path, expected_sha, failure_count):
    if sha(sample_path) != expected_sha:
        raise ValueError("Compact prediction/observation bytes changed")
    case_mae, all_valid = {}, {}
    index = {}
    for row in rows:
        key = tuple(row[k] for k in ("scene", "side", "arm", "frame"))
        if key in index:
            raise ValueError("Duplicate scored row")
        index[key] = row
    if len(index) != 180:
        raise ValueError("Full inventory of 180 scored views required")
    with np.load(sample_path, allow_pickle=False) as data:
        for endpoint in ("native", "final"):
            case_mae[endpoint], all_valid[endpoint] = {}, {}
            for scene in SCENES:
                for side in SIDES:
                    for arm in ARMS:
                        maes = []
                        good = []
                        for frame in FRAMES:
                            row = index[scene, side, arm, frame]
                            prefix = f"{scene}_{side}_{frame}"
                            xy, measured = data[prefix+"__xy"], data[prefix+"__measured_m"]
                            values = data[f"{prefix}__{arm}_{endpoint}_m"]
                            if (xy.shape != (len(measured),2) or values.shape != measured.shape
                                    or not np.isfinite(measured).all() or (measured <= 0).any()):
                                raise ValueError("Invalid compact measurements or sample shape")
                            finite = np.isfinite(values)
                            errors = np.abs(values[finite] - measured[finite])
                            representable = np.isfinite(errors)
                            valid = bool(finite.all() and (values > 0).all() and representable.all())
                            mean = float(np.mean(errors)) if finite.all() and representable.all() else None
                            original = row[endpoint]
                            if (original["all_samples_valid"] != valid or original["count"] != len(measured)
                                    or original["nonfinite_count"] != int((~finite).sum())
                                    or original["nonpositive_count"] != int((finite & (values <= 0)).sum())
                                    or (mean is None) != (original["mae_m"] is None)):
                                raise ValueError("Recorded view validity or count differs")
                            if mean is not None:
                                np.testing.assert_allclose(mean, original["mae_m"], rtol=RTOL, atol=ATOL)
                            maes.append(mean)
                            good.append(valid)
                        run = f"{scene}_{side}_{arm}"
                        case_mae[endpoint][run] = None if None in maes else float(np.mean(maes))
                        all_valid[endpoint][run] = all(good)
    if sha(sample_path) != expected_sha:
        raise ValueError("Compact samples changed during replay")

    means = {}
    for endpoint in ("native", "final"):
        means[endpoint] = {}
        for side in SIDES:
            means[endpoint][side] = {}
            for arm in ARMS:
                values = [case_mae[endpoint][f"{scene}_{side}_{arm}"] for scene in SCENES]
                means[endpoint][side][arm] = None if None in values else float(np.mean(values))
    positive_cases = [scene for scene in SCENES if all(
        all_valid["final"][f"{scene}_{side}_{arm}"] for side in SIDES for arm in ("R1","P1")) and all(
        case_mae["final"][f"{scene}_{side}_R1"] < case_mae["final"][f"{scene}_{side}_P1"]
        for side in SIDES)]
    eligible = failure_count == 0 and len(positive_cases) >= 2
    for side in SIDES:
        for arm in CONTROLS:
            valid = all(all_valid["final"][f"{scene}_{side}_{a}"] for scene in SCENES for a in ("R1", arm))
            if not valid or means["final"][side][arm] <= 0:
                eligible = False
                continue
            gain = 1 - means["final"][side]["R1"]/means["final"][side][arm]
            eligible = eligible and (gain >= .01 if arm == "P1" else gain > 0)
    return dict(nomination="R1" if eligible else None, case_mae_m=case_mae, directional_mae_m=means,
                same_scenes_improving_over_p1=positive_cases)


def verify(*, output=None):
    path = ROOT / PUBLIC / "held_source_scores.json"
    report_hash = sha(path)
    report = read(path)
    samples = ROOT / PUBLIC / "held_source_replay.npz"
    expected = report["sample_replay"]
    if expected["file"] != samples.relative_to(ROOT).as_posix():
        raise ValueError("Unexpected compact evidence path")
    actual = independent_verdict(report["rows"], samples, expected["sha256"], report["numerical_failures"])
    expected_verdict = report["verdict"]
    if actual["nomination"] != expected_verdict["nomination"] or (
            actual["same_scenes_improving_over_p1"] != expected_verdict["same_scenes_improving_over_p1"]):
        raise ValueError("Independent nomination or paired scene decision differs")
    for endpoint, cases in actual["case_mae_m"].items():
        for case, score in cases.items():
            other = expected_verdict["case_mae_m"][endpoint][case]
            if score is not None:
                np.testing.assert_allclose(score, other, rtol=RTOL, atol=ATOL)
            elif other is not None:
                raise ValueError("Independent case validity differs")
        for side in SIDES:
            for arm in ARMS:
                first, second = (v["directional_mae_m"][endpoint][side][arm]
                                 for v in (actual, expected_verdict))
                if first is None or second is None:
                    if first != second:
                        raise ValueError("Independent directional validity differs")
                else:
                    np.testing.assert_allclose(first, second, rtol=RTOL, atol=ATOL)
    if sha(path) != report_hash:
        raise ValueError("Original score report changed")
    result = dict(pass_all=True, view_rows=180, report_sha256=report_hash,
                  samples_sha256=expected["sha256"], verified_at_utc=now(),
                  nomination=actual["nomination"], method="Independent array means and gate recomputation")
    if output is not None:
        write_new(output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    verify(output=p.parse_args().output)
