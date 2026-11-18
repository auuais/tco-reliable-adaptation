"""Replay public view means, scene means and gates without measurement vectors."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/lora_profiled_compensation_v1"
SCENES = ("heads", "chess", "fire")
SIDES = ("earlier", "later")
ARMS = ("P0", "P1", "P3", "R0", "R1", "R2")
FRAMES = (25, 225, 425, 625, 825)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def average(values):
    values = list(values)
    return None if None in values else math.fsum(v/len(values) for v in values)


def equal(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def replay(report):
    """Check reduction and nomination from the already published view evidence."""
    wanted = {(s, d, a, f) for s in SCENES for d in SIDES for a in ARMS for f in FRAMES}
    indexed = {(row["scene"], row["side"], row["arm"], row["frame"]): row for row in report["rows"]}
    require(len(report["rows"]) == len(wanted) and set(indexed) == wanted,
            "A full registered grid of 180 view results is required")
    verdict = report["verdict"]
    cases, good = {}, {}
    directional = {}
    for endpoint in ("final", "native"):
        grouped = defaultdict(list)
        for (scene, side, arm, _), row in indexed.items():
            metric = row[endpoint]
            mean = metric["mae_m"]
            require(mean is None or (math.isfinite(mean) and mean >= 0), "Invalid error value")
            expected_count = row["held_count"]
            require(metric["count"] == expected_count and expected_count > 0
                    and metric["candidate_dependent_exclusions"] == 0,
                    "View count differs from the registered held-source mask")
            key = f"{scene}_{side}_{arm}"
            grouped[key].append(metric)
        cases[endpoint] = {key: average(item["mae_m"] for item in values)
                           for key, values in grouped.items()}
        good[endpoint] = {key: all(item["all_samples_valid"] for item in values)
                          for key, values in grouped.items()}
        directional[endpoint] = {side: {arm: average(
            cases[endpoint][f"{scene}_{side}_{arm}"] for scene in SCENES)
            for arm in ARMS} for side in SIDES}
        require(all(equal(value, verdict["case_mae_m"][endpoint][key])
                    for key, value in cases[endpoint].items())
                and good[endpoint] == verdict["validity"][endpoint], "Scene arithmetic differs")
        require(all(equal(value, verdict["directional_mae_m"][endpoint][side][arm])
                    for side, items in directional[endpoint].items() for arm, value in items.items()),
                "Directional arithmetic differs")

    improving = [scene for scene in SCENES if all(
        good["final"][f"{scene}_{side}_R1"] and good["final"][f"{scene}_{side}_P1"]
        and cases["final"][f"{scene}_{side}_R1"] < cases["final"][f"{scene}_{side}_P1"]
        for side in SIDES)]
    require(improving == verdict["same_scenes_improving_over_p1"], "Common improving rooms differ")
    passes = len(improving) >= 2 and report["numerical_failures"] == 0
    for control in ("P1", "P3", "R0", "R2"):
        for side in SIDES:
            valid = all(good["final"][f"{scene}_{side}_{arm}"]
                        for scene in SCENES for arm in (control, "R1"))
            denominator = directional["final"][side][control]
            numerator = directional["final"][side]["R1"]
            gain = (1 - numerator/denominator if valid and numerator is not None
                    and denominator is not None and denominator > 0 else None)
            require(equal(gain, verdict["r1_directional_relative_gain"][control][side]),
                    "Registered comparator gain differs")
            passes = passes and gain is not None and (gain >= .01 if control == "P1" else gain > 0)
    nomination = "R1" if passes else None
    require(verdict["nomination"] == nomination, "Nomination differs from independently reduced metrics")
    return dict(pass_all=True, view_rows=len(indexed), nomination=nomination,
                scope="Public aggregate arithmetic beginning with published view statistics")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=PUBLIC/"held_source_scores.json")
    p.add_argument("--proof", type=Path, default=PUBLIC/"independent_replay.json")
    args = p.parse_args()
    data = args.report.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    proof = json.loads(args.proof.read_bytes())
    require(proof["pass_all"] is True and proof["report_sha256"] == digest,
            "Independent sample replay receipt does not bind this score report")
    report = json.loads(data)
    require(report["sample_replay"]["sha256"] == proof["samples_sha256"],
            "Independent replay receipt binds a different sample archive")
    result = replay(report)
    require(proof["nomination"] == result["nomination"], "Independent sample replay verdict differs")
    print(json.dumps(dict(**result, report_sha256=digest), indent=2))


if __name__ == "__main__":
    main()
