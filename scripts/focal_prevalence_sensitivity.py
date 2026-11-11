"""How does focal-gradient error change with the fraction of visible outliers that are faults?"""
import argparse
import json
from statistics import mean

from score_saved_geometry import read
from visual_focal_baseline import PUBLIC as CLASSICAL


def interval(target, alternatives):
    """Return the p interval with risk(target) <= risk(alternative) for every alternative."""
    lo, hi = 0., 1.
    for other in alternatives:
        if target == other:
            continue
        d0, d1 = (a - b for a, b in zip(target, other))
        if d0 > 0 and d1 > 0:
            return None
        if d0 > 0:
            lo = max(lo, d0 / (d0 - d1))
        elif d1 > 0:
            hi = min(hi, -d0 / (d1 - d0))
        if lo > hi:
            return None
    return [lo, hi]


def analyze(rows):
    visible = [r for r in rows if r["case"] in ("valid_zoom", "metadata_fault")]
    if len(visible) != 300:
        raise ValueError("Expected 300 visible cases across pose perturbations")
    groups = {}
    for level in ("clean", "mild", "strong"):
        selected = [r for r in visible if r["noise"] == level]
        valid = [r for r in selected if r["case"] == "valid_zoom"]
        faults = [r for r in selected if r["case"] == "metadata_fault"]
        if len(valid) != len(faults):
            raise ValueError("Unbalanced reference rows")
        values = {m: [mean(r["errors"][m]["focal_gradient_relative"] for r in group) for group in (valid, faults)]
                  for m in selected[0]["errors"]}
        method = values["both_folds"]
        groups[level] = dict(valid_configurations=len(valid), fault_configurations=len(faults),
            errors_valid_then_fault=values,
            rule_no_worse_than_both_unconditional=interval(method, [values["retain"], values["consensus"]]),
            rule_no_worse_than_all_compared=interval(method, list(values.values())),
            rule_no_worse_than_pretrained=interval(method, [values["model_nearest"]]),
            rule_no_worse_than_solver_snapped=interval(method, [values["iterative_nearest"]]))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = analyze(read(CLASSICAL / "summary.json")["rows"])
    destination = CLASSICAL / "prevalence_sensitivity.json"
    if args.write:
        with destination.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    elif result != read(destination):
        raise ValueError("Prevalence sensitivity differs from stored evidence")
    print(json.dumps(result["mild"]))


if __name__ == "__main__":
    main()
