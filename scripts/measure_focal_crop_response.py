"""Paired intervention response of the frozen VGGT camera head to a known digital zoom."""
import argparse
import json
from statistics import mean, median

import numpy as np

from focal_zoom_observations import PUBLIC, SCALE
from score_saved_geometry import read


def focal_xy(camera):
    angles = np.asarray(camera["encoding"], dtype=float)[0, :, 7:9]
    if angles.shape != (5, 2) or not np.all((angles > 0) & (angles < np.pi)):
        raise ValueError("Invalid vertical and horizontal field of view")
    return (np.array([392., 518.]) / (2 * np.tan(angles / 2)))[:, ::-1]


def summary(rows):
    return dict(paired_inputs=len(rows), correct_mean_direction=sum(r["mean_sensitivity"] > 0 for r in rows),
                correct_both_axis_direction=sum(min(r["sensitivity_xy"]) > 0 for r in rows),
                mean_sensitivity=mean(r["mean_sensitivity"] for r in rows),
                median_sensitivity=median(r["mean_sensitivity"] for r in rows),
                mean_affected_log_error=mean(r["mean_affected_log_error"] for r in rows),
                mean_untouched_absolute_log_change=mean(r["mean_untouched_absolute_log_change"] for r in rows),
                mean_signed_common_response=mean(r["common_sensitivity"] for r in rows),
                mean_neighbor_centered_sensitivity=mean(r["mean_sensitivity"] - r["common_sensitivity"] for r in rows))


def build(observations):
    inputs = {(o["camera"]["scene"], o["camera"]["batch"]): focal_xy(o["camera"]) for o in observations}
    scenes = sorted({scene for scene, _ in inputs})
    if len(inputs) != 33 or len(scenes) != 3:
        raise ValueError("Expected 11 model predictions per development scene")
    rows = []
    for scene in scenes:
        base = inputs[scene, "base"]
        for v in range(5):
            mask = np.arange(5) != v
            for sign, kind in ((-1, "wide"), (1, "tight")):
                goal = sign * np.log(SCALE)
                prediction = inputs[scene, f"v{v}_{kind}"]
                change = np.log(prediction / base)
                sensitivity = change[v] / goal
                common = np.median(change[mask], axis=0)
                rows.append(dict(scene=scene, view=v, direction=sign, expected_log_change=float(goal),
                                 log_change_by_view_xy=change.tolist(), sensitivity_xy=sensitivity.tolist(),
                                 mean_sensitivity=float(np.mean(sensitivity)),
                                 common_sensitivity=float(np.mean(common / goal)),
                                 mean_affected_log_error=float(np.mean(np.abs(change[v] - goal))),
                                 mean_untouched_absolute_log_change=float(np.mean(abs(change[mask])))))
    return dict(overall=summary(rows), fixed_focal_affected_log_error=float(np.log(SCALE)),
                by_scene={s: summary([r for r in rows if r["scene"] == s]) for s in scenes},
                by_direction={str(n): summary([r for r in rows if r["direction"] == n]) for n in (-1, 1)},
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build(read(PUBLIC / "observations.json"))
    path = PUBLIC / "crop_response.json"
    if args.write:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        from verify_masked_camera_gradients import compare
        stored = read(path)
        archived_rows, fresh_rows = stored.pop("rows"), result.pop("rows")
        compare(stored, result)
        if len(archived_rows) != len(fresh_rows):
            raise ValueError("Changed intervention grid")
        for archived, fresh in zip(archived_rows, fresh_rows):
            for key in ("log_change_by_view_xy", "sensitivity_xy"):
                np.testing.assert_allclose(archived.pop(key), fresh.pop(key), atol=1e-12, rtol=1e-12)
            compare(archived, fresh)
    print(json.dumps(result["overall"]))


if __name__ == "__main__":
    main()
