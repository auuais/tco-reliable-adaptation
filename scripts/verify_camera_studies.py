"""Check saved study identities, paired comparisons and aggregate decisions on CPU."""
import json
import math
from pathlib import Path
from statistics import mean

from evaluation_stability import sha

ROOT = Path(__file__).resolve().parents[1]


def load(study, name):
    return json.loads((ROOT / "results" / study / name).read_text())


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)


def main():
    for study, counts in (("camera_depth_swap_v12", (65, 45)), ("track_camera_selection_v13", (48, 140))):
        seal = load(study, "seal.json")
        summary = load(study, "summary.json")
        assert seal["manifest_sha256"] == sha(ROOT / "manifests" / (study + ".json"))
        assert summary["seal_sha256"] == sha(ROOT / "results" / study / "seal.json")
        assert (len(seal["rows"]), len(seal["entries"])) == counts
        assert summary["unique_evaluations"] == len(seal["entries"])
        assert load(study, "archive.json")["verified_local_copies"] == 2
    swap = load("camera_depth_swap_v12", "summary.json")
    for mode, group in swap["modes"].items():
        rows = [r for r in swap["rows"] if r["mode"] == mode and r["repeat"] == 0]
        assert len({(r["scene"], r["seed"]) for r in rows}) == 9
        assert close(mean(r["score"] for r in rows), group["score"])
    assert swap["comparable_endpoints"] == (swap["endpoint_max_relative_distance_difference"] <= .005)
    track = load("track_camera_selection_v13", "summary.json")
    seal = load("track_camera_selection_v13", "seal.json")
    cases = {(r["scene"], r["profile"], r["perturbation_seed"]): r for r in seal["rows"]}
    assert len(cases) == 48 and len(track["rows"]) == 48 * 6
    for row in track["rows"]:
        case = cases[(row["scene"], row["profile"], row["seed"])]
        assert row["key"] == case["outputs"][row["method"]]
    for group, result in track["decisions"].items():
        rows = [r for r in track["rows"] + track["matched_clean_tco_scene_means"]
                if (r["profile"] == "clean") == (group == "clean")]
        methods = {}
        for method in {r["method"] for r in rows}:
            values = [r for r in rows if r["method"] == method]
            assert len(values) == (3 if group == "clean" else 45)
            methods[method] = mean(r["score"] for r in values)
            assert close(methods[method], track["tables"][group][method]["score"])
        best = min(sorted(m for m in methods if m != "selected"), key=methods.get)
        assert best == result["best_fixed"]
        assert close(1 - methods["selected"] / methods[best], result["relative_gain"])
        assert result["passed"] == all(result[k] for k in ("gain_pass", "harm_pass", "scenes_pass"))
    assert track["passes_screen"] == all(r["passed"] for r in track["decisions"].values())
    mixture = load("track_camera_selection_v13", "mixture.json")
    for m, v in mixture["scores"].items():
        assert close(v, (track["tables"]["clean"][m]["score"] + track["tables"]["all_noisy"][m]["score"]) / 2)
    frames = load("sensor_frame_v14", "summary.json")
    assert len(frames["poses"]) == 15 and len(frames["rows"]) == 27
    assert sum(r.get("frames_used", 0) for r in load("sensor_frame_v14", "alternative_rgb_sfm.json")["rows"]) == 15
    print("Verified camera-depth comparisons, track selection decisions and reference-frame records.")


if __name__ == "__main__":
    main()
