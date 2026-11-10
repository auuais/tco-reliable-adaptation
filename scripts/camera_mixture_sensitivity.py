"""Expected reconstruction error as the fraction of clean camera priors changes."""
from pathlib import Path

from evaluation_stability import dump, sha
from score_saved_geometry import read


def strict_winning_interval(differences):
    """Intersect linear inequalities d_noisy + p * (d_clean - d_noisy) < 0."""
    lower, upper = 0., 1.
    for clean, noisy in differences:
        slope = clean - noisy
        if slope == 0:
            if noisy >= 0:
                return None
        elif slope > 0:
            upper = min(upper, -noisy / slope)
        else:
            lower = max(lower, -noisy / slope)
    if lower >= upper:
        return None
    return dict(lower=lower, upper=upper,
                includes_zero=lower == 0 and all(n < 0 for _, n in differences),
                includes_one=upper == 1 and all(c < 0 for c, _ in differences))


def main():
    root = Path(__file__).resolve().parents[1]
    public = root / "results/track_camera_selection_v13"
    source = public / "summary.json"
    study = read(source)
    clean = study["tables"]["clean"]
    noisy = study["tables"]["all_noisy"]
    common = sorted(set(clean) & set(noisy))
    scores = {m: .5 * (clean[m]["score"] + noisy[m]["score"]) for m in common}
    best = min((m for m in common if m != "selected"), key=scores.get)
    diffs = [(clean["selected"]["score"] - clean[m]["score"],
              noisy["selected"]["score"] - noisy[m]["score"]) for m in common if m != "selected"]
    metrics = {m: {k: .5 * (clean[m]["metrics"][k] + noisy[m]["metrics"][k])
                   for k in clean[m]["metrics"]} for m in common}
    scenes = {s: {m: .5 * (clean[m]["scenes"][s] + noisy[m]["scenes"][s]) for m in common}
              for s in clean["selected"]["scenes"]}
    dump(public / "mixture.json", dict(summary_sha256=sha(source),
         plan_sha256=sha(root / "manifests/track_camera_selection_v13_mixture.json"), scores=scores,
         mean_metrics=metrics, scene_scores=scenes, best_fixed=best,
         relative_gain=1 - scores["selected"] / scores[best],
         clean_fraction_where_selected_beats_all_fixed=strict_winning_interval(diffs)))


if __name__ == "__main__":
    main()
