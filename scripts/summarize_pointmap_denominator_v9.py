"""Post-hoc sensitivity to the zero comparator, never a replacement for the v6 gate."""
import csv
import json
from pathlib import Path
from statistics import mean
from evaluation_stability import dump

ROOT = Path(__file__).resolve().parents[1]
def main():
    audit = ROOT / "results/pointmap_precision_v9"
    rows = json.loads((audit / "summary.json").read_text())["rows"]
    runs = list(csv.DictReader((ROOT / "results/repeat_calibration_v6/per_run.csv").open()))
    keys = ["accuracy_mean_m", "completeness_mean_m"]
    cells = [r for r in runs if r["condition"] == "both" and int(r["execution_repeat"]) == 0]
    assert len(cells) == 27
    bymode = {}
    for mode in ["saved_zero", "upstream_cpu_float32"]:
        zeros = {r["scene"]:r["metrics"] for r in rows if r["mode"] == mode}
        variants = {}
        for variant in ["released","axis_nominal","axis_registration"]:
            local = [r for r in cells if r["calibration"] == variant]
            assert len(local) == 9
            ratios = [[float(r[k])/zeros[r["scene"]][k] for k in keys] for r in local]
            variants[variant] = dict(score=mean(mean(v) for v in ratios), harmful_cells=sum(max(v)>1.01 for v in ratios))
        bymode[mode] = variants
    dump(audit / "denominator_sensitivity.json", dict(primary_v6_unchanged=True,
         note="Post-hoc only: fixed original 27 adapted GPU outputs and released evaluation reference, same CPU zero rescore or reconstructed FP32 zero.",
         unit="3 scenes by 3 seeds within condition; 1 means equal mean normalized distance to the scene zero",
         results=bymode))
if __name__ == "__main__":main()
