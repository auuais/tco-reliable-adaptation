"""Verify arithmetic, coverage and recorded provenance using the published evidence only."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cpu():
    evidence = read(ROOT / "results/selected_readouts/cached_operand_cpu.json")
    if (evidence["implementation_sha256"] != sha(ROOT / "scripts/frozen_linear_storage.py")
            or evidence["test_sha256"] != sha(ROOT / "scripts/check_backbone_operands.py")
            or evidence["adapter_tensors"] != 384):
        raise ValueError("Unexpected CPU control provenance")
    recorded = evidence["results"]
    fields = {"step", "loss", "gradient_norm_before_clip", "head_sha256", "features_sha256",
              "clipped_gradient_sha256", "updated_adapter_sha256"}
    rows = [recorded[k]["updates"] for k in ("normal", "cached")]
    if any([r["step"] for r in arm] != [0, 1] for arm in rows):
        raise ValueError("Incomplete two-update CPU control")
    equal = all(a[k] == b[k] for a, b in zip(*rows) for k in fields)
    if not equal or evidence["bitwise_match"] != equal:
        raise ValueError("CPU update agreement failed")


def verify_bridge():
    folder = ROOT / "results/staged_heads_bridge"
    plan = ROOT / "manifests/staged_heads_bridge.json"
    seal, summary = (read(folder / name) for name in ("seal.json", "summary.json"))
    if (seal["manifest_sha256"] != sha(plan) or summary["seal_sha256"] != sha(folder / "seal.json")
            or summary["arms"] != seal["arms"] or summary["paired_executions"] != 1):
        raise ValueError("Published bridge provenance changed")
    p = read(plan)
    for rel, signature in p["sources"].items():
        if sha(ROOT / rel) != signature:
            raise ValueError("Frozen bridge input changed: " + rel)
    arms, refs, scene = set(p["arms"]), set(p["references"]), p["scene"]
    grid = {(r["method"], r["reference"]): r for r in summary["rows"]}
    if len(grid) != len(summary["rows"]) or set(grid) != {(a,r) for a in arms for r in refs}:
        raise ValueError("Incomplete pair of full-size evaluations")
    for (arm, ref), row in grid.items():
        entry = seal["entries"][seal["maps"][scene][arm]]
        if (row["scene"] != scene or row["prediction_sha256"] != entry["sha256"]
                or not all(math.isfinite(v) and v > 0 for v in row["metrics"].values())):
            raise ValueError("Invalid bridge prediction or metrics")
    for ref in refs:
        a, b = (grid[(arm, ref)] for arm in ("original", "staged"))
        if a["reference_sha256"] != b["reference_sha256"] or a["metrics"].keys() != b["metrics"].keys():
            raise ValueError("The comparison does not use one common reference")
        expected = {k:b["metrics"][k] / v - 1 for k,v in a["metrics"].items()}
        if summary["relative_change"][ref] != expected:
            raise ValueError("Published metric differences did not reproduce")


if __name__ == "__main__":
    verify_cpu()
    verify_bridge()
    print("Verified two recorded CPU updates and four full-size matched-pair scores.")
