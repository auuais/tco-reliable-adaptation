"""Prospective roster, input locks and follow-up arms for the five-view study."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_final_global_layernorm_v1"
RAW = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
PREP = Path("E:/ResearchData/TCO/preparation") / STUDY
MANIFEST = ROOT / "manifests" / (STUDY + ".json")
LOCK = ROOT / "manifests" / (STUDY + "_stanford_lock.json")
IMAGE = "sha256:541dd1e09a32b8da7851929e73810882415aa30967b0ae2acd5473c9cf5ec3f1"
ALLOCATOR = "garbage_collection_threshold:0.75,expandable_segments:True"
SEEDS = [42, 43, 44]
ORDER_SEED = 2026112802
ARMS = {
    "L10": (10, "none", 0), "LN10": (10, "both", 3e-5), "L40": (40, "none", 0),
    "ZERO": (0, "none", 0), "L20": (20, "none", 0), "LN20": (20, "both", 3e-5),
    "LN10_scale": (10, "scale", 3e-5), "LN10_bias": (10, "bias", 3e-5),
    "LN10_low": (10, "both", 1e-5), "L2": (2, "none", 0), "LN2": (2, "both", 3e-5),
    "LN2_zero": (2, "both", 0),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(path, record):
    """Accept identical replay and refuse to overwrite evidence."""
    path = Path(path)
    payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if json.loads(path.read_bytes()) != record:
            raise ValueError("Immutable record differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def job(room, seed, arm, role, profile="clean", tag=None):
    steps, ln_policy, ln_lr = ARMS[arm]
    key = tag or (room["slug"] + "_" + str(seed) + "_" + arm)
    return dict(**room, id=role + "_" + key, seed=seed, arm=arm, steps=steps,
                ln_policy=ln_policy, ln_lr=ln_lr, role=role, profile=profile,
                diagnostic=profile == "diagnostic")


def matrix(rooms, arms, role):
    rng = random.Random(ORDER_SEED)
    blocks = list(itertools.product(sorted(rooms, key=lambda r: r["room"]), SEEDS))
    rng.shuffle(blocks)
    permutations = list(itertools.permutations(arms))
    rng.shuffle(permutations)
    orders = [permutations[i % len(permutations)] for i in range(len(blocks))]
    rng.shuffle(orders)
    return [job(room, seed, arm, role) for (room, seed), order in zip(blocks, orders) for arm in order]


def registration():
    historical = read(ROOT / "manifests/tco_native_stalls_quality_capacity_v1.json")
    dev = read(RAW / "development_inputs.json")["rooms"]
    small, = read(RAW / "qualification_inputs.json")["rooms"]
    heads = next(r for r in dev if r["room"].startswith("heads"))
    chess = next(r for r in dev if r["room"].startswith("chess"))
    engineering = []
    for tag, profile, arm in (("bridge_fixed", "fixed_bridge", "L2"), ("standard_fixed", "fixed", "L2"),
                              ("zerolr_fixed", "fixed", "LN2_zero"), ("bridge_native", "native_bridge", "L2"),
                              ("standard_native", "diagnostic", "L2"), ("repeat_native", "diagnostic", "L2"),
                              ("zerolr_native", "diagnostic", "LN2_zero"), ("positive_native", "diagnostic", "LN2")):
        engineering.append(job(small if "fixed" in profile else heads, 42, arm, "engineering", profile, tag))
    return dict(
        study=STUDY, handoff_commit="b39c61c", handoff_sha256=sha(ROOT / "docs/TCO_FINAL_GLOBAL_LAYERNORM_HANDOFF.md"),
        pins={k:historical[k] for k in ("docker_image","upstream","torch","gpu","gsplat_sha256","checkpoint_sha256","allocator")},
        protocol=dict(views=5, width=518, seeds=SEEDS, schedule_seed=ORDER_SEED, cuda_cap_mib=8960,
                      rental_compute_usd=0, deadline_seconds=1200, lora_lr=0.0003, rank=4, alpha=16,
                      separate_clipping=[1,1], camera_checkpoint=True, render_mode="RGB+ED",
                      bootstrap_seed=20261128, bootstrap_draws=10000, fixed_qual_width=294,
                      fresh_rooms=["apt2/living","apt2/luke","all eligible office1 rooms in lexical order"],
                      min_formal_rooms=6, prelock_rule="First metadata-eligible window per immutable handoff; all eligible rooms",
                      timing="Synchronized scene, CPU checkpoint loaded through final FP32 readout, clean GPU, CPU scoring separate",
                      retry="One unchanged retry of provisioning/interruption; one diagnostic repeat for ambiguous watchdog. Preserve first outcome."),
        arms={k:dict(steps=v[0],ln_policy=v[1],ln_lr=v[2]) for k,v in ARMS.items()},
        thresholds=dict(LN10_vs_L10={"accuracy_strict":1.0,"completeness_inclusive":1.03},
                        LN10_vs_L40={"accuracy_inclusive":1.01,"completeness_inclusive":1.03,"time_strict":1.0},
                        gate="Both ratio estimate and one-sided 95% upper bound, joint intersection"),
        arkit=dict(commit="474aaa83d7b1c9080619d8485de2d7c885c01fc5",
                   metadata_sha256="5a817c0ee8150cea1de4fa9d6399880eaa646c2b62326c3314e674d201536c57",
                   downloader_sha256="1eca7fb832aab84c15da8ed5ffe38d4f1f7ffb4256fcfd755ef4b39a31b48137",
                   selection="Validation, ascending distinct numeric visit ID, lowest eligible video ID, audit project exposure, six, one window each. Fix metadata-only geometry rules on Training."),
        development_rooms=dev, engineering=engineering, development=matrix(dev, ["L10","LN10","L40"], "development"),
        duplicates=[job(chess,42,a,"duplicate") for a in ("L10","LN10","L40")],
        followups={"all_pass":["LN10_scale","LN10_bias"],"equal_update_pass_only":["L20","LN20"],
                   "other":["LN10_low"]},
        followup_gate="Follow handoff. No fresh-room tuning. At most one independently locked candidate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register","verify"))
    args = parser.parse_args()
    expected = registration()
    if args.action == "register":
        seal(MANIFEST, expected)
    if read(MANIFEST) != expected:
        raise ValueError("Registration replay mismatch")
    print("Verified", MANIFEST.name, sha(MANIFEST))


if __name__ == "__main__":
    main()
