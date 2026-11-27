"""CPU-only matched surface score, with immutable prediction and evaluator provenance."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "upstream_Umeyama_then_rigid_ICP_0.1m_then_nearest_neighbor"
REFERENCE_FILES = (
    "scripts/run_experiment.py", "third_party/TCO/evaluation/mv_recon/utils.py",
    "scripts/run_tco_frozen_prefix.py", "scripts/run_tco_courtyard_camera.py",
)
# This numerical check is on an independent CPU evaluation of the same prediction in mm.
RESCORE_ATOL_MM = 1e-4


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def expected_names(policy, steps):
    require(policy in {"none","both","scale","bias"}, "Unknown LN parameter policy")
    if not steps:
        require(policy == "none", "Zero anchor cannot update pretrained weights")
        return set(), set()
    factors = {f"aggregator.{kind}_blocks.{i}.{target}.lora_{factor}"
               for kind in ("frame", "global") for i in range(24)
               for target in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2") for factor in ("A","B")}
    vectors = {"none": (), "scale": ("weight",), "bias": ("bias",), "both": ("weight","bias")}[policy]
    ln = {f"aggregator.global_blocks.23.norm{i}.{kind}" for i in (1,2) for kind in vectors}
    return factors, ln


def check_optimizer(receipt):
    steps, policy, opt = receipt["steps"], receipt["ln_policy"], receipt["optimizer"]
    require(type(steps) is int and steps >= 0 and receipt["completed_adam_updates"] == steps,
            "Incomplete or unregistered Adam step count")
    lora, ln = expected_names(policy, steps)
    for key, names in (("lora_names", lora), ("ln_names", ln)):
        received = opt[key]
        require(len(received) == len(names) and set(received) == names, f"Wrong named parameter in {key}")
    require(not opt["frozen_mutations"], "Mutation outside allowed parameter set")
    counts = opt["named_counts"]
    require(set(counts) == lora | ln and all(v == steps for v in counts.values()),
            "Missing named real Adam update")
    require(len(opt["steps"]) == steps, "Incomplete step records")
    for index, update in enumerate(opt["steps"], 1):
        require(update["update"] == update["counter_min"] == update["counter_max"] == index
                and update["lora_count"] == len(lora) and update["ln_count"] == len(ln),
                "Per-update optimizer counters differ from registration")
    if steps:
        initial, final = opt["ln_initial_sha256"], opt["ln_final_sha256"]
        _, all_ln = expected_names("both", 1)
        require(set(initial) == all_ln == set(final), "Missing final-block pretrained tensor receipt")
        moved = {name for name in all_ln if initial[name] != final[name]}
        require(moved == (ln if receipt["ln_lr"] > 0 else set()), "Pretrained tensor displacement mismatch")


def check_against(score, other):
    for k in ("room", "arm", "seed", "model_file_sha256", "input_sha256", "reference_file_sha256",
              "reference_sha256", "predictions_sha256", "model_receipt_sha256",
              "valid_points", "evaluator_sha256", "protocol"):
        require(score[k] == other[k], f"Independent scorer provenance mismatch for {k}")
    diffs = {k: abs(score[k]-other[k]) for k in ("accuracy_mm","completeness_mm")}
    require(all(v <= RESCORE_ATOL_MM for v in diffs.values()), "Independent rescore disagreed in mm")
    return diffs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "model", "reference", "out"):
        p.add_argument(f"--{name}", required=True, type=Path)
    for name in ("input-file-sha", "reference-file-sha"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--against", type=Path)
    args = p.parse_args()

    import numpy as np
    import torch
    from run_tco_courtyard_camera import seed_model
    from run_tco_frozen_prefix import signature
    from run_experiment import evaluate

    require(not torch.cuda.is_available(), "Scoring requires CPU-only execution")
    receipt_path, prediction = args.case / "complete.json", args.case / "prediction.npz"
    receipt = json.loads(receipt_path.read_text())
    check_optimizer(receipt)
    require(digest(args.model) == args.input_file_sha and digest(args.reference) == args.reference_file_sha,
            "Locked input/reference file checksum mismatch")
    prediction_sha = digest(prediction)
    require(receipt["predictions_sha256"] == prediction_sha and receipt["input_file_sha256"] == args.input_file_sha,
            "Prediction/input file differs from model receipt")
    with np.load(args.model, allow_pickle=False) as source, np.load(args.reference, allow_pickle=False) as gt:
        require(set(source.files) == {"images","extrs","intrs"}
                and set(gt.files) == {"valid_mask","pointclouds"}, "Input/reference separation invalid")
        images, poses, intrinsics = (source[k] for k in ("images","extrs","intrs"))
        mask, surface = gt["valid_mask"], gt["pointclouds"]
    input_sha = signature(dict(rgb=images, poses=poses, intrinsics=intrinsics))
    require(receipt["input_sha256"] == input_sha, "Scored RGB/priors differ from the model run")
    reference_sha = signature(dict(mask=mask, surface=surface))
    with np.load(prediction, allow_pickle=False) as pred:
        points = pred["canonical"]
    require(points.shape == surface.shape == (5, images.shape[2], images.shape[3], 3)
            and images.shape[:2] == (5,3) and points.dtype == np.float32 and np.isfinite(points).all(),
            "Predicted and reference raster mismatch")
    require(receipt["readout"] == {"dtype":"float32", "shape":list(points.shape)}
            and receipt["native_depth_batches"] == [[5]] and receipt["cap_mib"] == 8960
            and receipt["rental_compute_usd"] == 0 and receipt["runtime"]["gpu"] == "NVIDIA GeForce RTX 3080",
            "Production evidence differs from five-view absolute-cap setting")
    require(mask.shape == points.shape[:-1] and mask.dtype == np.bool_ and np.isfinite(surface[mask]).all()
            and all(mask[v].any() for v in range(5)), "Invalid reference support")
    eval_digests = {name: digest(ROOT/name) for name in REFERENCE_FILES}
    seed_model(receipt["seed"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    require(not args.out.exists(), "Scoring output already exists; preserve first or use a fresh rescore directory")
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        staged = Path(tmp)/"result"
        staged.mkdir()
        metrics = evaluate(points, dict(images=torch.from_numpy(images), pointclouds=surface, valid_mask=mask), staged)
        score = dict(room=receipt.get("room",receipt.get("scene")), seed=receipt["seed"], arm=receipt["arm"],
                     scoring_cpu=True, model_file_sha256=args.input_file_sha, input_sha256=input_sha,
                     reference_file_sha256=args.reference_file_sha, reference_sha256=reference_sha,
                     predictions_sha256=prediction_sha, model_receipt_sha256=digest(receipt_path),
                     valid_points=metrics["valid_points"], protocol=metrics["protocol"], evaluator_sha256=eval_digests,
                     accuracy_mm=1000*metrics["accuracy_mean_m"], completeness_mm=1000*metrics["completeness_mean_m"])
        require(score["valid_points"] == int(mask.sum()) and score["protocol"] == PROTOCOL and score["room"],
                "Scorer provenance or support mismatch")
        require(all(math.isfinite(score[k]) and score[k]>=0 for k in ("accuracy_mm","completeness_mm")),
                "Invalid geometry score")
        if args.against is not None:
            other = json.loads(args.against.read_text())
            score["independent_rescore_mm_difference"] = check_against(score, other)
            score["first_score_sha256"] = digest(args.against)
        (staged/"score.json").write_text(json.dumps(score, indent=2, allow_nan=False)+"\n")
        staged.rename(args.out)
    print("Verified CPU score", score["room"], score["arm"], score["seed"], flush=True)


if __name__ == "__main__":
    main()
