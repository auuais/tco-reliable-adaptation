"""Train one shared suffix initialization from a sealed, local ARKit Training ledger."""
import argparse
import gc
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import run_experiment as base
from run_tco_courtyard_camera import seed_model
from run_tco_final_ln_model import CAP, inputs, warm_local
from run_tco_frozen_prefix import CHECKPOINT
from tco_final_ln_protocol import ROOT, read, seal, sha
from tco_frozen_prefix import install, place
from tco_native_stalls_quality_capacity_protocol import GSPLAT
from tco_two_track_distill import ARMS, NamedAdam, factors_digest, normalizers, teacher_terms, training_order
from tco_two_track_hybrid import stage_number


class WindowFinished(Exception):
    """Complete the isolated scene before an unused post-update readout."""


def window(job, arm, idx, out, bank, norms, calibration):
    number = job["mount"]
    data, signature = inputs(Path(f"/inputs/{number}.npz"), job["input_file_sha256"], False)
    if signature != job["input_sha256"]:
        raise ValueError("Wrong training RGB/rig contents")
    target = None
    if arm == "Distilled":
        file = Path(f"/teachers/{number}.npz")
        if sha(file) != job["predictions_sha256"]:
            raise ValueError("Full40 teacher changed")
        with np.load(file, allow_pickle=False) as archive:
            if set(archive.files) != {"canonical", "depth", "depth_conf", "extrinsics", "intrinsics"}:
                raise ValueError("Unexpected teacher prediction fields")
            target = {key: torch.from_numpy(archive[key]).cuda() for key in ("extrinsics", "depth")}
    cfg = base.OmegaConf.load(ROOT / "configs/7scenes_baseline.yaml")
    cfg.tco_steps, cfg.tco_lr, cfg.seed = 1, .0001, 42 + idx
    seed_model(42 + idx)
    started = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT, enable_track=False).eval()
    path = install(model, "cached12", 42)
    place(model, path)
    frozen = [(p, p._version) for p in model.parameters()]
    original_setup = model.backbone_peft_setup
    native_loss = base.tco_module.compute_tco_losses
    record = dict(room=job["room"], visit=job["visit"], input_sha256=signature,
                  teacher_sha256=job.get("predictions_sha256") if target else None, sample_seed=42+idx)
    initial_updates = bank.updates

    def setup(*args, **kwargs):
        opt, params = original_setup(*args, **kwargs)
        factors = {n:p for n,p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))}
        if (len(factors) != 192 or len(params) != 192 or any(not 12 <= stage_number(n) < 24 for n in factors)):
            raise ValueError("Training suffix does not match deployment")
        bank.install(factors, opt)

        def finish(_opt, _args, _kwargs):
            bank.capture(factors, _opt)
            raise WindowFinished

        opt.register_step_post_hook(finish)
        return opt, params

    model.backbone_peft_setup = setup

    def loss(*args, **kwargs):
        if (kwargs["depths"] is not None or kwargs["depth_masks"] is not None or kwargs["lambda_depth"]
                or kwargs["mv_depth_weight"] or kwargs["mv_normal_weight"] or kwargs["mv_rgb_weight"] <= 0):
            raise ValueError("Unexpected original TCO loss configuration")
        if target is None:
            total, terms = native_loss(*args, **kwargs)
        else:
            rig = kwargs["gt_extrinsics"][0]
            depth, camera, support = teacher_terms(kwargs["predictions"], target, rig)
            record.update(raw_depth=float(depth.detach()), raw_camera=float(camera.detach()), teacher_valid_pixels=support)
            if calibration:
                raise WindowFinished
            total = .5 * (depth / norms[0] + camera / norms[1])
            terms = dict.fromkeys(("pose_rot_loss", "pose_trans_loss", "intrinsics_loss",
                                   "depth_loss", "mv_consistency_loss", "redundancy_loss"), total.detach().new_zeros(()))
        if not torch.isfinite(total):
            raise ValueError("Nonfinite offline loss")
        record["outer_loss"] = float(total.detach())
        return total, terms

    with patch.object(base.tco_module, "compute_tco_losses", loss):
        try:
            base.infer_mv_pointclouds_w_priors_v2(data, model, cfg, sample_idx=1, scene_name=job["room"])
        except WindowFinished:
            pass
        else:
            raise ValueError("Training window ended without the registered boundary")
    if bank.updates != initial_updates + int(not calibration) or any(p._version != version for p, version in frozen):
        raise ValueError("Incorrect optimizer count or changed frozen base")
    torch.cuda.synchronize()
    record.update(wall_seconds=time.perf_counter()-started, update=bank.updates, calibration=calibration,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if not 0 < record["peak_allocated_bytes"] <= CAP:
        raise ValueError("GPU cap exceeded")
    seal(out / (f"normalization_{idx+1:02d}.json" if calibration else f"update_{idx+1:02d}.json"), record)
    return record


def run(arm, plan, out):
    from gsplat.cuda._backend import _C
    if sha(_C.__file__) != GSPLAT:
        raise ValueError("Unexpected renderer binary")
    warm_local()
    jobs = training_order(read(plan)["jobs"])
    if [j["mount"] for j in jobs] != list(range(6)):
        raise ValueError("Invalid isolation mounts")
    started = time.perf_counter()
    bank = NamedAdam()
    norms, calibrations, training = None, [], []

    def one(job, idx, calibration):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return window(job, arm, idx, out, bank, norms, calibration)

    if arm == "Distilled":
        calibrations = [one(job, i, True) for i, job in enumerate(jobs)]
        norms = normalizers([(r["raw_depth"], r["raw_camera"]) for r in calibrations])
    for i, job in enumerate(jobs * 4):
        training.append(one(job, i, False))
    final = factors_digest(bank.values)
    if bank.updates != 24 or len(bank.values) != 192 or final == bank.initial:
        raise ValueError("Frozen or incomplete training")
    destination = out / "initialization.safetensors"
    if destination.exists():
        raise ValueError("Persistent suffix artifact already exists")
    save_file(bank.values, str(destination))
    if factors_digest(load_file(str(destination))) != final:
        raise ValueError("Invalid suffix serialization")
    receipt = dict(arm=arm, inputs_sha256=sha(plan), initialized_factors_sha256=bank.initial,
                   final_factors_sha256=final, initialization_sha256=sha(destination), factors=192,
                   optimizer_steps=24, training_rows=training, normalization_rows=calibrations,
                   depth_camera_normalizers=norms, offline_wall_seconds=time.perf_counter()-started,
                   training_window_seconds=sum(r["wall_seconds"] for r in training),
                   calibration_window_seconds=sum(r["wall_seconds"] for r in calibrations), rental_compute_usd=0)
    seal(out / "complete.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.arm, args.plan, args.out)
    except Exception as exc:
        seal(args.out / "failed.json", dict(arm=args.arm, error_type=type(exc).__name__, reason=str(exc)))
        raise
