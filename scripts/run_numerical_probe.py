"""Observe RNG, adapter initialization and gradients without editing baseline code.

Instrumentation changes synchronization and timing; do not use timings as a benchmark.
Pass the ordinary run_baseline.py CLI arguments. Fresh process for every repeat.
"""
import hashlib
import json
import os
import pathlib
import sys

import run_baseline as base
import numpy as np
import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def rng_record():
    return {"cpu": digest(torch.get_rng_state()),
            "cuda": [digest(state) for state in torch.cuda.get_rng_state_all()]}


def main():
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    output = base.ROOT / "runs" / run_id
    original_setup = base.tco_module.TCO_VGGT_LoRA.backbone_peft_setup
    record = {"probe_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "timing_is_instrumented": True, "steps": [],
              "variant": os.environ.get("TCO_NUMERICAL_VARIANT", "observe_original")}
    variant = record["variant"]
    assert variant in {"observe_original", "disable_mv", "strict_deterministic"}
    if variant == "strict_deterministic":
        torch.use_deterministic_algorithms(True)
    if variant == "disable_mv":
        original_loss = base.tco_module.compute_tco_losses
        def loss_without_mv(*args, **kwargs):
            kwargs["lambda_mv_consistency"] = 0.0
            return original_loss(*args, **kwargs)
        base.tco_module.compute_tco_losses = loss_without_mv

    def save():
        (output / "numerical_probe.json").write_text(json.dumps(record, indent=2) + "\n")

    def observed_setup(model, *args, **kwargs):
        record["rng_before_setup"] = rng_record()
        optimizer, params = original_setup(model, *args, **kwargs)
        record["rng_after_setup"] = rng_record()
        names_by_id = {id(p): name for name, p in model.named_parameters()}
        record["parameters"] = [{"name": names_by_id[id(p)], "dtype": str(p.dtype),
                                 "shape": list(p.shape), "initial_sha256": digest(p)} for p in params]
        def before_step(opt, unused_args, unused_kwargs):
            step = len(record["steps"])
            entry = {"step": step, "rng": rng_record(),
                     "gradient_sha256": [digest(p.grad) if p.grad is not None else None for p in params]}
            if step == 0:
                np.savez_compressed(output / "first_gradients.npz", **{
                    f"p{i:04d}": p.grad.detach().float().cpu().numpy() for i, p in enumerate(params) if p.grad is not None})
            record["steps"].append(entry)
            save()
        optimizer.register_step_pre_hook(before_step)
        save()
        return optimizer, params

    base.tco_module.TCO_VGGT_LoRA.backbone_peft_setup = observed_setup
    base.main()


if __name__ == "__main__":
    main()
