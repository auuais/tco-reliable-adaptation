"""Record adaptation milestones and replay them through the pinned TCO readout."""

import hashlib
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from run_profiled_d0 import read, sha, write_new
from run_tco_frozen_prefix import FIELDS, export, signature
from utils.geometry import unproject_depth_to_world_points_torch


MILESTONES = (0, 1, 5, 10, 39, 40)
COUNT = 384


def lora_parameters(model):
    factors = {name: factor for name, factor in model.named_parameters()
               if name.endswith((".lora_A", ".lora_B"))}
    if len(factors) != COUNT:
        raise ValueError(f"Expected {COUNT} adapter factors, found {len(factors)}")
    return factors


def factor_arrays(factors):
    if any(param.dtype != torch.float32 for param in factors.values()):
        raise ValueError("Milestone serialization requires the pinned FP32 adapter weights")
    return {name: export(param) for name, param in factors.items()}


def rng_hash(state):
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def stage_norms(arrays):
    sums = {}
    for name, array in arrays:
        match = re.search(r"\.(frame|global)_blocks\.(\d+)\.", name)
        if match is None:
            raise ValueError(f"Unexpected full-model adapter key {name}")
        stage = f"{match[1]}/{int(match[2]):02d}"
        flattened = array.astype(np.float64, copy=False).ravel()
        sums[stage] = sums.get(stage, 0.0) + float(np.dot(flattened, flattened))
    if len(sums) != 48:
        raise ValueError("Expected adapter factors in all 48 frame/global blocks")
    return {stage: value ** 0.5 for stage, value in sorted(sums.items())}


def prediction_arrays(outputs, points=None):
    values = {name: export(outputs[name]) for name in FIELDS}
    if "pose_enc" in outputs:
        values["pose_enc"] = export(outputs["pose_enc"])
    if points is not None:
        values["canonical"] = points
    if any(not np.isfinite(array).all() for array in values.values()):
        raise ValueError("A diagnostic milestone contains nonfinite predictions")
    return values


def write_arrays(directory, name, arrays):
    destination = directory / f"{name}.npz"
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return dict(file=destination.name, sha256=sha(destination), arrays_sha256=signature(arrays))


def read_arrays(directory, row):
    path = directory / row["file"]
    if row["file"] != Path(row["file"]).name or sha(path) != row["sha256"]:
        raise ValueError(f"Milestone file changed: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if signature(arrays) != row["arrays_sha256"]:
        raise ValueError(f"Milestone array payload changed: {path.name}")
    return arrays


class CourtyardTrace:
    """Observe the real loss and optimizer without keeping any CUDA activations."""

    def __init__(self, out):
        self.out = Path(out)
        self.directory = self.out / "milestones"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model = None
        self.parameters = None
        self.rows = []
        self.snapshots = {}
        self.pending_before = None
        self.updates = 0

    def bind(self, model):
        self.model = model

    def loss(self, *, outputs, terms, images, total, rng_before):
        if len(self.rows) != self.updates:
            raise ValueError("Trace observed a loss without an intervening Adam update")
        rng_after = torch.cuda.get_rng_state()
        generator = torch.Generator(device=images.device).set_state(rng_before)
        permutation = torch.randperm(images.shape[1], device=images.device, generator=generator).tolist()
        if not torch.equal(generator.get_state(), rng_after):
            raise ValueError("Trace cannot account for renderer RNG consumption")
        values = {"total": total, **terms}
        scalars = torch.stack([value.detach().float() for value in values.values()]).tolist()
        row = dict(completed_updates=self.updates, losses=dict(zip(values, scalars)),
                   render_permutation=permutation, renderer_rng_before=rng_hash(rng_before),
                   renderer_rng_after=rng_hash(rng_after))
        self.rows.append(row)
        if self.updates in MILESTONES:
            if self.parameters is None:
                self.parameters = lora_parameters(self.model)
            self.pending_before = factor_arrays(self.parameters)
            self.save(self.updates, outputs, self.pending_before, "adaptation_autocast")

    def before_clip(self):
        if self.pending_before is None:
            return
        if any(parameter.grad is None for parameter in self.parameters.values()):
            raise ValueError("A full-model adapter is missing its gradient")
        self.rows[-1]["preclip_gradient_l2_by_block"] = stage_norms(
            (name, export(parameter.grad)) for name, parameter in self.parameters.items())

    def after_step(self):
        if len(self.rows) != self.updates + 1:
            raise ValueError("Trace observed an Adam update without its real loss")
        if self.pending_before is not None:
            self.rows[-1]["adam_increment_l2_by_block"] = stage_norms(
                (name, export(parameter) - self.pending_before[name])
                for name, parameter in self.parameters.items())
            self.pending_before = None
        self.updates += 1

    def save(self, step, outputs, factors, precision, points=None):
        arrays = prediction_arrays(outputs, points)
        self.snapshots[str(step)] = dict(
            completed_updates=step, precision=precision,
            source_prediction_dtypes={key: str(outputs[key].dtype) for key in arrays if key != "canonical"},
            prediction=write_arrays(self.directory, f"prediction_{step:02d}", arrays),
            adapters=write_arrays(self.directory, f"adapters_{step:02d}", factors))

    def final(self, predictions, points):
        if self.updates != 40 or len(self.rows) != 40 or self.pending_before is not None:
            raise ValueError("Trace did not observe 40 completed Adam updates")
        self.save(40, predictions, factor_arrays(self.parameters), "production_final_readout", points)
        if tuple(map(int, self.snapshots)) != MILESTONES:
            raise ValueError("Incomplete Courtyard milestone roster")
        receipt = dict(updates=40, milestones=self.snapshots, loss_trajectory=self.rows,
                       rental_compute_usd=0)
        write_new(self.out / "trace.json", receipt)
        return receipt


@torch.no_grad()
def replay_snapshots(model, supplied, origin, out, cfg):
    """Load sealed adapters in a new process and keep the production readout math."""
    origin, out = Path(origin), Path(out)
    source = read(origin / "trace.json")
    if tuple(map(int, source["milestones"])) != MILESTONES or source["updates"] != 40:
        raise ValueError("Common readout requires a completed registered trace")
    directory = out / "milestones"
    directory.mkdir(parents=True, exist_ok=False)
    images = supplied["images"].to(cfg.device).unsqueeze(0)
    mixed_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast(cfg.device, dtype=mixed_dtype):
        # The production pipeline extracts before inserting adapters. Its final
        # aggregate uses this same patch and the enclosing autocast context.
        patch_tokens = model.extract_patch_tokens(images)
        model.backbone_peft_setup(lr=cfg.tco_lr, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                 lora_target_modules=cfg.lora_target_modules,
                                 lora_dropout=cfg.lora_dropout, merge_weights=False)
        model.disable_lora_gradients()
        params = lora_parameters(model)
        restored = {}
        with patch.object(model, "extract_patch_tokens", lambda _images: patch_tokens):
            for step, row in source["milestones"].items():
                adapters = read_arrays(origin / "milestones", row["adapters"])
                if set(adapters) != set(params):
                    raise ValueError("Snapshot inventory differs from the pinned model")
                for name, parameter in params.items():
                    arr = torch.from_numpy(adapters[name])
                    if arr.shape != parameter.shape or arr.dtype != parameter.dtype:
                        raise ValueError(f"Snapshot dtype or shape differs for {name}")
                    parameter.copy_(arr)
                # Cached autocast parameter casts must reflect this snapshot.
                torch.clear_autocast_cache()
                _, predictions = model.forward_original_vggt(images)
                with torch.amp.autocast(cfg.device, enabled=False):
                    points = unproject_depth_to_world_points_torch(
                        *(predictions[key].float() for key in ("depth", "extrinsics", "intrinsics")))
                arrays = prediction_arrays(predictions, export(points[0]))
                restored[step] = dict(
                    completed_updates=int(step), precision="production_final_readout",
                    adapter_arrays_sha256=row["adapters"]["arrays_sha256"],
                    prediction=write_arrays(directory, f"readout_{int(step):02d}", arrays))
    expected = read_arrays(origin / "milestones", source["milestones"]["40"]["prediction"])
    parity = {}
    for name, old in expected.items():
        new = arrays[name]
        delta = new.astype(np.float64) - old.astype(np.float64)
        parity[name] = dict(
            bitwise_equal=bool(np.array_equal(old, new)),
            max_abs=float(np.max(np.abs(delta))),
            relative_l2=float(np.linalg.norm(delta.ravel()) /
                              max(np.linalg.norm(old.astype(np.float64).ravel()), 1e-12)))
    if any(row["relative_l2"] > 1e-5 for row in parity.values()):
        raise ValueError(f"Common readout does not reproduce the step-40 production outputs: {parity}")
    receipt = dict(updates=0, source_trace_sha256=sha(origin / "trace.json"), milestones=restored,
                   production_step40_parity=parity, rental_compute_usd=0)
    write_new(out / "readout.json", receipt)
    return receipt
