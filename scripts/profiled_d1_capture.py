"""Bounded D1 teacher-first capture helpers; not a scientific launch entry point.

The future release controller must verify committed sources, input bytes, runtime,
ownership and D0 replay before calling these helpers. They neither load evaluation
labels nor fit/update an adapter. CPU fixtures exercise the ordering separately
from production model acceptance, which has not yet occurred.
"""
from contextlib import contextmanager
import hashlib
import itertools
import re
import time


PAIRS = tuple(itertools.permutations(range(5), 2))
ORIGINAL = "5d5ff52b270319abac6d3e202deec09d24e6b9013769147b167df2e87d93c69f"
INITIAL = "0ba2991166e023174d2dccfb5ea5fb749689501c3ce161f478c0baa88b8643ad"
FINAL = "e9f9f52fd89ad49b5457517c0d08afe2864f37b0dfe92245988a8358bcd84d12"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_views(rgb, archive_sha256):
    """Reproduce images-only views_for preprocessing and instance identities.

The execution controller configures the registered deterministic runtime before
preprocessing. No other archive member is required or opened by this function.
"""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from profiled_runtime import verify_runtime
    verify_runtime()
    require(isinstance(archive_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", archive_sha256),
            "Expected the byte-verified complete RGB archive hash")
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.float32 and
            rgb.shape == (5, 3, 392, 518) and np.isfinite(rgb).all(),
            "Expected the fixed images-only float32 raster")
    require(bool((rgb >= 0).all() and (rgb <= 1).all()), "Unexpected RGB normalization")
    with torch.no_grad():
        images = 2*functional.interpolate(torch.from_numpy(rgb), size=(384, 512),
            mode="bilinear", align_corners=False, antialias=True)-1
    return [dict(img=im[None], true_shape=np.array([[384, 512]], dtype=np.int32),
                 idx=i, instance=f"heads_base_{archive_sha256[:12]}_{i}")
            for i, im in enumerate(images)]


@contextmanager
def counted_method(model, name, maximum, guard):
    """Count actual calls, reject excess before execution, restore even on failure."""
    original = getattr(model, name)
    previous = model.__dict__.get(name)
    was_instance_attribute = name in model.__dict__
    counts = {"calls": 0}

    def observed(*args, **kwargs):
        guard()
        require(counts["calls"] < maximum, "D1 call cap exceeded: " + name)
        counts["calls"] += 1
        result = original(*args, **kwargs)
        guard()
        return result

    setattr(model, name, observed)
    try:
        yield counts
    finally:
        if was_instance_attribute:
            setattr(model, name, previous)
        else:
            delattr(model, name)


def _ordered_capture(ops, views, load_adapter, teacher_gate, guard, progress):
    """Shared order/call controller. Fake ops are only for generated CPU fixtures."""
    require(len(views) == 5, "D1 requires exactly five registered views")
    ops.require_frozen()
    original_state = ops.original_state()
    require(ops.digest(original_state) == ORIGINAL, "Unexpected pristine model state")
    with counted_method(ops.model, "_encode_image", 5, guard) as enc:
        features = ops.features(views)
    require(enc["calls"] == 5 and len(features) == 5, "Expected five encoder calls/features")
    feature_hash = ops.feature_digest(features)
    with counted_method(ops.model, "_decoder", 20, guard) as before:
        points0, confidence = ops.collect(features)
    require(before["calls"] == 20, "Expected twenty pristine decoder calls")
    depth0, fused_confidence = ops.fuse(points0, confidence)
    guard()
    # The callback persists comparison evidence before returning or raising.
    teacher_receipt = teacher_gate(depth0, fused_confidence)
    require(isinstance(teacher_receipt, dict) and teacher_receipt.get("pass_all") is True,
            "Teacher replay gate failed; trained adapter must remain unopened")
    progress("teacher_verified", {"encoder_calls": 5, "decoder_calls": 20})
    require(ops.digest(original_state) == ORIGINAL, "Pristine forward modified original state")
    require(ops.feature_digest(features) == feature_hash, "Teacher changed cached features")
    ops.reset_adapter_seed()
    modules = ops.install()
    named = ops.adapters()
    ops.validate_initial(modules, named)
    require(ops.digest(named) == INITIAL, "Unexpected initial adapter state")
    ops.freeze()
    ops.require_frozen()
    # This is the first access to final factors, after the persisted teacher gate.
    ops.assign(named, load_adapter())
    require(ops.digest(named) == FINAL, "Unexpected final P3 adapter state")
    with counted_method(ops.model, "_decoder", 20, guard) as after:
        points1, discarded_confidence = ops.collect(features)
    del discarded_confidence
    require(after["calls"] == 20, "Expected twenty final-state decoder calls")
    depth1, _ = ops.fuse(points1, confidence)
    guard()
    require(ops.digest(original_state) == ORIGINAL, "Final forward modified original state")
    require(ops.digest(named) == FINAL, "Final forward modified adapter state")
    require(ops.feature_digest(features) == feature_hash, "Final forward changed cached features")
    ops.require_frozen()
    metadata = dict(encoder_calls=5, pristine_decoder_calls=20, final_decoder_calls=20,
                    decoder_calls=40, backward_calls=0, optimizer_updates=0,
                    original_state_sha256=ORIGINAL, initial_adapter_sha256=INITIAL,
                    final_adapter_sha256=FINAL, feature_sha256=feature_hash,
                    final_confidence_discarded=True, teacher_verified_before_adapter_access=True,
                    rental_usd=0)
    progress("capture_complete", metadata)
    return dict(points0=points0, points1=points1, confidence=confidence,
                depth0=depth0, depth1=depth1, fused_confidence=fused_confidence,
                teacher_receipt=teacher_receipt, metadata=metadata)


def validate_pairs(points, confidence, *, shape=(384, 512), device_type="cuda"):
    """Production uses fixed FP32 self-pointmaps; tuple branch 1 is not fused."""
    import torch
    require(set(points) == set(PAIRS) == set(confidence), "Incomplete directed-pair inventory")
    common_device = None
    for pair in PAIRS:
        branches = points[pair]
        require(isinstance(branches, (tuple, list)) and len(branches) == 2,
                "Pair must expose the historical two-branch structure")
        xyz, conf = branches[0], confidence[pair]
        require(isinstance(xyz, torch.Tensor) and isinstance(conf, torch.Tensor), "Expected tensors")
        require(tuple(xyz.shape) == tuple(shape)+(3,) and tuple(conf.shape) == tuple(shape),
                "Unexpected pointmap/confidence raster")
        require(xyz.dtype == conf.dtype == torch.float32, "Fusion inputs must be float32")
        require(xyz.device == conf.device and xyz.device.type == device_type,
                "Pointmaps/confidence must use the same registered device")
        common_device = xyz.device if common_device is None else common_device
        require(xyz.device == common_device, "Mixed devices in canonical fusion")
        require(not xyz.requires_grad and not conf.requires_grad, "D1 capture must be graph free")
        require(bool(torch.isfinite(xyz).all()) and bool(torch.isfinite(conf).all()),
                "Nonfinite canonical fusion input")
        require(bool((conf > .999).all()), "Canonical confidence weight is not positive")


def fuse_common(points, pristine_confidence):
    """The single CUDA FP32 fusion path for B0/B1/F0/F1; no confidence variant."""
    import torch
    from profiled_network import fuse
    validate_pairs(points, pristine_confidence)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        depth, confidence = fuse(points, pristine_confidence)
    require(depth.dtype == confidence.dtype == torch.float32, "Fusion changed its dtype")
    require(bool(torch.isfinite(depth).all()) and bool(torch.isfinite(confidence).all()),
            "Nonfinite fused outputs")
    return depth, confidence


def tensor_sequence_digest(features):
    """Read-only byte identity including BF16 tensors, positions and true shapes."""
    import torch
    digest = hashlib.sha256()
    for i, feature in enumerate(features):
        require(isinstance(feature, (tuple, list)) and len(feature) == 3,
                "Unexpected encoded-feature schema")
        for j, value in enumerate(feature):
            require(isinstance(value, torch.Tensor) and not value.requires_grad,
                    "Features must be detached tensors")
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{i}:{j}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TorchCaptureOps:
    def __init__(self, model, lora_path):
        self.model, self.lora_path = model, lora_path

    def require_frozen(self):
        require(not self.model.training, "D1 model must stay in eval mode")
        require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),
                "D1 must not enable or accumulate gradients")

    def original_state(self):
        return list(self.model.state_dict().items())

    def digest(self, state):
        from mast3r_network_priors import digest
        return digest(state)

    def features(self, views):
        from mast3r_network_capture import training_features
        return training_features(self.model, views)

    def feature_digest(self, features):
        return tensor_sequence_digest(features)

    def collect(self, features):
        from profiled_network import collect_points
        result = collect_points(self.model, features, precision="bfloat16", pairs=PAIRS)
        validate_pairs(*result)
        return result

    def fuse(self, points, confidence):
        return fuse_common(points, confidence)

    def reset_adapter_seed(self):
        import torch
        torch.manual_seed(42)

    def install(self):
        from mast3r_network_priors import install
        result = install(self.model, self.lora_path)
        self.model.eval()
        return result

    def adapters(self):
        # Preserve named_parameters traversal order for historical digest equality.
        return [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]

    def validate_initial(self, modules, named):
        from profiled_d0_artifacts import MODULES
        require(len(modules) == 144 and set(modules) == set(MODULES), "Wrong LoRA placement")
        require(len(named) == 288 and sum(p.numel() for _, p in named) == 1032192,
                "Wrong LoRA factor population")
        expected = {m+s for m in MODULES for s in (".lora_A", ".lora_B")}
        require({n for n, _ in named} == expected, "Unexpected trainable parameter")
        require(all(not bool(p.detach().any()) for n, p in named if n.endswith(".lora_B")),
                "LoRA B factors must start at zero")

    def freeze(self):
        self.model.requires_grad_(False)

    def assign(self, named, factors):
        import numpy as np
        import torch
        require(set(factors) == {n for n, _ in named}, "Final factor names differ")
        for name, parameter in named:
            value = factors[name]
            require(isinstance(value, np.ndarray) and value.dtype == np.float32 and
                    tuple(value.shape) == tuple(parameter.shape) and np.isfinite(value).all(),
                    "Invalid final factor: " + name)
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(torch.as_tensor(factors[name], device=parameter.device))


class LocalCaptureGuard:
    """Inner numerical/resource check; the launcher must enforce an outer timeout."""
    def __init__(self):
        self.started = time.monotonic()

    def __call__(self):
        import torch
        from profiled_runtime import verify_runtime
        verify_runtime()
        require(time.monotonic()-self.started < 3600, "D1 local model wall cap exceeded")
        require(torch.cuda.max_memory_allocated()/2**20 <= 8192, "D1 allocated CUDA cap exceeded")
        require(torch.cuda.max_memory_reserved()/2**20 <= 9216, "D1 reserved CUDA cap exceeded")


def capture_training_paths(weights_dir, lora_path, views, load_adapter, teacher_gate, progress):
    """Called only after the future execution registry/release controller passes."""
    import random
    import numpy as np
    import torch
    from profiled_runtime import configure_runtime
    configure_runtime()
    from mast3r.model import AsymmetricMASt3R
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    guard = LocalCaptureGuard()
    model = AsymmetricMASt3R.from_pretrained(
        str(weights_dir), local_files_only=True).eval().cuda().requires_grad_(False)
    guard()
    with torch.no_grad():
        result = _ordered_capture(TorchCaptureOps(model, lora_path), views, load_adapter,
                                  teacher_gate, guard, progress)
    torch.cuda.synchronize()
    result["metadata"].update(seconds=time.monotonic()-guard.started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    guard()
    return result
