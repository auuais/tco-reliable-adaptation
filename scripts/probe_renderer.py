"""Bounded real CUDA 2DGS forward/backward probe before any model experiment."""
import importlib.metadata
import argparse
import json
import pathlib
import platform
import time

import torch
from gsplat import rasterization_2dgs

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "environment" / "renderer_probe.json")
parser.add_argument("--require-precompiled", action="store_true")
args = parser.parse_args()
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
if args.require_precompiled:
    from gsplat import csrc
    assert pathlib.Path(csrc.__file__).is_file()
torch.manual_seed(42)
torch.cuda.reset_peak_memory_stats()
means = torch.tensor([[-0.15, 0.0, 2.0], [0.15, 0.0, 2.2]], device="cuda", requires_grad=True)
quats = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda", requires_grad=True)
scales = torch.full((2, 3), 0.12, device="cuda", requires_grad=True)
opacities = torch.full((2,), 0.8, device="cuda", requires_grad=True)
colors = torch.tensor([[1., .2, .1], [.1, .2, 1.]], device="cuda", requires_grad=True)
viewmats = torch.eye(4, device="cuda")[None]
intrinsics = torch.tensor([[[60., 0., 32.], [0., 60., 32.], [0., 0., 1.]]], device="cuda")
started = time.perf_counter()
rendered = rasterization_2dgs(means, quats, scales, opacities, colors[None], viewmats,
                             intrinsics, 64, 64, render_mode="RGB", packed=False)
rgb, alpha = rendered[:2]
loss = rgb.square().mean() + alpha.mean()
loss.backward()
torch.cuda.synchronize()
grads = {name: {"finite": bool(torch.isfinite(tensor.grad).all()),
                "norm": float(tensor.grad.norm())}
         for name, tensor in {"means": means, "quats": quats, "scales": scales,
                               "opacities": opacities, "colors": colors}.items()}
assert torch.isfinite(rgb).all() and torch.isfinite(loss)
assert all(g["finite"] for g in grads.values())
assert grads["means"]["norm"] > 0 and grads["colors"]["norm"] > 0
record = {"status": "passed", "python": platform.python_version(), "torch": torch.__version__,
          "cuda": torch.version.cuda, "gsplat": importlib.metadata.version("gsplat"),
          "gpu": torch.cuda.get_device_name(), "loss": float(loss), "gradients": grads,
          "compile_and_probe_seconds": time.perf_counter() - started,
          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
          "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
record["required_precompiled_extension"] = args.require_precompiled
if args.require_precompiled:
    record["precompiled_extension"] = csrc.__file__
out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
