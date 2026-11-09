"""Lock only the distributions added/changed from the pinned PyTorch base."""
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1]
normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
base = {normalize(p["name"]): p["version"] for p in json.loads((root / "environment/base-pip-list.json").read_text(encoding="utf-8-sig"))}
local = {normalize(p["name"]): p["version"] for p in json.loads((root / "environment/local-pip-list.json").read_text(encoding="utf-8-sig"))}
delta = {name: version for name, version in local.items() if base.get(name) != version}
assert delta["gsplat"] == "1.5.3"
assert not any(name in delta for name in ["torch", "torchvision", "triton"])
lines = [f"{name}=={version}" for name, version in sorted(delta.items()) if name != "gsplat"]
(root / "environment/requirements-cloud-delta.txt").write_text("\n".join(lines) + "\n")
record = {"base_distributions": base, "expected_distributions": local, "delta": delta,
          "cuda_wheel": {"name": "gsplat", "version": "1.5.3", "architectures": ["sm_86", "sm_89"]}}
(root / "environment/cloud-dependency-lock.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"Locked {len(delta)} changed/added distributions; Torch/CUDA stay in the pinned base.")
