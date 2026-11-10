"""Observation wrapper for the local full-resolution pose-precision comparison.

Compose frozen v6 capture, local offloading and the tested pose-energy wrapper.
No historical runner is edited; both precision modes use this same instrumented path.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import run_pose_precision as precision
import run_repeat_calibration as base
from repeatability_capture import array_digest


def main():
    precision.memory.base = base
    original_dump = base.dump
    wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def dump(path, content):
        if Path(path).name == "run.json":
            content["precision_study_wrapper_sha256"] = wrapper_hash
        return original_dump(path, content)
    base.dump = dump
    klass = base.tco_module.TCO_VGGT_LoRA
    original_factory = klass.from_pretrained
    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        original_setup = model.backbone_peft_setup
        def setup(*positional, **keyword):
            optimizer, params = original_setup(*positional, **keyword)
            captured = False
            def first_gradient(opt, positional_args, keyword_args):
                nonlocal captured
                if captured:
                    return
                import sys
                output = base.ROOT / "runs" / sys.argv[sys.argv.index("--run-id") + 1]
                values = {name: p.grad.detach().float().cpu().numpy()
                          for name, p in model.named_parameters()
                          if ("lora_A" in name or "lora_B" in name) and p.grad is not None}
                assert values and all(np.isfinite(v).all() for v in values.values())
                path = output / "first_clipped_gradients.npz"
                assert not path.exists()
                np.savez_compressed(path, **values)
                (output / "first_clipped_gradients.json").write_text(json.dumps(dict(
                    tensor_sha256=array_digest(values), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    note="After upstream global clipping; observation only; included in instrumented runtime"), indent=2) + "\n")
                captured = True
            optimizer.register_step_pre_hook(first_gradient)
            return optimizer, params
        model.backbone_peft_setup = setup
        return model
    klass.from_pretrained = staticmethod(factory)
    precision.main()


if __name__ == "__main__":
    main()
