"""Local engineering probe: remove unused head and offload the cached image encoder.

Does not edit upstream files or historical runners. No numerical-equivalence
claim follows from a successful run; compare with a matched control separately.
"""
import hashlib
import pathlib

import run_experiment as base
import torch


def main():
    info = dict(variant="inactive_modules_cpu_v1", engineering_probe=True,
                wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                changes=["remove unused point_head before GPU transfer", "offload patch_embed after token caching", "clear autocast weight cache after offload"])
    original_dump = base.dump
    def dump(path, content):
        if pathlib.Path(path).name == "run.json":
            content["memory_optimization"] = dict(info)
        return original_dump(path, content)
    base.dump = dump
    klass = base.tco_module.TCO_VGGT_LoRA
    original_factory = klass.from_pretrained
    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        info["unused_point_head_parameters"] = sum(p.numel() for p in model.point_head.parameters())
        model.point_head = None
        original_forward = model.forward_original_vggt
        def cached_forward(images):
            output = original_forward(images)
            info["offloaded_patch_embed_parameters"] = sum(p.numel() for p in model.aggregator.patch_embed.parameters())
            model.aggregator.patch_embed.to("cpu")
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            return output
        model.forward_original_vggt = cached_forward
        return model
    klass.from_pretrained = staticmethod(factory)
    base.main()


if __name__ == "__main__":
    main()
