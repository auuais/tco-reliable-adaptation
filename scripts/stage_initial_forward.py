"""Run upstream initial VGGT inference with one group of large modules resident at a time."""
import torch


def stage_initial_forward(model, observe):
    original = model.forward_original_vggt

    def run(images):
        device = images.device
        if device.type != "cuda":
            raise ValueError("This memory experiment requires CUDA")
        blocks = (model.aggregator.frame_blocks, model.aggregator.global_blocks)
        heads = (model.camera_head, model.depth_head)
        extract = model.extract_patch_tokens
        aggregate = model.aggregate_with_patch_tokens

        def move(modules, destination):
            for module in modules:
                module.to(destination)
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()

        def boundary(name):
            torch.cuda.synchronize(device)
            observe(name, {"allocated_bytes": torch.cuda.memory_allocated(device),
                           "reserved_bytes": torch.cuda.memory_reserved(device)})

        def extract_then_load_blocks(*args, **kwargs):
            tokens = extract(*args, **kwargs)
            move((model.aggregator.patch_embed,), "cpu")
            move(blocks, device)
            boundary("patch_tokens_ready")
            return tokens

        def aggregate_then_load_heads(*args, **kwargs):
            features = aggregate(*args, **kwargs)
            move(blocks, "cpu")
            move(heads, device)
            boundary("head_features_ready")
            return features

        with torch.no_grad():
            move((*blocks, *heads), "cpu")
            boundary("image_encoder_ready")
            model.extract_patch_tokens = extract_then_load_blocks
            model.aggregate_with_patch_tokens = aggregate_then_load_heads
            try:
                outputs = original(images)
                boundary("initial_heads_ready")
                # LoRA initialization must see the same devices as the upstream model.
                move(blocks, device)
                boundary("ready_for_adapters")
                return outputs
            finally:
                model.extract_patch_tokens = extract
                model.aggregate_with_patch_tokens = aggregate

    model.forward_original_vggt = run
