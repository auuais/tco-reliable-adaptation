"""Retain used VGGT outputs and reuse the frozen portion of a scene forward."""
from contextlib import nullcontext
import hashlib
import math

import torch


TAPS = (4, 11, 17, 23)


class SceneAggregator:
    def __init__(self, model, start=0, reuse=False, stream=False):
        self.model = model
        self.aggregator = model.aggregator
        self.start = start
        self.reuse = reuse
        self.stream = stream
        self.cache = {}
        self.source_tokens = self.identity = None
        self.builds = self.hits = self.prefix_stages = 0
        self.cached_bytes = 0
        self.boundary_observer = None
        a = self.aggregator
        if (a.aa_block_num != 24 or a.aa_block_size != 1 or list(a.aa_order) != ["frame", "global"]
                or tuple(model.depth_head.intermediate_layer_idx) != TAPS or start not in (0, 12, 18)):
            raise ValueError("Unsupported VGGT stage or head configuration")

    def __call__(self, patch_tokens, B, S, H, W, register_token_override=None, use_checkpoint=False):
        if register_token_override is not None:
            raise ValueError("Expected checkpoint register tokens")
        a = self.aggregator
        device = patch_tokens.device
        kind = device.type
        context = (torch.is_autocast_enabled(kind), torch.get_autocast_dtype(kind))
        identity = (patch_tokens._version, B, S, H, W, self.model.training)
        if patch_tokens is not self.source_tokens or identity != self.identity:
            self.cache.clear()
            self.source_tokens = patch_tokens
            self.identity = identity
        stored = self.cache.get(context)
        outputs = [None] * 24
        if stored is None:
            from vggt.models.aggregator import slice_expand_and_flatten

            camera = slice_expand_and_flatten(a.camera_token, B, S)
            registers = slice_expand_and_flatten(a.register_token, B, S)
            tokens = torch.cat([camera, registers, patch_tokens], dim=1)
            pos = a.position_getter(B*S, H//a.patch_size, W//a.patch_size, device=device) + 1
            pos = torch.cat([pos.new_zeros(B*S, a.patch_start_idx, 2), pos], dim=1)
            for index in range(self.start):
                tokens = self.stage(index, tokens, pos, outputs, B, S, False)
            if self.reuse:
                self.builds += 1
                frozen = {idx: outputs[idx].detach() for idx in TAPS if idx < self.start}
                self.cache[context] = (tokens.detach(), pos, frozen)
                self.cached_bytes = sum(t.numel()*t.element_size() for z, p, taps in self.cache.values()
                                        for t in (z, p, *taps.values()))
        else:
            self.hits += 1
            tokens, pos, taps = stored
            for index, tensor in taps.items():
                outputs[index] = tensor
        if self.start and self.boundary_observer is not None:
            self.boundary_observer(tokens)
        for index in range(self.start, 24):
            tokens = self.stage(index, tokens, pos, outputs, B, S, use_checkpoint)
        return outputs, a.patch_start_idx

    def stage(self, index, tokens, pos, outputs, batch, views, checkpoint):
        frozen = index < self.start
        modules = (self.aggregator.frame_blocks[index], self.aggregator.global_blocks[index])
        if frozen:
            self.prefix_stages += 1
            if self.stream:
                for module in modules:
                    module.to(tokens.device)
        channels = tokens.size(-1)
        patches = tokens.numel() // (batch * views * channels)
        with torch.no_grad() if frozen else nullcontext():
            tokens, _, frame = self.model._process_frame_attention(
                tokens, batch, views, patches, channels, index, pos, use_checkpoint=checkpoint)
            tokens, _, global_view = self.model._process_global_attention(
                tokens, batch, views, patches, channels, index, pos, use_checkpoint=checkpoint)
            if index in TAPS:
                outputs[index] = torch.cat([frame[0], global_view[0]], dim=-1)
        if frozen and self.stream:
            for module in modules:
                module.to("cpu")
            torch.clear_autocast_cache()
        return tokens


def install(model, arm, seed):
    """Bind the variant and make common adapter initializations independent of inventory."""
    start = {"zero": 0, "original": 0, "four": 0, "cached12": 12, "cached6": 18,
             "uncached12": 12, "uncached6": 18}[arm]
    original_targets = model.get_lora_target_modules
    original_setup = model.backbone_peft_setup
    original_extract = model.extract_patch_tokens
    original_forward = model.forward_original_vggt
    a = model.aggregator
    path = None
    model.point_head = None

    def extract(images):
        output = original_extract(images)
        a.patch_embed.to("cpu")
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        return output

    model.extract_patch_tokens = extract

    def initial_forward(images):
        result = original_forward(images)
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()
        return result

    model.forward_original_vggt = initial_forward
    if arm not in ("zero", "original"):
        path = SceneAggregator(model, start, reuse=arm.startswith("cached"), stream=bool(start))
        model.aggregate_with_patch_tokens = path
    if start:
        def targets():
            return [*a.frame_blocks[start:], *a.global_blocks[start:]]
        model.get_lora_target_modules = targets

    def setup(*args, **kwargs):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            optimizer, params = original_setup(*args, **kwargs)
        initial_names = []
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.endswith(".lora_A"):
                    continue
                digest = hashlib.sha256(f"{seed}/{name}".encode()).digest()
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(int.from_bytes(digest[:8], "little"))
                limit = 1/math.sqrt(parameter.shape[-1])
                parameter.uniform_(-limit, limit, generator=generator)
                initial_names.append(name)
        expected_blocks = original_targets() if not start else model.get_lora_target_modules()
        if len(initial_names) != 4*len(expected_blocks) or len(params) != 2*len(initial_names):
            raise ValueError("Unexpected adapter inventory")
        return optimizer, params

    model.backbone_peft_setup = setup
    return path


def place(model, path):
    """Keep the cached or recomputed prefix on the host between calls."""
    if path is None or not path.start:
        model.cuda()
        return
    a = model.aggregator
    for index in range(path.start, 24):
        a.frame_blocks[index].cuda()
        a.global_blocks[index].cuda()
    for module in (a.patch_embed, model.depth_head, model.camera_head, a.rope):
        module.cuda()
    for name in ("camera_token", "register_token"):
        parameter = getattr(a, name)
        setattr(a, name, torch.nn.Parameter(parameter.detach().cuda(), requires_grad=parameter.requires_grad))
    a._resnet_mean = a._resnet_mean.cuda()
    a._resnet_std = a._resnet_std.cuda()
