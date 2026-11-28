"""A real TCO update schedule with explicit prefix generations and one Adam instance."""
from collections import Counter
import hashlib
import re
import time


import torch

from tco_frozen_prefix import SceneAggregator, install as install_baseline


SUFFIX = 12
HYBRIDS = {"Refresh1", "Refresh1Uncached", "Refresh2", "Narrow", "NarrowUncached"}
INITIALIZED = {"Distilled", "OfflineOriginal"}


def stage_number(name):
    match = re.match(r"aggregator\.(?:frame_blocks|global_blocks)\.(\d+)\.", name)
    if match is None:
        raise ValueError("Unexpected adapter target " + name)
    return int(match.group(1))


def tensor_digest(t):
    array = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def move_frozen_base(module, device):
    """Stream the base while keeping the adapters and their Adam moments on the GPU."""
    adapters = []
    try:
        for leaf in module.modules():
            for name in ("lora_A", "lora_B"):
                if name in leaf._parameters:
                    adapters.append((leaf, name, leaf._parameters.pop(name)))
        module.to(device)
    finally:
        for leaf, name, value in adapters:
            leaf.register_parameter(name, value)


class RefreshPath(SceneAggregator):
    def __init__(self, model, arm):
        super().__init__(model, start=0)
        self.arm = arm
        self.early_start = 11 if arm.startswith("Narrow") else 0
        self.refresh_steps = (1, 6) if arm == "Refresh2" else (1,)
        self.start = self.early_start
        self.generation = 0
        self.calls_by_generation = Counter()
        self.last_shape = None



    def invalidate(self):
        self.generation += 1
        self.cache.clear()
        self.cached_bytes = 0

    def configure(self, next_step):
        self.start = self.early_start if next_step in self.refresh_steps else SUFFIX
        if self.start < SUFFIX:
            # The broader backward has to execute stages before the old suffix.
            # Its next update will advance the weight generation separately.
            self.cache.clear()
            self.cached_bytes = 0
        self.reuse = self.start == SUFFIX and not self.arm.endswith("Uncached")

    def __call__(self, patch_tokens, B, S, H, W, register_token_override=None, use_checkpoint=False):
        self.last_shape = (B, S, H, W)
        self.calls_by_generation[self.generation] += 1
        return super().__call__(patch_tokens, B, S, H, W, register_token_override, use_checkpoint)

    def stage(self, index, tokens, pos, outputs, batch, views, checkpoint):
        blocks = (self.aggregator.frame_blocks[index], self.aggregator.global_blocks[index])
        # The narrow arm always streams pairs 0..10. Pairs participating in the
        # next backward must remain resident until checkpoint recomputation ends.
        if index < SUFFIX:
            for block in blocks:
                if next(block.parameters()).device != tokens.device:
                    move_frozen_base(block, tokens.device)
        result = super().stage(index, tokens, pos, outputs, batch, views, checkpoint)
        if index < self.start:
            for block in blocks:
                move_frozen_base(block, "cpu")
            torch.clear_autocast_cache()
        return result

    @torch.no_grad()
    def stale_control(self, adapter):
        """Hold a real cache constant under an intentional upstream LoRA perturbation."""
        old = adapter.detach().clone()
        token = self.source_tokens
        with torch.autocast("cuda", dtype=torch.bfloat16):
            def get_tap():
                return self(token, *self.last_shape)[0][11].detach().cpu().clone()
            self.invalidate()
            before = get_tap()
            try:
                adapter.add_(0.125)
                stale = get_tap()
                self.invalidate()
                torch.clear_autocast_cache()
                rebuilt = get_tap()
                self.cache.clear()
                fresh = get_tap()
            finally:
                adapter.copy_(old)
                self.invalidate()
        evidence = dict(stale_unchanged=torch.equal(before, stale), rebuilt_matches_fresh=torch.equal(rebuilt, fresh),
                        changed_elements=int((before != rebuilt).sum()), compared_elements=before.numel(),
                        max_change=float((before.float() - rebuilt.float()).abs().max()))
        if not evidence["stale_unchanged"] or not evidence["rebuilt_matches_fresh"] or not evidence["changed_elements"]:
            raise ValueError("Stale-cache control failed: " + str(evidence))
        return evidence


class NarrowPath(RefreshPath):
    """Cache immutable pairs 0..10 separately from the mutable pair-11 prefix."""
    def __init__(self, model, arm):
        super().__init__(model, arm)
        self.stable = {}
        self.stable_builds = self.stable_hits = 0
        self.stable_key = None

    def __call__(self, patch_tokens, B, S, H, W, register_token_override=None, use_checkpoint=False):
        signature = (patch_tokens._version, B, S, H, W, self.model.training)
        if self.source_tokens is not patch_tokens or self.identity != signature:
            self.stable.clear()
        device = patch_tokens.device.type
        self.stable_key = (torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
        return super().__call__(patch_tokens, B, S, H, W, register_token_override, use_checkpoint)

    def stage(self, index, tokens, pos, outputs, batch, views, checkpoint):
        cache_first_boundary = index < 11 and not self.arm.endswith("Uncached")
        stored = self.stable.get(self.stable_key) if cache_first_boundary else None
        if stored is not None:
            if index == 4:
                outputs[4] = stored[1]
            if index == 10:
                self.stable_hits += 1
                return stored[0]
            return tokens
        result = super().stage(index, tokens, pos, outputs, batch, views, checkpoint)
        if index == 10 and cache_first_boundary:
            self.stable[self.stable_key] = (result.detach(),outputs[4].detach())
            self.stable_builds += 1
        return result

    def stable_bytes(self):
        return sum(t.numel()*t.element_size() for pair in self.stable.values() for t in pair)


class HybridAdamAudit:
    def __init__(self, model, arm, seed, profile, initialization=None):
        self.model, self.arm, self.profile = model, arm, profile
        self.initialization = initialization
        self.initialization_seconds = None
        self.updates = 0
        self.signed = {}
        self.initial = {}
        self.steps = []
        self.factors = {}
        self.optimizer = None
        if arm.startswith("Narrow"):
            agg = model.aggregator
            model.get_lora_target_modules = lambda: [*agg.frame_blocks[11:], *agg.global_blocks[11:]]
        native = ("zero" if arm == "ZERO" else "cached12" if arm == "Cache10" or arm in INITIALIZED else "four")
        self.path = install_baseline(model, native, seed)
        if arm in HYBRIDS:
            self.path = NarrowPath(model, arm) if arm.startswith("Narrow") else RefreshPath(model, arm)
            model.aggregate_with_patch_tokens = self.path
        original = model.backbone_peft_setup

        def setup(*args, **kwargs):
            opt, params = original(*args, **kwargs)
            self.optimizer = opt
            self.factors = {n:p for n,p in model.named_parameters() if n.endswith((".lora_A",".lora_B"))}
            if set(map(id, params)) != set(map(id, self.factors.values())) or len(params) != len(self.factors):
                raise ValueError("Optimizer factors differ from model identity")
            expected = 208 if arm.startswith("Narrow") else 192 if native == "cached12" else 384
            if len(self.factors) != expected:
                raise ValueError("Incorrect LoRA tensor count: " + str(len(self.factors)))
            if initialization is not None:
                from safetensors.torch import load_file
                torch.cuda.synchronize()
                init_started = time.perf_counter()
                source = load_file(str(initialization))
                if set(source) != set(self.factors):
                    raise ValueError("Teacher initialization factor names differ")
                with torch.no_grad():
                    for n,p in self.factors.items():
                        if tuple(source[n].shape) != tuple(p.shape) or source[n].dtype != p.dtype:
                            raise ValueError("Invalid seed factor " + n)
                        p.copy_(source[n])
                torch.cuda.synchronize()
                self.initialization_seconds = time.perf_counter() - init_started
            if profile in ("fixed", "stale"):
                self.initial = {n:tensor_digest(p) for n,p in self.factors.items()}
                for n,p in self.factors.items():
                    def collect(g, name=n):
                        self.signed[f"grad/{self.updates+1}/{name}"] = g.detach().float().cpu().numpy().copy()
                    p.register_hook(collect)
            opt.register_step_pre_hook(self.before)
            opt.register_step_post_hook(self.after)
            return opt, params

        model.backbone_peft_setup = setup

    def is_live(self, name, step):
        if self.arm not in HYBRIDS:
            return True
        return stage_number(name) >= SUFFIX or step in self.path.refresh_steps

    def before(self, opt, _args, _kw):
        self.sleeping = {}
        for name,p in self.factors.items():
            active = self.is_live(name, self.updates + 1)
            if active != (p.grad is not None):
                raise ValueError(f"Gradient scope mismatch on update {self.updates+1}: {name}")
            if not active:
                self.sleeping[name] = (p._version, tuple((k,id(t),t._version) for k,t in opt.state[p].items()))
        self.steps.append(len(self.factors)-len(self.sleeping))

    def after(self, opt, _args, _kw):
        self.updates += 1
        if self.arm not in HYBRIDS:
            return
        for n,previous in self.sleeping.items():
            p = self.factors[n]
            if previous != (p._version, tuple((k,id(t),t._version) for k,t in opt.state[p].items())):
                raise ValueError("Inactive Adam state or factor changed: " + n)
        if self.updates in self.path.refresh_steps:
            self.path.invalidate()
        self.path.configure(self.updates + 1)
        for n,p in self.factors.items():
            if stage_number(n) < SUFFIX:
                p.requires_grad_(self.is_live(n,self.updates+1))
                if not p.requires_grad:
                    p.grad = None

    def finish(self):
        expected = dict(self.model.named_parameters())
        counts = Counter()
        for n,p in self.factors.items():
            state = self.optimizer.state[p]
            planned = sum(self.is_live(n, step) for step in range(1, self.updates+1))
            if expected[n] is not p or int(state["step"].item()) != planned:
                raise ValueError("Adapter identity/Adam count mismatch: " + n)
            counts[("prefix" if stage_number(n)<SUFFIX else "suffix",planned)] += 1
            if self.profile in ("fixed","stale"):
                self.signed["final/"+n] = p.detach().float().cpu().numpy().copy()
        return {f"{region}_{step}":num for (region,step),num in sorted(counts.items())}

