"""Train and verify the two LayerNorms of the last VGGT global block."""
import hashlib

import torch


PREFIX = "aggregator.global_blocks.23."
NAMES = tuple(PREFIX + norm + "." + field for norm in ("norm1", "norm2") for field in ("weight", "bias"))


def selected_names(policy):
    if policy not in ("none", "both", "scale", "bias"):
        raise ValueError("Unknown LayerNorm policy: " + policy)
    fields = {"none": (), "both": ("weight", "bias"), "scale": ("weight",), "bias": ("bias",)}[policy]
    return [name for name in NAMES if name.rsplit(".", 1)[1] in fields]


def tensor_hash(value):
    array = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(array).hexdigest()


def resolve_norms(model):
    block = model.aggregator.global_blocks[23]
    members = dict(model.named_parameters())
    found = {}
    for name in NAMES:
        norm, field = name[len(PREFIX):].split(".")
        layer = getattr(block, norm)
        param = getattr(layer, field)
        if (not isinstance(layer, torch.nn.LayerNorm) or param is not members.get(name)
                or tuple(param.shape) != (1024,) or param.dtype != torch.float32):
            raise ValueError("Incorrect pretrained LayerNorm target: " + name)
        found[name] = param
    if len({p.untyped_storage().data_ptr() for p in found.values()}) != 4:
        raise ValueError("Aliased LayerNorm storage")
    return found


class ParameterAudit:
    """Install separate clipping while leaving the upstream LoRA clipping list intact."""

    def __init__(self, model, job, profile, initial):
        self.model = model
        self.job = job
        self.profile = profile
        self.detail = profile != "clean"
        self.fixed = profile in ("fixed", "fixed_bridge")
        self.native = profile in ("native_bridge", "fixed_bridge")
        self.targets = resolve_norms(model)
        self.selected = {k:self.targets[k] for k in selected_names(job["ln_policy"])}
        self.initial = initial
        self.frozen = [(name, param, param._version) for name, param in model.named_parameters()
                       if name not in self.selected]
        self.optimizer = None
        self.lora = {}
        self.steps = []
        self.preclip = {"lora": [], "ln": []}
        self.postclip = {"lora": [], "ln": []}
        self.displacements = []
        self.gradients = {}
        self.lora_start = None
        self.signed_samples = []
        self.sampler = []
        self.original_clip = torch.nn.utils.clip_grad_norm_
        if self.native and self.selected:
            raise ValueError("Native bridge must use only the original optimizer parameters")

    def install(self):
        setup = self.model.backbone_peft_setup
        self.model.backbone_peft_setup = lambda *a, **kw: self.configure(*setup(*a, **kw))
        if self.selected:
            # Upstream enables LoRA immediately after it freezes all base tensors.
            enable = self.model.enable_lora_gradients
            def enable_with_norms():
                result = enable()
                for p in self.selected.values():
                    p.requires_grad_(True)
                return result
            self.model.enable_lora_gradients = enable_with_norms

    def configure(self, optimizer, lora):
        self.optimizer = optimizer
        ids = {id(p) for p in lora}
        self.lora = {name:param for name, param in self.model.named_parameters() if id(param) in ids}
        if (len(self.lora) != 384 or any(not n.endswith((".lora_A", ".lora_B")) for n in self.lora)
                or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in lora]
                or len(optimizer.param_groups) != 1):
            raise ValueError("The baseline optimizer or clipping inventory changed")
        if self.detail:
            self.lora_start = self.parameter_digest()
        first = optimizer.param_groups[0]
        if (first["lr"], first["betas"], first["eps"], first["weight_decay"]) != (0.0003, (0.9, 0.999), 1e-8, 0):
            raise ValueError("Baseline Adam differs from registration")
        if self.selected:
            optimizer.add_param_group(dict(params=list(self.selected.values()), lr=self.job["ln_lr"], weight_decay=0.0))
            ln_group = optimizer.param_groups[-1]
            if any(ln_group[k] != first[k] for k in ("betas","eps","amsgrad","maximize","foreach","fused")):
                raise ValueError("LayerNorm group does not match the baseline Adam settings")
        optimizer.register_step_pre_hook(self.before_adam)
        optimizer.register_step_post_hook(self.after_adam)
        return optimizer, lora

    def parameter_digest(self):
        digest = hashlib.sha256()
        for name, param in self.lora.items():
            digest.update(name.encode())
            digest.update(bytes.fromhex(tensor_hash(param)))
        return digest.hexdigest()

    def clip(self, params, *args, **kwargs):
        if [id(p) for p in params] != [id(p) for p in self.optimizer.param_groups[0]["params"]]:
            raise ValueError("Upstream did not clip exactly the standard LoRA tensors")
        result = self.original_clip(params, *args, **kwargs)
        self.preclip["lora"].append(result.detach())
        return result

    def before_adam(self, optimizer, args, kwargs):
        for name, param in {**self.lora, **self.selected}.items():
            if not param.requires_grad or param.grad is None:
                raise ValueError("Missing actual trainable gradient: " + name)
        seen = {id(p) for p in self.model.parameters() if p.requires_grad}
        if seen != {id(p) for p in (*self.lora.values(), *self.selected.values())}:
            raise ValueError("Trainable tensor outside the allowed inventory")
        if self.selected:
            result = self.original_clip(list(self.selected.values()), max_norm=1.0)
            self.preclip["ln"].append(result.detach())
        if self.detail:
            for role, group in (("lora", self.lora), ("ln", self.selected)):
                if group:
                    norm = torch.stack([p.grad.detach().norm() for p in group.values()]).norm()
                    self.postclip[role].append(norm)
            sample = {}
            for name, p in self.selected.items():
                grad = p.grad.detach()
                sample[name] = dict(first=grad[:8].cpu().tolist(), last=grad[-8:].cpu().tolist(), norm=float(grad.norm()))
            self.signed_samples.append(sample)
        if self.fixed:
            for name, p in self.lora.items():
                self.gradients[f"{len(self.steps)+1}/{name}"] = p.grad.detach().cpu().numpy().copy()

    def after_adam(self, optimizer, args, kwargs):
        n = len(self.steps) + 1
        eligible = (*self.lora.values(), *self.selected.values())
        if len(optimizer.state) != len(eligible):
            raise ValueError("An optimizer state was missing or added")
        for param in eligible:
            step = optimizer.state[param]["step"]
            if step.device.type != "cpu" or step.item() != n:
                raise ValueError(f"Wrong Adam state counter at update {n}")
        self.steps.append(dict(update=n, lora_count=len(self.lora), ln_count=len(self.selected), counter_min=n, counter_max=n))
        if self.detail:
            self.displacements.append({name:float((p.detach().cpu() - self.initial[name]).norm())
                                       for name, p in self.selected.items()})
        print(f"Verified Adam update {n} with {len(eligible)} states", flush=True)

    def finish(self):
        mutations = [name for name, param, version in self.frozen if param._version != version]
        if mutations:
            raise ValueError("Unregistered pretrained tensor mutation: " + str(mutations[:5]))
        if len(self.steps) != self.job["steps"] or len(self.preclip["lora"]) != len(self.steps):
            raise ValueError("Missing actual Adam update or LoRA clipping call")
        start = {name:tensor_hash(p) for name, p in self.initial.items()}
        end = {name:tensor_hash(p) for name, p in self.targets.items()}
        for name in NAMES:
            changed = end[name] != start[name]
            if changed != (name in self.selected and self.job["ln_lr"] > 0 and len(self.steps) > 0):
                raise ValueError("Unexpected terminal LN weight change: " + name)
        named = {**self.lora, **self.selected}
        counts = {n:int(self.optimizer.state[p]["step"].item()) for n,p in named.items()} if self.optimizer else {}
        state_bytes = sum(x.numel()*x.element_size() for s in (self.optimizer.state.values() if self.optimizer else [])
                          for x in s.values() if isinstance(x, torch.Tensor) and x.is_cuda)
        def norms(data):
            return {role:torch.stack(values).cpu().tolist() if values else [] for role, values in data.items()}
        return dict(lora_names=list(self.lora), ln_names=list(self.selected), named_counts=counts, steps=self.steps,
                    lora_initial_sha256=self.lora_start, lora_final_sha256=self.parameter_digest() if self.detail else None,
                    ln_initial_sha256=start, ln_final_sha256=end, frozen_mutations=mutations,
                    frozen_version_count=len(self.frozen), separate_group_clipping=True,
                    preclip_norms=norms(self.preclip), postclip_norms=norms(self.postclip),
                    signed_ln_gradients=self.signed_samples, ln_delta_norms=self.displacements, cuda_state_bytes=state_bytes)
