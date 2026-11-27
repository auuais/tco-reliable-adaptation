"""CPU qualification for signed derivatives and the final global block update."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from tco_final_ln_policy import NAMES, ParameterAudit, resolve_norms
from tco_final_ln_raster import cpu_vjp


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(2219)
    model = nn.Module()
    model.aggregator = nn.Module()
    model.aggregator.global_blocks = nn.ModuleList([nn.Identity() for _ in range(23)] + [nn.Module()])
    block = model.aggregator.global_blocks[-1]
    block.norm1, block.norm2 = nn.LayerNorm(1024), nn.LayerNorm(1024)
    block.marker = nn.Parameter(torch.tensor(0.11), requires_grad=False)
    targets = resolve_norms(model)
    inputs = torch.randn((3,1024))
    upstream = torch.randn_like(inputs)
    def readout():
        return torch.sin(block.norm2(0.5*inputs + torch.tanh(block.norm1(inputs))))
    def scalar():
        return (readout()*upstream).sum()/math.sqrt(inputs.numel())
    grads = torch.autograd.grad(scalar(), tuple(targets.values()))
    directions = {}
    errors = {}
    for (name, p), grad in zip(targets.items(), grads):
        # Use directions with substantial positive AND negative derivatives at the known input.
        direction = torch.sign(grad) / math.sqrt(p.numel())
        sign = -1 if name.endswith("bias") else 1
        direction *= sign
        analytical = float(torch.dot(grad, direction))
        measurements = []
        start = p.detach().clone()
        with torch.no_grad():
            for eps in (.05, .025, .0125):
                p.copy_(start+eps*direction)
                pos = float(scalar())
                p.copy_(start-eps*direction)
                neg = float(scalar())
                measurements.append((pos-neg)/(2*eps))
            p.copy_(start)
        relative = [abs(x-analytical)/abs(analytical) for x in measurements]
        if (abs(analytical) <= .01 or max(relative) > .003
                or any(math.copysign(1,x) != sign for x in measurements)):
            raise AssertionError((name, analytical, measurements))
        if any(abs(x+analytical) <= .003*abs(analytical) for x in measurements):
            raise AssertionError("Sign-reversal negative control was accepted")
        directions[name] = dict(autograd=analytical, central_differences=measurements, sign_flip_rejected=True)
        errors[name] = max(relative)
    before = readout().detach().clone()
    p = next(iter(targets.values()))
    with torch.no_grad():
        p[4].add_(0.1)
        change = float((readout()-before).abs().max())
        p[4].sub_(0.1)
    if change <= .001:
        raise AssertionError("One-channel perturbation failed to affect the readout")

    # Exercise the actual policy's setup, clip path, named counters and mutation validator.
    # Scalar dummy adapters keep the exact 192-by-two membership contract inexpensive.
    def apply(lr):
        import copy
        base = copy.deepcopy(model)
        initial = {k:t.detach().clone() for k,t in resolve_norms(base).items()}
        def setup():
            base.adapters = nn.ModuleList(nn.Module() for _ in range(192))
            for mod in base.adapters:
                mod.lora_A = nn.Parameter(torch.tensor([.24, -.31]))
                mod.lora_B = nn.Parameter(torch.tensor([.49, .13]))
            std = list(base.adapters.parameters())
            return torch.optim.Adam(std, lr=.0003), std
        base.backbone_peft_setup = setup
        base.enable_lora_gradients = lambda: [p.requires_grad_(True) for p in base.adapters.parameters()]
        audit = ParameterAudit(base, dict(ln_policy="both", ln_lr=lr, steps=1), "diagnostic", initial)
        audit.install()
        opt, adapters = base.backbone_peft_setup()
        for name,p in base.named_parameters():
            if name in NAMES:
                p.requires_grad_(False)
        base.enable_lora_gradients()
        for g, p in zip(grads, resolve_norms(base).values()):
            p.grad = g.clone()
        for i,p in enumerate(adapters):
            p.grad = torch.tensor([1.,-.5])*(1+i*.001)
        audit.clip(adapters, max_norm=1.0)
        lora_before = [p.detach().clone() for p in adapters]
        opt.step()
        result = audit.finish()
        if (len(result["named_counts"]) != 388 or set(result["named_counts"].values()) != {1}
                or any(torch.equal(old,p) for old,p in zip(lora_before,adapters))):
            raise AssertionError("Optimizer membership/counters/actual movement failed")
        for name, p in resolve_norms(base).items():
            # Independent bias-corrected Adam step, applied on the clipped recorded gradient.
            g = p.grad.detach().double()
            first, second = .1*g, .001*g.square()
            expected = initial[name].double() - lr*(first/.1)/((second/.001).sqrt()+1.e-8)
            torch.testing.assert_close(p.detach().double(), expected, rtol=1.e-7, atol=2.e-8)
        return [p.detach().clone() for p in adapters], result
    no_ln, zero = apply(0.0)
    with_ln, active = apply(3.e-5)
    if any(not torch.equal(a,b) for a,b in zip(no_ln,with_ln)):
        raise AssertionError("LoRA update was affected by changing only the LN learning rate on identical gradients")
    return dict(passed=True, dtype="float32", max_relative_derivative_error=max(errors.values()),
                directions=directions, lr_zero_layernorm_unchanged=zero["ln_initial_sha256"]==zero["ln_final_sha256"],
                lr_positive_all_changed=all(active["ln_initial_sha256"][n]!=active["ln_final_sha256"][n] for n in NAMES),
                adam_independent_reference=True, zero_lr_keeps_lora_update=True, one_channel_readout_max_change=change,
                optimizer_states=388, rgb_ed_cpu=cpu_vjp())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--receipt", type=Path)
    args = cli.parse_args()
    data = json.dumps(fixture(), allow_nan=False, indent=2)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as stream:
            stream.write(data + "\n")
    print(data)
