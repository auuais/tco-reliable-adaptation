"""Exercise a deterministic CPU fixture or verify paired GPU debug captures."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from tco_frozen_prefix_protocol import ROOT, PRIVATE, PUBLIC
from run_profiled_d0 import read, write_new


def fixture():
    import torch
    from torch import nn

    sys.path.insert(0, str(ROOT/"third_party/TCO/base_models/vggt"))
    from tco_frozen_prefix import SceneAggregator, TAPS
    from vggt.layers.rope import PositionGetter

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Module()
            a = self.aggregator
            a.aa_block_num, a.aa_block_size = 24, 1
            a.aa_order, a.patch_start_idx, a.patch_size = ["frame", "global"], 2, 1
            a.position_getter = PositionGetter()
            a.camera_token, a.register_token = torch.zeros(1, 2, 1, 3), torch.zeros(1, 2, 1, 3)
            a.frame_blocks = nn.ModuleList(nn.Linear(3, 3) for _ in range(24))
            a.global_blocks = nn.ModuleList(nn.Linear(3, 3) for _ in range(24))
            self.depth_head = nn.Module()
            self.depth_head.intermediate_layer_idx = TAPS

        def _process_frame_attention(self, tokens, b, s, p, c, i, pos, use_checkpoint):
            z = tokens.view(b*s, p, c)
            z = z + self.aggregator.frame_blocks[i](z) * .1
            return z, i+1, [z.view(b, s, p, c)]

        def _process_global_attention(self, tokens, b, s, p, c, i, pos, use_checkpoint):
            z = tokens.view(b, s*p, c)
            z = z + self.aggregator.global_blocks[i](z + z.mean(dim=1, keepdim=True)) * .1
            return z, i+1, [z.view(b, s, p, c)]

    torch.manual_seed(10)
    model = TinyModel().eval()
    for start in (12, 18):
        uncached = SceneAggregator(model, start)
        cached = SceneAggregator(model, start, reuse=True)
        x = torch.randn(3, 4, 3)
        params = tuple(model.aggregator.frame_blocks[start:].parameters())
        a, b = uncached(x, 1, 3, 2, 2)[0], cached(x, 1, 3, 2, 2)[0]
        c = cached(x, 1, 3, 2, 2)[0]
        for tap in TAPS:
            torch.testing.assert_close(a[tap], b[tap], rtol=0, atol=0)
            torch.testing.assert_close(a[tap], c[tap], rtol=0, atol=0)
        left = torch.autograd.grad(a[23].square().mean(), params)
        right = torch.autograd.grad(c[23].square().mean(), params)
        for old, new in zip(left, right):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        y = x.flip(0)
        expected, _ = uncached(y, 1, 3, 2, 2)
        changed, _ = cached(y, 1, 3, 2, 2)
        torch.testing.assert_close(expected[23], changed[23], rtol=0, atol=0)
        if cached.builds != 2 or cached.hits != 1 or cached.prefix_stages != 2*start:
            raise ValueError("Expected reuse followed by an input change")
    print("Both cache boundaries passed synthetic forward, gradient, and input invalidation checks")


def capture(job):
    hits = list((PRIVATE/"engineering"/job).glob("*/prediction.npz"))
    if len(hits) != 1:
        raise ValueError(f"Expected a single engineering capture for {job}")
    return np.load(hits[0], allow_pickle=False)


def differences(left, right):
    groups = {}
    with capture(left) as a, capture(right) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Compared engineering arms have different observables")
        for key in a.files:
            x, y = a[key].astype(np.float64), b[key].astype(np.float64)
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Invalid field {key}")
            name = key.split(".")[0]
            d = np.abs(x-y)
            if name == "sampler" and d.any():
                raise ValueError("The renderer's CUDA RNG states differ")
            if d.size:
                prev = groups.get(name, [0, 0])
                relative_l2 = float(np.linalg.norm(d.ravel()) / max(1.e-12, np.linalg.norm(x.ravel())))
                groups[name] = [max(prev[0], float(d.max())), max(prev[1], relative_l2)]
    return groups


def check(reference, repeat, candidate):
    variation = differences(reference, repeat)
    effects = differences(reference, candidate)
    floor = {key: [1.e-7, 1.e-6] for key in variation}
    tolerances = {key: [max(f, 1.25*d) for f, d in zip(floor[key], value)]
                  for key, value in variation.items()}
    for key in effects:
        if key == "sampler":
            continue
        if variation[key] == [0, 0] and (key.startswith("tap") or key.startswith("boundary")):
            tolerances[key] = [0, 0]
    valid = all(all(x <= t for x, t in zip(value, tolerances[key])) for key, value in effects.items())
    print(json.dumps(dict(reference=reference, repeat=repeat, candidate=candidate, tolerances=tolerances,
                          variation=variation, differences=effects, within_repeat_envelope=valid), indent=2))
    if not valid:
        raise SystemExit(2)


def qualify():
    pairs = (("zero_original", "zero_four"), ("orig_fixed", "four_fixed"),
             ("un12_fixed", "cache12_fixed"), ("un6_fixed", "cache6_fixed"))
    comparisons = {}
    for left, right in pairs:
        diff = differences(left, right)
        comparisons[right] = dict(reference=left, differences=diff,
                                  bitwise_equal=all(delta == [0, 0] for delta in diff.values()))
    real_loss = {}
    for reference, repeat, candidate in (("orig_a", "orig_b", "four_a"),
                                          ("un12_a", "un12_b", "cache12_a"),
                                          ("un6_a", "un6_b", "cache6_a")):
        real_loss[candidate] = dict(reference=reference, repeat=repeat,
                                    same_arm_variation=differences(reference, repeat),
                                    differences=differences(reference, candidate))
    cases = []
    for path in sorted((PRIVATE/"engineering").glob("*/*")):
        evidence = path/"complete.json"
        cases.append(read(evidence) if evidence.is_file() else dict(id=path.parent.name, complete=False))
    complete = all(c["bitwise_equal"] for c in comparisons.values())
    output = dict(deterministic_checks=comparisons, real_loss_checks=real_loss,
                  runs=cases, qualified=complete, rental_compute_usd=0)
    print(json.dumps(dict(qualified=complete, comparisons={key: item["bitwise_equal"]
                     for key, item in comparisons.items()}), indent=2))
    write_new(PUBLIC/"engineering.json", output)
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--qualify", action="store_true")
    cli.add_argument("jobs", nargs="*")
    args = cli.parse_args()
    if args.qualify:
        qualify()
    elif len(args.jobs) == 3:
        check(*args.jobs)
    elif len(args.jobs) == 2:
        print(json.dumps(differences(*args.jobs), indent=2))
    elif args.jobs:
        cli.error("Supply a reference and candidate, or also supply a same-seed repeat in between")
    else:
        fixture()
