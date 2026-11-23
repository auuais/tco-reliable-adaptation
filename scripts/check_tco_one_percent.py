"""Qualify the geometric correction and the modified execution paths before scoring."""

import argparse

from run_profiled_d0 import sha, write_new
from tco_one_percent_protocol import MANIFEST, PRIVATE, PUBLIC, collect, verify


def fixture():
    import torch
    from torch.nn import functional as F

    from check_tco_frozen_prefix import fixture as check_prior_cache
    from tco_one_percent_geometry import BoundaryCorrection, correspondences, sample_depth, scene_scale, shuffle_edges

    check_prior_cache()
    prior = torch.eye(4).repeat(3, 1, 1)
    prior[:, 0, 3] = torch.tensor([0., -2., -4.])
    predicted = prior.clone()
    predicted[:, :3, 3] /= 2
    intrinsics = torch.tensor([[4., 0., 4.5], [0., 4., 2.5], [0., 0., 1.]]).repeat(3, 1, 1)
    depth = torch.full((3, 6, 10), 2.)
    edges, record = correspondences(depth, prior, predicted, intrinsics, 2)
    assert record["scale"] == 2 and record["pairs"] == 3 and record["fallback"] is None
    assert edges.shape == (3, 15, 3)
    assert int(edges[0, 0, 1]) == -1 and int(edges[0, 1, 1]) == 15
    assert int(edges[1, 0, 0]) == 1 and int(edges[0, 2, 2]) == 30
    assert (edges[torch.arange(3), :, torch.arange(3)] == -1).all()
    occluded = depth.clone()
    occluded[1] *= 0.75
    bad, _ = correspondences(occluded, prior, predicted, intrinsics, 2)
    assert (bad[0, :, 1] == -1).all()
    identity = prior.clone()
    identity[1] = identity[0]
    identity_edges, _ = correspondences(depth, identity, identity, intrinsics, 2)
    torch.testing.assert_close(identity_edges[0, :, 1], 15 + torch.arange(15))
    stateless = torch.eye(4).repeat(3, 1, 1)
    empty, info = correspondences(depth, stateless, stateless, intrinsics, 2)
    assert (empty == -1).all() and info == dict(pairs=0, scale=None, scale_range=None,
                                                fallback="undefined_scale", patches_per_view=15,
                                                supported_patches=[0, 0, 0], partner_histogram=[45, 0, 0])
    assert scene_scale(prior[:2], predicted[:2])[0] is None
    _, info = correspondences(torch.zeros_like(depth), prior, predicted, intrinsics, 2)
    assert info["fallback"] == "empty_graph"
    with_invalid = depth[0].clone()
    with_invalid[1, 1] = float("nan")
    samples, ok = sample_depth(with_invalid, torch.tensor([[0.5, 0.5], [0., 0.], [-0.5, 1.], [9., 5.]]))
    assert ok.tolist() == [False, True, False, True] and samples[1] == samples[3] == 2

    order = torch.tensor([2, 0, 1])
    remapped, _ = correspondences(depth[order], prior[order], predicted[order], intrinsics[order], 2)
    expected = edges[order][:, :, order]
    expected = torch.where(expected >= 0, order.argsort()[expected.clamp_min(0) // 15] * 15 + expected % 15, -1)
    torch.testing.assert_close(remapped, expected)
    shuffled, shuffle_record = shuffle_edges(edges, seed=1729)
    assert torch.equal(shuffled >= 0, edges >= 0) and shuffle_record["changed_edges"] > 0
    torch.testing.assert_close(shuffled.sort(dim=1).values, edges.sort(dim=1).values)
    torch.testing.assert_close(shuffled, shuffle_edges(edges, 1729)[0])

    for mode, table in (("geometry", edges), ("shuffle", shuffled), ("own", edges)):
        transformer = BoundaryCorrection(7, table, mode, 99)
        generator = torch.Generator().manual_seed(330)
        tokens = torch.randn((1, 3 * 17, 7), generator=generator)
        assert torch.equal(transformer(tokens, 2), tokens)
        first_a, first_b = torch.autograd.grad(transformer(tokens, 2).square().sum(), transformer.parameters())
        assert not first_a.any() and first_b.any()
        with torch.no_grad():
            transformer.lora_B.copy_(torch.randn((7, 4), generator=generator))
        output = transformer(tokens, 2)
        grid = tokens.view(3, 17, 7)
        features = grid[:, 2:]
        valid = table >= 0
        count = valid.sum(-1)[..., None]
        borrowed = features.reshape(-1, 7)[table.clamp_min(0)]
        means = borrowed.masked_fill(~valid[..., None], 0).sum(-2) / count.clamp_min(1)
        values = features if mode == "own" else means - features
        residual = F.linear(F.linear(values, transformer.lora_A), transformer.lora_B).masked_fill(count == 0, 0)
        dense = torch.cat((grid[:, :2], features + residual), dim=1).view_as(tokens)
        torch.testing.assert_close(output, dense)
        fast_grad = torch.autograd.grad(output.square().sum(), transformer.parameters())
        dense_grad = torch.autograd.grad(dense.square().sum(), transformer.parameters())
        for left, right in zip(fast_grad, dense_grad):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=2e-5)
        assert torch.equal(output.view_as(grid)[:, :2], grid[:, :2])
        assert fast_grad[0].any() and fast_grad[1].any()
    print("Validated scale, reprojection, visibility, shuffle, input order, identity, and low-rank gradients")


def prediction_difference(a, b):
    import numpy as np

    def prediction(row):
        return PRIVATE / "cases" / row["id"] / row["attempt"] / "prediction.npz"

    results = {}
    with np.load(prediction(a), allow_pickle=False) as x, np.load(prediction(b), allow_pickle=False) as y:
        for key in x.files:
            difference = x[key].astype(np.float64) - y[key]
            results[key] = dict(max_abs=float(abs(difference).max()), relative_l2=float(
                np.linalg.norm(difference.ravel()) / max(1e-12, np.linalg.norm(x[key].ravel()))))
    return results


def qualify():
    plan = verify()
    data = collect(plan["probes"])
    rows = {row["id"]: row for row in data["rows"]}
    absent = [job["id"] for job in plan["probes"] if job["id"] not in rows]
    if absent:
        raise ValueError(f"Required engineering runs are missing: {absent}")
    if len({tuple(row["runtime"].items()) for row in rows.values()}) != 1:
        raise ValueError("The qualification runtimes differ")
    zero, four = rows["q_zero"], rows["q_four"]
    plain, geo, dummy = (rows[name] for name in ("q_plain_fixed", "q_geo_fixed", "q_leak_fixed"))
    last14, un14 = (rows[name] for name in ("q_last14_fixed", "q_un14_fixed"))
    real_a, real_b = (rows[name] for name in ("q_geo_real_a", "q_geo_real_b"))
    audited = [row["audit"] for row in rows.values() if row["audit"] is not None]
    g = geo["audit"]
    p = plain["audit"]
    r = real_a["audit"]
    factors = ["boundary_correction.lora_A", "boundary_correction.lora_B"]
    common = list(p["gradients"]["0"])
    checks = dict(
        zero_path_exact=zero["output_sha256"] == four["output_sha256"],
        cache14_exact=(last14["output_sha256"] == un14["output_sha256"]
                       and last14["audit"]["gradients"] == un14["audit"]["gradients"]
                       and last14["audit"]["render_orders"] == un14["audit"]["render_orders"]
                       and last14["audit"]["taps"] == un14["audit"]["taps"]),
        identity_predictions=(g["initial_prediction"] == p["initial_prediction"]
                              and g["update_predictions"][0] == p["update_predictions"][0]
                              and g["taps"][:2] == p["taps"][:2]),
        identity_loss=(geo["losses"]["first_loss"] == plain["losses"]["first_loss"]),
        same_suffix_initial=all(g["adapter_initial"][key] == p["adapter_initial"][key] for key in common),
        same_initial_suffix_gradients=all(g["gradients"]["0"][key] == p["gradients"]["0"][key] for key in common),
        matched_render_orders=(g["render_orders"] == p["render_orders"]
                               and r["render_orders"] == real_b["audit"]["render_orders"]),
        reference_isolation=(geo["reference_sha256"] != dummy["reference_sha256"]
                             and geo["input_sha256"] == dummy["input_sha256"]
                             and geo["output_sha256"] == dummy["output_sha256"]),
        frozen_weights=all(a["frozen_start_sha256"] == a["frozen_end_sha256"] for a in audited),
        frozen_taps=all(len({call[str(tap)] for call in r["taps"]}) == 1 for tap in (4, 11)),
        first_gradients=(not r["gradients"]["0"][factors[0]]["nonzero"]
                         and r["gradients"]["0"][factors[1]]["nonzero"]),
        learned_factors=all(r["gradients"]["1"][key]["nonzero"]
                            and r["adapter_initial"][key] != r["adapter_final"][key] for key in factors),
        learned_suffix=any(r["adapter_initial"][key] != r["adapter_final"][key] for key in common),
        final_cut_corrected=(r["boundary_forwards"][-1] == dict(changed=True, grad_enabled=False)),
    )
    if not all(checks.values()):
        print(checks)
        raise ValueError("Resolve qualification failures before scoring")
    full, last = (rows[name]["peak_allocated"] for name in ("q_full40_peak", "q_last14_peak"))
    screen = dict(eligible=last / full <= plan["thresholds"]["peak_allocated"], full40_bytes=full,
                  last14_bytes=last, ratio=last / full)
    compact = [{key: value for key, value in row.items() if key != "audit"} for row in rows.values()]
    write_new(PUBLIC / "qualification.json", dict(
        development_sha256=sha(MANIFEST, text=True), qualified=True, synthetic_checks=True, checks=checks,
        same_seed_real_loss_variation=prediction_difference(real_a, real_b),
        last14_screen=screen, runs=compact, attempts=data["attempts"], rental_compute_usd=0))
    print(f"All {len(checks)} checks passed. Last-14 memory ratio {screen['ratio']:.4f}; eligible={screen['eligible']}.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--summarize", action="store_true")
    args = cli.parse_args()
    if args.summarize:
        qualify()
    else:
        fixture()
