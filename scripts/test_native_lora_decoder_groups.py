"""CPU fixtures for the registered MASt3R decoder group trial."""
from copy import deepcopy
import unittest

import numpy as np

from native_lora_decoder_groups import (ARCHIVED, POLICIES, STATES, choose, group, summarize)
from native_lora_direction_inputs import project
from replay_native_lora_decoder_groups import replay
from score_native_lora_direction import evaluate, held_mask


def synthetic_evidence():
    def views(count, error):
        return [dict(view=v, count=count, invalid_predictions=0, mae_mm=error) for v in range(5)]

    entries, sealed, results = [], {}, {}
    for env in ("a", "b", "c"):
        for side in ("earlier", "later"):
            job = f"{env}_{side}"
            coverage = [dict(count=16) for _ in range(5)]
            entries.append(dict(id=job, room=env, side=side,
                                packet=dict(counts=[[8]*5]*3, held_coverage=coverage)))
            run, score = {}, {}
            observed = {key: 90 if key == "e050_l150" else 100 for key in STATES}
            for key, pair in STATES.items():
                value = observed[key]
                run[key] = dict(candidate=key, multipliers=list(pair), g0=1.25, native_sha256=job+key,
                                selector_mae_mm=value, selector_rows=views(8, value))
                score[key] = dict(applied_scale=1.25, native_sha256=job+key, views=views(16, value))
            choices = dict(scalar="s000", split="e050_l150")
            sealed[job] = dict(job=job, states=run, selected=choices, group_norms_sq=[1, 1])
            results[job] = dict(room=env, side=side, coverage=coverage, states=score, errors_mm=observed,
                                summary_mm=dict(scalar_best=100, split_best=90, scalar_selected=100,
                                                split_selected=90, s000=100, s100=100, legacy_selected=100),
                                selected=choices, hindsight_best=choices, legacy_selected="s000")
    return (dict(policies={p: {k: list(v) for k, v in s.items()} for p, s in POLICIES.items()},
                 development_cases=entries, archived=ARCHIVED, min_partition=8, min_held=16,
                 selection_tolerance_mm=.05, hindsight_relative_gain=.01),
            dict(cases=sealed), dict(cases=results, **summarize(results)))


class DecoderGroupTests(unittest.TestCase):
    def test_group_inventory_and_cost(self):
        for decoder in ("dec_blocks", "dec_blocks2"):
            for block in range(12):
                self.assertEqual(group(f"{decoder}.{block}.attn.qkv.lora_A"), block//6)
        for name in ("enc_blocks.1.attn.qkv.lora_A", "dec_blocks.12.attn.qkv.lora_A"):
            with self.assertRaises(ValueError):
                group(name)
        self.assertEqual([len(p) for p in POLICIES.values()], [7, 7])
        self.assertEqual(len(STATES), 9)

    def test_real_factor_assignment_on_cpu(self):
        import torch
        from run_native_lora_decoder_groups import assign_groups, effective_norms, verify_assignments

        names = ("dec_blocks.0.attn.qkv", "dec_blocks2.11.attn.qkv")
        groups = tuple([(base+tag, torch.zeros((4, 4))) for tag in (".lora_A", ".lora_B")] for base in names)
        factors = [item for section in groups for item in section]
        before = {name: torch.eye(4) if name.endswith("A") else torch.zeros((4, 4)) for name, _ in factors}
        after = {name: torch.eye(4) for name, _ in factors}
        self.assertEqual(effective_norms(groups, before, after), [.25, .25])
        verify_assignments(factors, groups, before, after)
        assign_groups(groups, before, after, (.5, 1.5))
        for values, scale in zip(groups, (.5, 1.5)):
            torch.testing.assert_close(values[0][1], torch.eye(4))
            torch.testing.assert_close(values[1][1], torch.eye(4)*scale)

    def test_tie_prefers_norm_then_tuple(self):
        candidates = {"x": (0, 1), "y": (1, 0), "z": (1, 1)}
        errors = dict(x=11.03, y=11.03, z=11.)
        self.assertEqual(choose(errors, [1, 2], candidates), "y")
        self.assertEqual(choose(errors, [1, 1], candidates), "x")
        self.assertEqual(choose(errors | {"z": 10.97}, [1, 1], candidates), "z")

    def test_geometry_and_fixed_invalid_prediction(self):
        source = np.array([[1000, 2000, 0]], dtype=np.uint16)
        target_k = np.diag([.1, 1., 1.])
        packet = project(source, 0, np.array([[0, 0], [1, 0], [2, 0]]),
                         np.eye(3), target_k, np.eye(4), np.eye(4))
        np.testing.assert_array_equal(packet["rows"], [[0, 0, 1.]])
        allowed = dict(xy=np.array([[1, 0]]), source_ids=np.array([[0, 1, 0]]))
        held = dict(xy=np.array([[0, 0], [1, 0]]), source_ids=np.array([[1, 0, 0], [1, 1, 0]]),
                    depth=np.array([1., 2.]))
        mask = held_mask(allowed, held)
        np.testing.assert_array_equal(mask, [True, False])
        results = evaluate([np.array([[np.nan, 2.]])], [held], [mask], 1., 1)
        self.assertEqual(results, [dict(view=0, count=1, invalid_predictions=1, mae_mm=None)])

    def test_replay_rejects_wrong_room_candidate_scale_and_held_contamination(self):
        manifest, seal, scores = synthetic_evidence()
        self.assertEqual(replay(manifest, seal, scores)["verdict"], "advance")
        for field, value in (("room", "b"), ("legacy_selected", "s100")):
            altered = deepcopy(scores)
            altered["cases"]["a_earlier"][field] = value
            with self.assertRaises(ValueError):
                replay(manifest, seal, altered)
        altered = deepcopy(scores)
        altered["cases"]["a_earlier"]["states"]["s100"]["applied_scale"] = 1.3
        with self.assertRaises(ValueError):
            replay(manifest, seal, altered)
        altered_seal = deepcopy(seal)
        # A choice cannot be changed to a different state based on the held labels.
        altered_seal["cases"]["a_earlier"]["selected"]["scalar"] = "s100"
        with self.assertRaises(ValueError):
            replay(manifest, altered_seal, scores)


if __name__ == "__main__":
    unittest.main()
