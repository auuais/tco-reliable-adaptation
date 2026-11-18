"""Check the coverage fix, nested provenance splits, and selection rule."""
import unittest

import numpy as np

from native_lora_step_inputs import grids, split
from score_native_lora_direction import held_mask
from score_native_lora_step import choose, summarize


class StepSelectionFixtures(unittest.TestCase):
    def test_complementary_sampling_preserves_coverage_with_no_motion(self):
        a = grids(640, 480, "earlier")
        b = grids(640, 480, "later")
        def packet(xy, frame):
            return dict(xy=xy, source_ids=np.column_stack((np.full(len(xy), frame), xy)))
        self.assertEqual(int(held_mask(packet(a, 1), packet(b, 3)).sum()), 64)
        self.assertEqual(int(held_mask(packet(b, 3), packet(a, 1)).sum()), 64)

    def test_nested_folds_reserve_partition_with_valid_counts(self):
        xy = grids(512, 384, "earlier")
        views = [dict(xy=xy, rows=np.column_stack((xy, np.ones(len(xy)))),
                      source_ids=np.column_stack((np.full(len(xy), i), xy))) for i in range(5)]
        labels, roles = split(views, "fixture/earlier")
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(sum(values) == 64 for values in zip(*roles)))
        self.assertTrue(all(n >= 8 for row in roles for n in row))

    def test_selector_chooses_smallest_step_within_tolerance(self):
        scores = dict(zero=11., quarter=10.04, half=10.03, full=10.)
        self.assertEqual(choose(scores), "quarter")
        scores["zero"] = 10.02
        self.assertEqual(choose(scores), "zero")
        with self.assertRaises(ValueError):
            choose(dict(zero=10., half=11., full=12.))

    def test_consistency_gate_rejects_average_gain_with_large_regression(self):
        def case(selected):
            return dict(valid=True, delta_mm=selected-10., relative_gain_vs_full=(10.-selected)/10.,
                        errors_mm=dict(selected=selected, zero=9., quarter=9., half=9., full=10.))
        cases = {f"{room}_{side}": case(9.) for room in ("apt1", "apt2", "office2")
                 for side in ("earlier", "later")}
        self.assertEqual(summarize(cases)["verdict"], "consistent_gain")
        cases["office2_later"] = case(10.3)
        self.assertEqual(summarize(cases)["verdict"], "mixed_or_small_gain")


if __name__ == "__main__":
    unittest.main()
