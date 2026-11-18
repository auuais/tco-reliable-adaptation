"""Synthetic fixtures for the compensation study's exact release and gate rules."""
import unittest

from profiled_compensation_protocol import ARMS, SCENES, SIDES, jobs
from score_profiled_compensation import FRAMES, summarize


def fixture(candidate=.98, *, changed=None):
    rows = []
    for scene in SCENES:
        for side in SIDES:
            for arm in ARMS:
                mae = candidate if arm == "R1" else 1.
                if changed:
                    mae = changed.get((scene, side, arm), mae)
                for frame in FRAMES:
                    row = dict(scene=scene, side=side, arm=arm, frame=frame)
                    for endpoint in ("final", "native"):
                        row[endpoint] = dict(mae_m=mae, all_samples_valid=True)
                    rows.append(row)
    return rows


class ScoreTests(unittest.TestCase):
    def test_complete_inventory_and_orthogonal_control(self):
        items = jobs()
        self.assertEqual(len(items), 36)
        self.assertEqual(len({row["run_id"] for row in items}), 36)
        self.assertEqual(sum(row["adapt_lora"] for row in items), 30)
        for item in items:
            self.assertNotEqual(item["adaptation_source"], item["development_evaluation_source"])
            if item["arm"] in ("R0", "R1", "R2"):
                self.assertEqual(item["graph_weight"], 0)
                self.assertEqual(item["differentiate_solve"], item["arm"] != "R2")

    def test_pass_strictly_against_all_controls(self):
        result = summarize(fixture(), 0)
        self.assertEqual(result["nomination"], "R1")
        self.assertEqual(result["same_scenes_improving_over_p1"], list(SCENES))

    def test_weak_or_tied_gain_fails(self):
        self.assertIsNone(summarize(fixture(.995), 0)["nomination"])
        touched = {(scene, "later", "R0"): .98 for scene in SCENES}
        result = summarize(fixture(changed=touched), 0)
        self.assertIsNone(result["nomination"])
        self.assertIn("R1_did_not_pass_against_R0_later", result["rejection_reasons"])

    def test_directional_means_cannot_hide_different_winners(self):
        rows = fixture(.8, changed={("heads", "earlier", "R1"): 1.01,
                                    ("chess", "later", "R1"): 1.01})
        result = summarize(rows, 0)
        self.assertIsNone(result["nomination"])
        self.assertEqual(result["same_scenes_improving_over_p1"], ["fire"])

    def test_missing_invalid_and_numerical_failures_fail(self):
        rows = fixture()
        self.assertIsNone(summarize(rows, 1)["nomination"])
        with self.assertRaisesRegex(ValueError, "Full five-view"):
            summarize(rows[:-1], 0)
        next(row for row in rows if row["arm"] == "R2")["final"]["all_samples_valid"] = False
        self.assertIsNone(summarize(rows, 0)["nomination"])


if __name__ == "__main__":
    unittest.main()
