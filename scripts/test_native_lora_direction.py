"""Guard camera conventions, observation isolation, and signed effect decisions."""
import unittest

import numpy as np

from native_lora_direction_inputs import project, valid_pose
from score_native_lora_direction import classify, evaluate, held_mask, verdict


class DirectionFixtures(unittest.TestCase):
    def test_projection_moves_depth_into_target_color_camera(self):
        depth = np.zeros((8, 8), dtype=np.uint16)
        depth[3, 2] = 2000
        source = np.eye(4)
        source[0, 3] = 1
        target = np.eye(4)
        target[2, 3] = 1
        intrinsic = np.array([[2., 0, 0], [0, 2., 0], [0, 0, 1.]])
        p = project(depth, 17, np.array([[2, 3], [4, 4]]), intrinsic,
                    intrinsic, source, target)
        np.testing.assert_array_equal(p["xy"], [[6, 6]])
        np.testing.assert_array_equal(p["source_ids"], [[17, 2, 3]])
        np.testing.assert_array_equal(p["rows"][:, 2], [1.])

    def test_invalid_publisher_tracking_poses_are_skipped(self):
        self.assertIsNone(valid_pose(b"-1.#INF"))
        self.assertIsNone(valid_pose(b"1 0 0 0 0 1 0 0 0 0 1 -inf 0 0 0 1"))

    def test_holdout_excludes_pixel_and_source_reuse(self):
        allowed = dict(xy=np.array([[0, 0]]), source_ids=np.array([[1, 2, 3]]))
        held = dict(xy=np.array([[0, 0], [1, 1], [2, 2]]), depth=np.array([2., 2., 2.]),
                    source_ids=np.array([[1, 2, 4], [1, 2, 3], [2, 2, 3]]))
        np.testing.assert_array_equal(held_mask(allowed, held), [False, False, True])
        depth = np.ones((1, 3, 3))
        np.testing.assert_allclose(evaluate(depth, [held], [held_mask(allowed, held)], 2, 1)[0]["mae_mm"], 0.)
        depth[0, 2, 2] = 0
        self.assertIsNone(evaluate(depth, [held], [held_mask(allowed, held)], 2, 1)[0]["mae_mm"])

    def test_mirrored_slopes_and_realized_step_can_disagree(self):
        mm = dict(zero=10., repeat_zero=10., plus_full=11., repeat_full=11.,
                  minus_full=12., minus_half=12., plus_half=10.)
        objective = {"zero": 2., "plus_full": 1.}
        cfg = dict(floor_mm=.05, repeat_multiplier=2.)
        result = classify(mm, -1, objective, cfg)
        self.assertEqual(result["signs"], [-1, -1, 1])
        self.assertEqual(result["status"], "mixed")
        self.assertEqual(result["realized"], "worsened")
        mm["plus_full"] = mm["repeat_full"] = 9.
        self.assertEqual(classify(mm, -1, objective, cfg)["status"], "aligned")
        mm["repeat_zero"] = 20.
        self.assertEqual(classify(mm, -1, objective, cfg)["realized"], "within_deadband")

    def test_independent_room_agreement_rejects_decisive_counterexample(self):
        classifications = {room + "_" + side: {"classification": {"status": "aligned"}}
                           for room in ("apt1", "apt2", "office2") for side in ("earlier", "later")}
        classifications["office2_earlier"]["classification"]["status"] = "mixed"
        self.assertEqual(verdict(classifications), "aligned")
        classifications["office2_earlier"]["classification"]["status"] = "opposed"
        self.assertEqual(verdict(classifications), "mixed")


if __name__ == "__main__":
    unittest.main()
