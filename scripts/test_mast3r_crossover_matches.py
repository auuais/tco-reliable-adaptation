"""Verify native-coordinate match errors and exclusions on a known two-camera plane."""
import unittest

import numpy as np

from audit_mast3r_crossover_matches import match_errors


class MatchProjectionTest(unittest.TestCase):
    def test_translation_and_invalid_pixels(self):
        intrinsic = np.array([[4., 0, 1.5], [0, 4., 1.5], [0, 0, 1.]])
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, 0, 3] = -.5
        y, x = np.indices((4, 4))
        rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsic).T
        points = np.stack([rays * 2, rays * 2 + [.5, 0, 0]])
        masks = np.ones((2, 4, 4), dtype=bool)
        matches = np.array([[2, 1, 1, 1], [2, 1, 1, 2], [-1, 1, 1, 1]])
        error, reason = match_errors(matches, points, masks, poses, intrinsic, 0, 1, (4, 4))
        np.testing.assert_allclose(error, [0, 1, 0])
        np.testing.assert_array_equal(reason, [0, 0, 1])
        masks[1, 1, 1] = False
        _, reason = match_errors(matches[:1], points, masks, poses, intrinsic, 0, 1, (4, 4))
        self.assertEqual(reason[0], 2)


if __name__ == "__main__":
    unittest.main()
