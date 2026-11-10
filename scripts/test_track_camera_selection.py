import unittest

import numpy as np

from camera_depth_swap import relative_rig, unproject
from track_camera_selection import fit_scales, scaled_depth_observations, validation_error


class GeometryControls(unittest.TestCase):
    def test_world_similarity_keeps_first_camera_points_in_same_depth_units(self):
        e = np.repeat(np.eye(4)[None], 2, axis=0)
        e[1, 0, 3] = -0.3
        k = np.repeat(np.diag([70., 70., 1.])[None], 2, axis=0)
        depth = np.ones((2, 3, 4))
        original = unproject(relative_rig(e, k), depth)
        q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        changed = e.copy()
        changed[:, :3, :3] = q.T
        changed[:, :3, 3] = 7 * e[:, :3, 3] - q.T @ [1., 2., 3.]
        np.testing.assert_allclose(unproject(relative_rig(changed, k, 7), depth), original)

    def test_triangulated_scale_recovers_known_depth_bias_in_each_view(self):
        e = np.repeat(np.eye(4)[None], 2, axis=0)
        e[1, 0, 3] = -0.2
        k = np.repeat(np.diag([100., 100., 1.])[None], 2, axis=0)
        p = np.column_stack((np.arange(20., 44.), np.arange(5., 29.)))
        pairs = [dict(i=0, j=1, xy_i=p, xy_j=p - [10., 0.], fold=np.arange(24) % 2)]
        depths = np.array([np.full((40, 60), 2 / 1.2), np.full((40, 60), 2 / .8)])
        edges = scaled_depth_observations(relative_rig(e, k), depths, pairs)
        scale = fit_scales(edges, 2, 0)
        np.testing.assert_allclose(scale, [1.2, .8], atol=1e-10)
        self.assertLess(validation_error(edges, scale, 1), 1e-10)
        self.assertGreater(validation_error(edges, np.ones(2), 1), 1.)
        e[1] = e[0]
        degenerate = scaled_depth_observations(relative_rig(e, k), depths, [dict(pairs[0], xy_j=p)])
        np.testing.assert_array_equal(fit_scales(degenerate, 2, None), np.ones(2))
