import unittest
import numpy as np

from partner_shape import fit, images, make_modes, tiles
from score_partner_shape import disjoint_test
from audit_partner_headroom import best_possible


class PartnerShapeTest(unittest.TestCase):
    def test_modes_obey_anchor_and_relief_constraints(self):
        rng = np.random.default_rng(11)
        fused = 1 + rng.normal(0, .03, (32, 48))
        fused[4::8, 4::8] = 1.
        independent = rng.normal(0, .01, (2, 32, 48))
        independent[:, 4::8, 4::8] = 0.
        single = np.stack([fused+independent[0], fused-independent[0], fused+independent[1], fused-independent[1]])
        real, shuffled, fraction = make_modes(single, fused, 1701)
        np.testing.assert_array_equal(images(tiles(fused), *fused.shape), fused)
        self.assertAlmostEqual(fraction.sum(), 1.)
        for modes in (real, shuffled):
            np.testing.assert_allclose(modes[:, 4::8, 4::8], 0., atol=1e-13)
            np.testing.assert_allclose((tiles(modes)*tiles(fused-1)).sum(-1), 0., atol=1e-13)
            self.assertAlmostEqual(np.sum(modes[0]*modes[1]), 0.)
        np.testing.assert_allclose(np.linalg.norm(real, axis=(1,2)), np.linalg.norm(shuffled, axis=(1,2)))
        self.assertGreater(np.linalg.norm(real-shuffled), .1)
        same, _, _ = make_modes(np.stack([fused]*4), fused, 1701)
        np.testing.assert_allclose(same, 0., atol=1e-15)

    def test_recovery_and_unobservable_columns(self):
        rng = np.random.default_rng(37)
        base = np.ones(50)*2.
        x = rng.normal(0, .15, (50, 2))
        truth = base + x @ [.4, -.2]
        a, _, fallback = fit(base, x, truth)
        self.assertFalse(fallback)
        np.testing.assert_allclose(a, [.4, -.2], atol=.005)
        a, _, fallback = fit(base, np.column_stack([x[:,0], x[:,0]]), truth)
        self.assertTrue(fallback)
        np.testing.assert_array_equal(a, [0,0])

    def test_overlap_after_native_pixel_resize(self):
        rows = np.array([[0., 0., 1.], [1., 1., 2.], [1., 1., 3.], [8., 8., 2.]])
        pixels, measured, excluded = disjoint_test(np.array([[0,0]]), rows)
        self.assertEqual(excluded, 1)
        np.testing.assert_array_equal(measured, [2.,2.])
        self.assertEqual(set(map(tuple,pixels)), {(1,1), (8,8)})

    def test_optimistic_headroom_respects_box(self):
        design = np.eye(2)
        base = np.ones(2)
        self.assertAlmostEqual(best_possible(base, design, base+[.2, -.4]), 0.)
        self.assertAlmostEqual(best_possible(base, design, base+[3., -2.]), 1.5)


if __name__ == "__main__":
    unittest.main()
