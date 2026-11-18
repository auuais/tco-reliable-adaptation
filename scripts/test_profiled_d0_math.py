"""Synthetic D0 tests only: no real model, source measurements or cache arrays."""
import json
import unittest

import numpy as np

import profiled_d0_math as d0


class WeightTests(unittest.TestCase):
    def assertClose(self, value, expected):
        np.testing.assert_allclose(value, expected, rtol=d0.SYNTHETIC_RTOL, atol=d0.SYNTHETIC_ATOL)

    def test_rectangular_gram_matches_independent_direct_product(self):
        rng = np.random.default_rng(520)
        for rank, inputs, outputs, alpha in ((4, 7, 11, 1), (2, 9, 3, 3), (1, 5, 8, -2)):
            A, B = rng.normal(size=(rank, inputs)), rng.normal(size=(outputs, rank))
            base = rng.normal(size=(outputs, inputs))
            actual = d0.effective_weight_stats(A, B, np.linalg.norm(base), rank, alpha)
            expected = np.linalg.norm((alpha / rank) * (B @ A))
            self.assertClose(actual["update_frobenius"], expected)
            self.assertClose(actual["relative_frobenius"], expected / np.linalg.norm(base))
            self.assertEqual(actual["factor_A_shape"], [rank, inputs])
            self.assertEqual(actual["factor_B_shape"], [outputs, rank])
            self.assertEqual(actual["scaling"], alpha / rank)
            json.dumps(actual, allow_nan=False)

    def test_zero_B_is_zero_effective_update_despite_nonzero_A(self):
        result = d0.effective_weight_stats(np.ones((4, 7)), np.zeros((11, 4)), 3)
        self.assertEqual(result["update_frobenius"], 0)
        self.assertEqual(result["relative_frobenius"], 0)

    def test_factor_rescaling_leaves_effective_update_unchanged(self):
        rng = np.random.default_rng(44)
        A, B = rng.normal(size=(4, 7)), rng.normal(size=(11, 4))
        base = d0.effective_weight_stats(A, B, 12)
        for c in (0.01, -7, 400):
            other = d0.effective_weight_stats(A*c, B/c, 12)
            self.assertClose(other["update_frobenius"], base["update_frobenius"])

    def test_zero_denominator_is_not_zero_or_infinity(self):
        result = d0.effective_weight_stats(np.ones((4, 7)), np.ones((11, 4)), 0)
        self.assertTrue(result["zero_base_denominator"])
        self.assertIsNone(result["relative_frobenius"])
        json.dumps(result, allow_nan=False)

    def test_invalid_factor_orientation_shape_type_and_rank(self):
        A, B = np.ones((4, 7)), np.ones((11, 4))
        for a, b, rank in ((A.T, B, 4), (A, B.T, 4), (A, B, 3), (A.ravel(), B, 4),
                           (A, B, 0), (A, B, True), (A, B, 4.0), (A[:, :0], B, 4),
                           (A.astype(complex), B, 4), (A.astype(bool), B, 4)):
            with self.subTest(shape=np.shape(a), rank=rank), self.assertRaises(ValueError):
                d0.effective_weight_stats(a, b, 1, rank)

    def test_nonfinite_factors_and_invalid_base_or_alpha_rejected(self):
        A, B = np.ones((4, 7)), np.ones((11, 4))
        for value in (np.nan, np.inf, -np.inf):
            bad = A.copy(); bad[0, 0] = value
            with self.assertRaises(ValueError): d0.effective_weight_stats(bad, B, 1)
            with self.assertRaises(ValueError): d0.effective_weight_stats(A, B, value)
            with self.assertRaises(ValueError): d0.effective_weight_stats(A, B, 1, alpha=value)
        with self.assertRaises(ValueError): d0.effective_weight_stats(A, B, -1)

    def test_aggregate_is_ratio_of_norms_not_mean_layer_ratios(self):
        rows = [dict(update_frobenius=3.0, base_frobenius=4.0),
                dict(update_frobenius=4.0, base_frobenius=12.0)]
        result = d0.aggregate_weight_stats(rows)
        self.assertEqual(result["update_frobenius"], 5)
        self.assertClose(result["base_frobenius"], np.sqrt(160))
        self.assertClose(result["relative_frobenius"], 5/np.sqrt(160))
        self.assertNotAlmostEqual(result["relative_frobenius"], (3/4 + 4/12)/2)
        self.assertEqual(result["denominator_scope"], "adapted_original_frozen_matrices_only")

    def test_empty_or_invalid_weight_aggregate_rejected(self):
        for values in ([], [dict(update_frobenius=-1, base_frobenius=1)],
                       [dict(update_frobenius=np.inf, base_frobenius=1)]):
            with self.assertRaises(ValueError): d0.aggregate_weight_stats(values)
        self.assertIsNone(d0.aggregate_weight_stats([dict(update_frobenius=0., base_frobenius=0.)])["relative_frobenius"])


class DepthTests(unittest.TestCase):
    def assertClose(self, value, expected):
        np.testing.assert_allclose(value, expected, rtol=d0.SYNTHETIC_RTOL, atol=d0.SYNTHETIC_ATOL)

    def row(self, base, candidate):
        base = np.asarray(base, dtype=float).reshape(1, -1)
        candidate = np.asarray(candidate, dtype=float).reshape(1, -1)
        xy = np.column_stack((np.arange(base.size), np.zeros(base.size, dtype=np.int64)))
        return d0.compare_depths(base, candidate, xy)

    def test_exact_registered_grid_and_xy_order(self):
        xy = d0.grid_coordinates()
        self.assertEqual(xy.dtype, np.dtype(np.int64))
        self.assertEqual(xy.shape, (768, 2))
        self.assertEqual(xy[0].tolist(), [8, 8])
        self.assertEqual(xy[31].tolist(), [504, 8])
        self.assertEqual(xy[32].tolist(), [8, 24])
        self.assertEqual(xy[-1].tolist(), [504, 376])
        self.assertEqual(len(set(map(tuple, xy))), 768)
        self.assertTrue(((xy >= [0, 0]) & (xy < [512, 384])).all())

    def test_xy_samples_rows_and_columns_without_transpose(self):
        base = np.arange(20, dtype=float).reshape(4, 5) + 1
        candidate = base.copy(); candidate[1, 3] += 2
        actual = d0.compare_depths(base, candidate, np.array([[3, 1]]))
        self.assertEqual(actual["delta_mean"], 2)
        self.assertEqual(actual["base_rms"], 9)

    def test_scalar_change_has_zero_remaining_shape(self):
        result = self.row([1, 2, 3, 4], [1.25, 2.5, 3.75, 5])
        self.assertClose(result["scalar_coefficient"], .25)
        self.assertClose(result["shape_rms"], 0)
        self.assertClose(result["scalar_energy_fraction"], 1)
        self.assertClose(result["shape_fraction_l2"], 0)

    def test_orthogonal_change_has_zero_scalar_component(self):
        result = self.row([2, 2], [3, 1])
        self.assertEqual(result["scalar_coefficient"], 0)
        self.assertEqual(result["shape_rms"], 1)
        self.assertEqual(result["shape_fraction_l2"], 1)
        self.assertEqual(result["scalar_energy_fraction"], 0)

    def test_mixed_change_orthogonal_decomposition_and_energy(self):
        result = self.row([2, 2], [4, 2])  # scalar [1,1], shape [1,-1]
        self.assertEqual(result["scalar_coefficient"], .5)
        self.assertEqual(result["delta_rms"], np.sqrt(2))
        self.assertEqual(result["scalar_rms"], 1)
        self.assertEqual(result["shape_rms"], 1)
        self.assertEqual(result["base_shape_inner_product"], 0)
        self.assertEqual(result["decomposition_reconstruction_max_abs"], 0)
        self.assertEqual(result["scalar_energy_fraction"] + result["shape_energy_fraction"], 1)

    def test_signed_absolute_and_shape_quantiles_use_linear_convention(self):
        result = self.row([2, 2, 2, 2], [1, 2, 3, 4])
        self.assertEqual(result["quantile_probabilities"], list(d0.QUANTILES))
        self.assertClose(result["signed_delta_quantiles"], np.quantile([-1, 0, 1, 2], d0.QUANTILES, method="linear"))
        self.assertClose(result["absolute_delta_quantiles"], np.quantile([1, 0, 1, 2], d0.QUANTILES, method="linear"))
        self.assertClose(result["absolute_shape_quantiles"], np.quantile([1.5, .5, .5, 1.5], d0.QUANTILES, method="linear"))

    def test_zero_change_preserves_undefined_fraction(self):
        result = self.row([1, 2, 3], [1, 2, 3])
        self.assertTrue(result["zero_change"])
        self.assertEqual(result["scalar_coefficient"], 0)
        self.assertEqual(result["relative_delta_l2"], 0)
        self.assertIsNone(result["shape_fraction_l2"])
        self.assertIsNone(result["scalar_energy_fraction"])
        self.assertIsNone(result["shape_energy_fraction"])

    def test_zero_baseline_has_defined_delta_but_undefined_projection(self):
        result = self.row([0, 0], [1, 2])
        self.assertTrue(result["zero_base_denominator"])
        self.assertEqual(result["delta_mean"], 1.5)
        self.assertIsNone(result["relative_delta_l2"])
        self.assertIsNone(result["scalar_coefficient"])
        self.assertIsNone(result["absolute_shape_quantiles"])
        self.assertFalse(result["geometrically_valid"])

    def test_nonfinite_values_keep_all_coordinates_and_undefined_statistics(self):
        for baseline, candidate in (([1, 2, 3], [2, np.nan, 4]),
                                    ([1, np.inf, 3], [2, 3, 4]),
                                    ([np.nan, np.inf], [-np.inf, np.nan])):
            result = self.row(baseline, candidate)
            self.assertEqual(result["count"], len(baseline))
            self.assertFalse(result["selected_coordinates_filtered"])
            self.assertFalse(result["numeric_statistics_defined"])
            self.assertFalse(result["geometrically_valid"])
            for key in d0.DEPTH_SCALARS + d0.DEPTH_QUANTILES:
                self.assertIsNone(result[key])
            json.dumps(result, allow_nan=False)

    def test_nonpositive_finite_values_are_not_discarded(self):
        result = self.row([0, -1, 2], [1, 0, -2])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["base_nonpositive_finite_count"], 2)
        self.assertEqual(result["candidate_nonpositive_finite_count"], 2)
        self.assertEqual(result["delta_mean"], -2/3)
        self.assertTrue(result["numeric_statistics_defined"])
        self.assertFalse(result["geometrically_valid"])

    def test_invalid_pixels_outside_fixed_set_do_not_change_selected_statistic(self):
        base = np.array([[1., np.nan], [np.inf, 2.]])
        candidate = np.array([[2., -np.inf], [np.nan, 3.]])
        result = d0.compare_depths(base, candidate, np.array([[0, 0], [1, 1]]))
        self.assertTrue(result["geometrically_valid"])
        self.assertEqual(result["delta_mean"], 1)

    def test_invalid_coordinates_and_shapes_rejected(self):
        base = np.ones((3, 4))
        bad = (np.array([[4, 1]]), np.array([[-1, 1]]), np.array([[0, 3]]),
               np.empty((0, 2), dtype=int), np.array([[1., 2.]]), np.array([[1, 2], [1, 2]]),
               np.array([1, 2]), np.array([[True, False]]))
        for xy in bad:
            with self.subTest(xy=xy.tolist()), self.assertRaises(ValueError):
                d0.compare_depths(base, base, xy)
        with self.assertRaises(ValueError): d0.compare_depths(base, base.T, np.array([[0, 0]]))

    def test_equal_pair_means_are_not_point_pooled(self):
        a = self.row([1], [2])
        b = self.row([1, 1, 1], [4, 4, 4])
        result = d0.aggregate_equal_pairs([a, b])
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["total_selected_points"], 4)
        self.assertEqual(result["means"]["delta_mean"], 2)
        self.assertNotEqual(result["means"]["delta_mean"], 2.5)
        self.assertEqual(result["mean_of_pair_quantiles"]["absolute_delta_quantiles"], [2]*7)

    def test_undefined_pair_is_not_removed_from_aggregate(self):
        a = self.row([1], [2]); b = self.row([1, 1], [np.nan, 2])
        result = d0.aggregate_equal_pairs([a, b])
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["total_selected_points"], 3)
        self.assertEqual(result["pairs_with_defined_numeric_statistics"], 1)
        self.assertEqual(result["invalid_counts"]["candidate_nonfinite_count"], 1)
        self.assertIsNone(result["means"]["delta_mean"])
        self.assertIsNone(result["mean_of_pair_quantiles"]["absolute_delta_quantiles"])
        self.assertEqual(result["undefined_pairs_per_statistic"]["delta_mean"], 1)
        json.dumps(result, allow_nan=False)

    def test_equal_pair_aggregate_rejects_changed_contract(self):
        with self.assertRaises(ValueError): d0.aggregate_equal_pairs([])
        for key, value in (("selected_coordinates_filtered", True), ("count", 0),
                           ("quantile_probabilities", [0, 1]), ("schema", "other")):
            row = self.row([1], [2]); row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): d0.aggregate_equal_pairs([row])


if __name__ == "__main__":
    unittest.main()
