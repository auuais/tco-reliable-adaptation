"""Synthetic-only tests of the D1 calibration response and fixed FD contract."""
import json
import unittest

import numpy as np

import profiled_d1_math as d1


def generic_fixture():
    rng = np.random.default_rng(6027)
    A0 = np.vstack((2*np.eye(5), np.eye(5), .1*rng.normal(size=(6, 5))))
    Q0 = np.vstack((np.eye(5), .2*rng.normal(size=(3, 5))))
    b = A0 @ np.array([1.1, .9, 1.03, .97, 1.02]) + .02*rng.normal(size=len(A0))
    d = Q0 @ np.ones(5) + .05*rng.normal(size=len(Q0))
    A1 = A0 + .01*rng.normal(size=A0.shape)
    Q1 = Q0 + .01*rng.normal(size=Q0.shape)
    return A0, Q0, A1, Q1, b, d, .02


def fd_rows(decisions):
    return [dict(h=h, residual_vector_pass=value, loss_pass=value,
                 perturbed_stationarity_pass=value, pass_check=value)
            for h, value in zip(d1.FD_STEPS, decisions)]


class ResponseTests(unittest.TestCase):
    def assertClose(self, first, second, rtol=1e-11, atol=1e-11):
        np.testing.assert_allclose(first, second, rtol=rtol, atol=atol)

    def test_ridge_solve_matches_augmented_least_squares(self):
        A0, _, _, _, b, _, ridge = generic_fixture()
        augmented = np.vstack((A0, np.sqrt(ridge)*np.eye(5)))
        target = np.r_[b, np.sqrt(ridge)*np.ones(5)]
        expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
        self.assertClose(d1.solve_scales(A0, b, ridge), expected)

    def test_exact_finite_identity_and_losses_independently_recomputed(self):
        A0, Q0, A1, Q1, b, d, ridge = generic_fixture()
        result = d1.finite_response(A0, Q0, A1, Q1, b, d, ridge)
        s0 = np.linalg.solve(A0.T@A0+ridge*np.eye(5), A0.T@b+ridge*np.ones(5))
        s1 = np.linalg.solve(A1.T@A1+ridge*np.eye(5), A1.T@b+ridge*np.ones(5))
        e0, e1 = Q0@s0-d, Q1@s1-d
        self.assertClose(result["e1_minus_e0"], (Q1-Q0)@s0+Q1@(s1-s0))
        self.assertClose(result["J0_minus_J1"], .5*np.sum(e0**2)-.5*np.sum(e1**2))
        self.assertTrue(result["identity_pass"])
        self.assertTrue(result["pass_all"])
        self.assertFalse(result["fixed_s0_counterfactual_is_detached_gradient_arm"])
        json.dumps(result, allow_nan=False)

    def test_full_rank_zero_ridge_column_scaling_is_absorbed(self):
        A0, Q0, _, _, b, d, _ = generic_fixture()
        D = np.diag([1.1, .9, 1.03, .96, 1.2])
        result = d1.analyze_response(A0, Q0, A0@D, Q0@D, b, d, 0.)
        finite, derivative = result["finite_response"], result["derivative"]
        self.assertClose(finite["s1"], np.linalg.solve(D, finite["s0"]))
        self.assertClose(finite["e1"], finite["e0"])
        self.assertGreater(finite["norm_t"], .01)
        self.assertClose(finite["cosine_t_c"], -1.)
        self.assertLess(finite["norm_t_plus_c_over_norm_t"], 1e-12)
        self.assertClose(derivative["query_full_response"], np.zeros(len(Q0)))
        self.assertTrue(result["pass_all"])
        self.assertEqual(finite["query_loss_interpretation"]["classification"], "unresolved")
        self.assertEqual(finite["compensation_interpretation"]["classification"], "attenuation")

    def test_ridge_breaks_exact_column_scaling_invariance(self):
        A0, Q0, _, _, _, d, _ = generic_fixture()
        b = A0 @ np.ones(5)
        D = np.diag([1.1, .9, 1.03, .96, 1.2])
        result = d1.finite_response(A0, Q0, A0@D, Q0@D, b, d, .5)
        self.assertGreater(result["norm_t_plus_c"], .001)
        self.assertGreater(result["norm_t_plus_c_over_norm_t"], 0)
        self.assertTrue(result["pass_all"])

    def test_query_shape_change_without_support_change_has_no_compensation(self):
        A = np.eye(5)
        Q0 = np.vstack((np.eye(5), np.ones((1, 5))))
        Q1 = Q0.copy(); Q1[-1, 0] += .1
        result = d1.analyze_response(A, Q0, A, Q1, np.ones(5), np.zeros(6), .1)
        finite, derivative = result["finite_response"], result["derivative"]
        self.assertEqual(finite["norm_c"], 0)
        self.assertIsNone(finite["cosine_t_c"])
        self.assertEqual(finite["norm_t_plus_c_over_norm_t"], 1)
        self.assertClose(derivative["full_outer_derivative"], derivative["detached_at_same_state_outer_derivative"])
        self.assertTrue(result["pass_all"])

    def test_exact_derivative_needs_nonzero_support_residual_term(self):
        A = np.vstack((np.eye(5), np.eye(5)))
        b = np.r_[np.full(5, 1.1), np.full(5, .9)]
        Q, d = 1.2*np.eye(5), np.full(5, .8)
        dotA = np.vstack((.05*np.eye(5), np.zeros((5, 5))))
        dotQ = .02*np.eye(5)
        result = d1.directional_response(A, Q, b, d, .2, dotA, dotQ)
        self.assertTrue(result["pass_all"])
        self.assertGreater(result["residual_response_component_norm"], .001)
        self.assertClose(result["scale_response"], np.full(5, -(.05-.005)/2.2))
        wrong = np.asarray(result["query_direct_tQ"])+np.asarray(result["gauss_newton_response_component"])
        fd = np.asarray(result["finite_differences"][-1]["finite_difference_residual"])
        self.assertFalse(np.allclose(wrong, fd, rtol=d1.FD_RTOL, atol=d1.FD_ATOL))

    def test_full_and_detached_use_the_identical_forward_state(self):
        result = d1.analyze_response(*generic_fixture())["derivative"]
        self.assertEqual(result["full_forward_loss"], result["detached_forward_loss"])
        self.assertTrue(result["forward_state_identical"])
        e = np.asarray(result["residual"])
        self.assertClose(result["full_outer_derivative"], e @ result["query_full_response"])
        self.assertClose(result["detached_at_same_state_outer_derivative"], e @ result["query_direct_tQ"])
        self.assertNotAlmostEqual(result["full_outer_derivative"], result["detached_at_same_state_outer_derivative"], places=8)

    def test_three_fixed_central_differences_refit_both_sides(self):
        A0, Q0, A1, Q1, b, d, ridge = generic_fixture()
        result = d1.analyze_response(A0, Q0, A1, Q1, b, d, ridge)["derivative"]
        self.assertEqual([row["h"] for row in result["finite_differences"]], [.01, .003, .001])
        for row in result["finite_differences"]:
            h = row["h"]
            for sign, label in ((-1, "minus"), (1, "plus")):
                A = A1 + sign*h*(A1-A0)
                Q = Q1 + sign*h*(Q1-Q0)
                independently_fitted = np.linalg.lstsq(np.vstack((A, np.sqrt(ridge)*np.eye(5))),
                                                       np.r_[b, np.sqrt(ridge)*np.ones(5)], rcond=None)[0]
                self.assertClose(row[label+"_scales"], independently_fitted)
                self.assertClose(row[label+"_residual"], Q@independently_fitted-d)
            self.assertTrue(row["pass_check"])
        self.assertTrue(result["finite_difference_pass"])

    def test_zero_direction_ratios_are_undefined_not_epsilon_biased(self):
        A0, Q0, _, _, b, d, ridge = generic_fixture()
        result = d1.analyze_response(A0, Q0, A0, Q0, b, d, ridge)
        finite, derivative = result["finite_response"], result["derivative"]
        self.assertIsNone(finite["cosine_t_c"])
        self.assertIsNone(finite["norm_t_plus_c_over_norm_t"])
        self.assertEqual(finite["J0_minus_J1"], 0)
        self.assertTrue(derivative["zero_matrix_direction"])
        self.assertIsNone(derivative["survival_ratio"])
        self.assertIsNone(derivative["cosine_direct_compensation"])
        self.assertEqual(derivative["full_outer_derivative"], 0)
        self.assertTrue(result["pass_all"])
        json.dumps(result, allow_nan=False)

    def test_zero_query_direct_change_can_have_nonzero_compensation(self):
        A0, Q0, A1, _, b, d, ridge = generic_fixture()
        result = d1.analyze_response(A0, Q0, A1, Q0, b, d, ridge)
        self.assertEqual(result["finite_response"]["norm_t"], 0)
        self.assertGreater(result["finite_response"]["norm_c"], 0)
        self.assertIsNone(result["finite_response"]["norm_t_plus_c_over_norm_t"])
        self.assertIsNone(result["derivative"]["survival_ratio"])

    def test_ridge_solvability_is_not_reported_as_data_identifiability(self):
        A0 = np.zeros((3, 5))
        Q = np.eye(5)
        result = d1.finite_response(A0, Q, A0, Q, np.ones(3), np.ones(5), .1)
        self.assertEqual(result["state0"]["support_data_rank"], 0)
        self.assertClose(result["s0"], np.ones(5))
        self.assertTrue(result["state0"]["stationarity_pass"])

    def test_nonfinite_arrays_or_ridge_are_rejected_without_row_removal(self):
        fixture = generic_fixture()
        for item in range(6):
            for bad_value in (np.nan, np.inf, -np.inf):
                values = list(fixture)
                values[item] = values[item].copy()
                values[item].flat[0] = bad_value
                with self.subTest(item=item, value=bad_value), self.assertRaises(ValueError):
                    d1.analyze_response(*values)
        for ridge in (-1, np.nan, np.inf, True):
            with self.assertRaises(ValueError): d1.analyze_response(*fixture[:-1], ridge)

    def test_wrong_or_empty_shapes_and_nonreal_values_rejected(self):
        fixture = generic_fixture()
        for item, bad in ((0, np.ones((4, 4))), (1, np.empty((0, 5))),
                          (2, np.ones((2, 5))), (4, np.ones((16, 1))),
                          (5, np.ones(7)), (0, np.ones((16, 5), dtype=complex)),
                          (1, np.ones((8, 5), dtype=bool))):
            values = list(fixture); values[item] = bad
            with self.subTest(item=item), self.assertRaises(ValueError): d1.analyze_response(*values)

    def test_negative_or_zero_fitted_scales_rejected(self):
        for target in (-np.ones(5), np.zeros(5)):
            with self.assertRaises(ValueError): d1.solve_scales(np.eye(5), target, 0.)
        with self.assertRaises(ValueError): d1.solve_scales(np.eye(5), -np.ones(5), .1)

    def test_zero_ridge_rank_deficiency_is_rejected(self):
        with self.assertRaises(ValueError): d1.solve_scales(np.ones((8, 5)), np.ones(8), 0.)

    def test_invalid_perturbation_scales_stop_directional_probe(self):
        with self.assertRaises(ValueError):
            d1.directional_response(np.eye(5), np.eye(5), np.ones(5), np.ones(5),
                                     .001, 200*np.eye(5), np.zeros((5, 5)))

    def test_fd_acceptance_requires_complete_inventory_and_smallest_step(self):
        rows = fd_rows([True, True, True])
        self.assertTrue(d1.finite_difference_acceptance(rows))
        rows = fd_rows([False, True, True])
        self.assertTrue(d1.finite_difference_acceptance(rows))
        rows = fd_rows([True, True, False])
        self.assertFalse(d1.finite_difference_acceptance(rows))
        rows = fd_rows([False, True, False])
        self.assertFalse(d1.finite_difference_acceptance(rows))
        with self.assertRaises(ValueError): d1.finite_difference_acceptance(rows[:-1])
        with self.assertRaises(ValueError): d1.finite_difference_acceptance(rows[::-1])

    def test_fd_only_one_passing_step_is_insufficient_even_when_smallest_passes(self):
        rows = fd_rows([False, False, True])
        self.assertFalse(d1.finite_difference_acceptance(rows))
        spec = d1.tolerance_spec()
        self.assertEqual(spec["finite_difference_minimum_passes"], 2)
        self.assertTrue(spec["finite_difference_smallest_step_must_pass"])
        self.assertEqual(spec["finite_difference_steps"], [.01, .003, .001])

    def test_fd_acceptance_rejects_forged_pass_or_missing_component(self):
        for component in ("residual_vector_pass", "loss_pass", "perturbed_stationarity_pass"):
            rows = fd_rows([True, True, True]); rows[-1][component] = False
            with self.subTest(component=component), self.assertRaises(ValueError):
                d1.finite_difference_acceptance(rows)
        rows = fd_rows([True, True, True]); del rows[-1]["loss_pass"]
        with self.assertRaises(ValueError): d1.finite_difference_acceptance(rows)

    def test_unresolved_nonlinear_fd_preserves_failed_evidence_without_step_search(self):
        result = d1.directional_response(np.eye(5), np.eye(5), np.ones(5), np.full(5, .5),
                                         1., 100*np.eye(5), np.zeros((5, 5)))
        self.assertFalse(result["pass_all"])
        self.assertFalse(result["finite_difference_pass"])
        self.assertEqual([row["h"] for row in result["finite_differences"]], list(d1.FD_STEPS))
        self.assertTrue(all(not row["pass_check"] for row in result["finite_differences"]))
        self.assertTrue(all(min(row["minus_scales"]+row["plus_scales"]) > 0
                            for row in result["finite_differences"]))
        json.dumps(result, allow_nan=False)

    def test_folds_and_inference_paths_are_not_averaged_or_selected(self):
        fixture = generic_fixture()
        A0, Q0, A1, Q1, b, d, ridge = fixture
        first = d1.analyze_response(*fixture, with_derivative=False)
        reverse = d1.analyze_response(A1, Q1, A0, Q0, b, d, ridge, with_derivative=False)
        self.assertClose(first["finite_response"]["J0_minus_J1"], -reverse["finite_response"]["J0_minus_J1"])
        self.assertFalse(first["folds_aggregated"])
        self.assertFalse(reverse["folds_aggregated"])
        self.assertIsNone(first["derivative"])
        self.assertFalse(first["accuracy_improvement_established"])
        self.assertFalse(first["redesign_authorized"])
        self.assertFalse(first["rental_authorized"])
        json.dumps([first, reverse], allow_nan=False)


class InterpretationTests(unittest.TestCase):
    def test_exact_tie_and_one_ulp_loss_change_remain_unresolved(self):
        for J1 in (1., np.nextafter(1., 0.), np.nextafter(1., 2.)):
            actual = d1.classify_query_loss_change(1., J1)
            self.assertEqual(actual["classification"], "unresolved")
            self.assertEqual(actual["J0_minus_J1"], 1.-J1)
            self.assertTrue(actual["within_arithmetic_resolution"])
            self.assertFalse(actual["statistical_test_performed"])
            self.assertFalse(actual["accuracy_improvement_established"])

    def test_resolved_gain_and_loss_preserve_raw_direction(self):
        gain = d1.classify_query_loss_change(1., 1.-1e-8)
        loss = d1.classify_query_loss_change(1.-1e-8, 1.)
        self.assertEqual(gain["classification"], "decrease")
        self.assertEqual(loss["classification"], "increase")
        self.assertEqual(gain["J0_minus_J1"], -loss["J0_minus_J1"])
        self.assertTrue(gain["resolved_decrease"])
        self.assertTrue(loss["resolved_increase"])
        self.assertFalse(gain["within_arithmetic_resolution"])

    def test_loss_floor_is_absolute_plus_relative_and_not_a_percent_gain_gate(self):
        unresolved = d1.classify_query_loss_change(1e9, 1e9-.001)
        resolved = d1.classify_query_loss_change(1e9, 1e9-1.)
        self.assertEqual(unresolved["arithmetic_floor"], 1e-10+1e-10*1e9)
        self.assertEqual(unresolved["classification"], "unresolved")
        self.assertEqual(resolved["classification"], "decrease")
        self.assertEqual(d1.classify_query_loss_change(1e-12, 0.)["classification"], "unresolved")
        spec = d1.tolerance_spec()["query_loss_resolution"]
        self.assertEqual(spec["atol"], 1e-10)
        self.assertEqual(spec["rtol"], 1e-10)

    def test_invalid_query_losses_rejected(self):
        for value in (-1., np.nan, np.inf, True):
            with self.assertRaises(ValueError): d1.classify_query_loss_change(value, 1.)
            with self.assertRaises(ValueError): d1.classify_query_loss_change(1., value)

    def test_attenuation_requires_opposition_and_resolved_norm_reduction(self):
        actual = d1.classify_compensation(np.array([1., 0.]), np.array([-.5, 0.]))
        self.assertEqual(actual["classification"], "attenuation")
        self.assertTrue(actual["opposing"])
        self.assertEqual(actual["norm_reduction"], .5)
        self.assertEqual(actual["survival_ratio"], .5)
        self.assertTrue(actual["resolved_attenuation"])
        self.assertFalse(actual["harmfulness_established"])
        self.assertFalse(actual["parameter_space_generalization_established"])
        self.assertFalse(actual["redesign_authorized"])

    def test_opposition_can_amplify_when_compensation_overshoots(self):
        actual = d1.classify_compensation(np.array([1., 0.]), np.array([-3., 0.]))
        self.assertEqual(actual["classification"], "amplification")
        self.assertTrue(actual["opposing"])
        self.assertEqual(actual["survival_ratio"], 2.)
        self.assertFalse(actual["resolved_attenuation"])
        self.assertTrue(actual["resolved_amplification"])
        self.assertEqual(actual["arithmetic_floor"], 1e-10+3e-10)

    def test_orthogonal_compensation_amplifies_and_is_not_opposing(self):
        actual = d1.classify_compensation(np.array([1., 0.]), np.array([0., .5]))
        self.assertEqual(actual["classification"], "amplification")
        self.assertFalse(actual["opposing"])
        self.assertGreater(actual["survival_ratio"], 1.)

    def test_neutral_includes_exact_zero_and_subfloor_norm_change(self):
        for vector in (np.zeros(2), np.array([-1e-12, 0.]), np.array([0., 1e-6])):
            actual = d1.classify_compensation(np.array([1., 0.]), vector)
            self.assertEqual(actual["classification"], "neutral")
            self.assertFalse(actual["resolved_attenuation"])
            self.assertFalse(actual["resolved_amplification"])
            self.assertLessEqual(abs(actual["norm_reduction"]), actual["arithmetic_floor"])

    def test_zero_direct_vector_is_undefined_even_with_nonzero_compensation(self):
        for vector in (np.zeros(2), np.array([1., 0.])):
            actual = d1.classify_compensation(np.zeros(2), vector)
            self.assertEqual(actual["classification"], "undefined")
            self.assertIsNone(actual["survival_ratio"])
            self.assertIsNone(actual["opposing"])
            self.assertIsNone(actual["cosine_direct_compensation"])
            json.dumps(actual, allow_nan=False)

    def test_norm_floor_scales_with_vector_magnitudes(self):
        small = d1.classify_compensation(np.array([1e9]), np.array([-.05]))
        resolved = d1.classify_compensation(np.array([1e9]), np.array([-.2]))
        self.assertEqual(small["classification"], "neutral")
        self.assertEqual(resolved["classification"], "attenuation")
        self.assertEqual(small["arithmetic_floor"], 1e-10+1e-10*1e9)
        spec = d1.tolerance_spec()["compensation_norm_resolution"]
        self.assertEqual(spec["atol"], 1e-10)
        self.assertEqual(spec["rtol"], 1e-10)

    def test_nonfinite_or_mismatched_compensation_vectors_are_rejected(self):
        for vector in (np.array([np.nan, 1.]), np.array([np.inf, 1.]), np.ones(3), np.ones((1, 2))):
            with self.assertRaises(ValueError): d1.classify_compensation(np.ones(2), vector)


if __name__ == "__main__":
    unittest.main()
