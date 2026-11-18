"""Synthetic CPU fixtures for the leaf surface evaluator; no study artifacts."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import profiled_m3_dense as dense


ROOT = Path(__file__).resolve().parents[1]


def fixture(dtype=np.float64):
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(2, 4, 5, 3)).astype(dtype)
    reference[..., 2] += 2.
    prediction = (reference * 1.1 + [.05, -.03, .08]).astype(dtype)
    prediction[0, 2, 3] += [2., -1., .7]
    mask = np.ones(reference.shape[:-1], dtype=bool)
    mask[:, 0, 0] = False
    return prediction, reference, mask


def fake_backend(*, no_correspondences=False, invalid_transform=False, invalid_normals=False):
    """Explicit recording stub; these tests do not claim a genuine ICP run."""
    calls = []

    class Cloud:
        def __init__(self):
            self.points = np.empty((0, 3), dtype=np.float64)
            self.normals = np.empty((0, 3), dtype=np.float64)

        def transform(self, matrix):
            calls.append(("transform", matrix.copy()))
            self.points = self.points @ matrix[:3, :3].T + matrix[:3, 3]
            return self

        def estimate_normals(self):
            calls.append(("normals", len(self.points)))
            self.normals = np.tile([0., 0., 1.], (len(self.points), 1))
            if invalid_normals:
                self.normals[0, 0] = np.nan

    def registration(pred, ref, threshold, initial, estimator):
        calls.append(("icp", pred.points.copy(), ref.points.copy(), threshold, initial.copy(), estimator))
        transform = np.eye(4)
        transform[:3, 3] = [.01, -.02, .005]
        if invalid_transform:
            transform[0, 0] = np.nan
        count = 0 if no_correspondences else len(pred.points)
        ids = np.arange(count, dtype=np.int32)
        return SimpleNamespace(transformation=transform, fitness=0. if no_correspondences else 1.,
                               inlier_rmse=0. if no_correspondences else .04,
                               correspondence_set=np.stack([ids, ids], axis=-1))

    backend = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=Cloud),
        utility=SimpleNamespace(Vector3dVector=lambda value: np.asarray(value, dtype=np.float64).copy()),
        pipelines=SimpleNamespace(registration=SimpleNamespace(
            registration_icp=registration, TransformationEstimationPointToPoint=lambda: "rigid_point_to_point")))
    return backend, calls


class DenseKernelFixtures(unittest.TestCase):
    def test_metric_helper_asts_are_pinned_copies(self):
        original = ast.parse((ROOT / "third_party/TCO/evaluation/mv_recon/utils.py").read_text(encoding="utf-8"))
        actual = ast.parse((ROOT / "scripts/profiled_m3_dense.py").read_text(encoding="utf-8"))
        for name in ("umeyama", "accuracy", "completion"):
            first = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
            second = next(node for node in actual.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(first, include_attributes=False), ast.dump(second, include_attributes=False))

    def test_open3d_is_lazy(self):
        tree = ast.parse((ROOT / "scripts/profiled_m3_dense.py").read_text(encoding="utf-8"))
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("open3d" in ast.dump(node) for node in imports))
        p, r, m = fixture()
        p[0, 0, 0, 0] = np.nan
        with patch.object(dense, "_open3d", side_effect=AssertionError("should not import")):
            with self.assertRaisesRegex(ValueError, "Nonfinite prediction"):
                dense.evaluate(p, r, m)

    def test_shapes_dtypes_and_mask_are_strict(self):
        p, r, m = fixture()
        cases = [(p[0], r[0], m[0]), (p, r[:, :, :-1], m), (p, r, m.astype(np.uint8)),
                 (p.astype(np.int32), r, m), (p, r, m[:, :, :-1])]
        for args in cases:
            with self.subTest(shapes=[a.shape for a in args]), self.assertRaises(ValueError):
                dense.validate_inputs(*args)

    def test_too_few_fixed_points_rejected(self):
        p, r, m = fixture()
        for count in (0, 1, 2):
            mask = np.zeros_like(m)
            mask.flat[:count] = True
            with self.assertRaisesRegex(ValueError, "At least three"):
                dense.validate_inputs(p, r, mask)

    def test_nonfinite_prediction_outside_mask_rejected(self):
        p, r, m = fixture()
        p[~m] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite prediction"):
            dense.validate_inputs(p, r, m)

    def test_reference_holes_may_be_undefined(self):
        p, r, m = fixture()
        r[~m] = np.nan
        backend, _ = fake_backend()
        with patch.object(dense, "_open3d", return_value=backend):
            result = dense.evaluate(p, r, m)
        self.assertTrue(np.isfinite(result["arrays"]["reference_points"]).all())
        self.assertEqual(result["metrics"]["valid_points"], m.sum())

    def test_nonfinite_masked_reference_rejected(self):
        p, r, m = fixture()
        r[0, 1, 1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite fixed-mask reference"):
            dense.validate_inputs(p, r, m)

    def test_zero_variance_has_no_fallback(self):
        p, r, m = fixture()
        for first, second in ((np.ones_like(p), r), (p, np.ones_like(r))):
            with self.assertRaisesRegex(ValueError, "zero/nonfinite scene variance"):
                dense.validate_inputs(first, second, m)

    def test_variance_overflow_rejected(self):
        p, r, m = fixture(np.float32)
        p[0] = np.float32(1e30)
        p[1] = np.float32(-1e30)
        with self.assertRaisesRegex(ValueError, "zero/nonfinite scene variance"):
            dense.validate_inputs(p, r, m)

    def test_kernel_keeps_population_order_and_input_bytes(self):
        p, r, m = fixture(np.float32)
        originals = [value.copy() for value in (p, r, m)]
        backend, calls = fake_backend()
        with patch.object(dense, "_open3d", return_value=backend):
            output = dense.evaluate(p, r, m)
        for original, after in zip(originals, (p, r, m)):
            np.testing.assert_array_equal(original, after)
        np.testing.assert_array_equal(output["arrays"]["reference_points"], r[m].astype(np.float64))
        scale, rotation, translation = dense.umeyama(p[m].T, r[m].T)
        before = (scale * np.einsum("nhwj,ij->nhwi", p, rotation) + translation.T)[m]
        np.testing.assert_array_equal(output["arrays"]["similarity_aligned_predicted"], before)
        np.testing.assert_array_equal(output["arrays"]["aligned_predicted"], before + [.01, -.02, .005])
        icp_calls = [call for call in calls if call[0] == "icp"]
        self.assertEqual(len(icp_calls), 1)
        self.assertEqual(icp_calls[0][3], .1)
        np.testing.assert_array_equal(icp_calls[0][4], np.eye(4))
        self.assertEqual(icp_calls[0][5], "rigid_point_to_point")
        self.assertEqual(sum(call[0] == "normals" for call in calls), 2)
        self.assertIsNone(output["diagnostics"]["icp_iterations"])
        self.assertFalse(output["diagnostics"]["release_authorized"])

    def test_finite_negative_coordinates_are_not_filtered(self):
        p, r, m = fixture()
        p[0, 1, 1, 2] = -2.
        backend, calls = fake_backend()
        with patch.object(dense, "_open3d", return_value=backend):
            output = dense.evaluate(p, r, m)
        self.assertEqual(output["metrics"]["valid_points"], int(m.sum()))
        self.assertEqual(output["arrays"]["aligned_predicted"].shape, (int(m.sum()), 3))

    def test_zero_icp_correspondences_are_reported_not_retried(self):
        p, r, m = fixture()
        backend, calls = fake_backend(no_correspondences=True)
        with patch.object(dense, "_open3d", return_value=backend):
            output = dense.evaluate(p, r, m)
        self.assertEqual(output["diagnostics"]["icp_correspondence_count"], 0)
        self.assertEqual(output["diagnostics"]["icp_fitness"], 0.)
        self.assertEqual(sum(call[0] == "icp" for call in calls), 1)
        self.assertEqual(output["arrays"]["icp_correspondences_sorted"].shape, (0, 2))

    def test_invalid_icp_or_normals_are_failures(self):
        p, r, m = fixture()
        for option in ("invalid_transform", "invalid_normals"):
            backend, _ = fake_backend(**{option: True})
            with self.subTest(option=option), patch.object(dense, "_open3d", return_value=backend), self.assertRaises(ValueError):
                dense.evaluate(p, r, m)

    def test_nn_directions_cutoff_and_unsigned_normals(self):
        ref = np.array([[0., 0., 0.], [1., 0., 0.]])
        pred = np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
        ref_normals = np.tile([0., 0., 1.], (2, 1))
        pred_normals = np.tile([0., 0., -1.], (3, 1))
        accuracy = dense.accuracy(ref, pred, ref_normals, pred_normals)
        completeness = dense.completion(ref, pred, ref_normals, pred_normals)
        self.assertEqual(accuracy, (4./3., 1., 1., 1.))
        self.assertEqual(completeness, (.5, .5, 1., 1.))
        self.assertGreater(accuracy[0], .1)

    def test_sim3_known_recovery_and_correspondence_order(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(3, 50))
        a = .28
        rotation = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        y = 1.3 * rotation @ x + np.array([[.1], [-.2], [.4]])
        scale, estimated_rotation, translation = dense.umeyama(x, y)
        np.testing.assert_allclose(scale * estimated_rotation @ x + translation, y, rtol=0, atol=3e-15)
        bad_scale, bad_rotation, bad_translation = dense.umeyama(x, y[:, ::-1])
        self.assertGreater(float(np.mean((bad_scale * bad_rotation @ x + bad_translation - y) ** 2)), .1)

    def test_paired_rmse_is_not_nn_error(self):
        prediction = np.array([[0., 0., 0.], [1., 0., 0.]])
        reference = prediction[::-1].copy()
        self.assertEqual(dense.paired_rmse(prediction, reference), 1.)
        self.assertEqual(dense.accuracy(reference, prediction)[0], 0.)


if __name__ == "__main__":
    unittest.main()
