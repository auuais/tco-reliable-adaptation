"""Generated-data tests; these do not establish real-model D1 acceptance."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from profiled_d1_capture import (ORIGINAL, INITIAL, FINAL, PAIRS, TorchCaptureOps,
                                _ordered_capture, counted_method, tensor_sequence_digest,
                                validate_pairs, make_views)


class FakeModel:
    def _encode_image(self):
        return 1

    def _decoder(self):
        return 2


class FakeOps:
    def __init__(self):
        self.model = FakeModel()
        self.events = []
        self.adapter_hash = INITIAL
        self.original_hash = ORIGINAL
        self.n_encoder, self.n_decoder = 5, 20
        self.frozen = True
        self.feature_hash = "fixture_features"
        self.collections = 0

    def require_frozen(self):
        if not self.frozen:
            raise ValueError("not frozen")

    def original_state(self):
        return "original"

    def digest(self, state):
        return self.original_hash if state == "original" else self.adapter_hash

    def features(self, views):
        self.events.append("features")
        return [self.model._encode_image() for _ in range(self.n_encoder)]

    def feature_digest(self, features):
        return self.feature_hash

    def collect(self, features):
        self.events.append("collect")
        for _ in range(self.n_decoder):
            self.model._decoder()
        self.collections += 1
        return "points"+str(self.collections), "confidence"+str(self.collections)

    def fuse(self, points, confidence):
        self.events.append(("fuse", points, confidence))
        return "depth_"+points, "fused_"+confidence

    def reset_adapter_seed(self):
        self.events.append("seed42")

    def install(self):
        self.events.append("install")
        self.frozen = False
        return ["module"]

    def adapters(self):
        return "adapters"

    def validate_initial(self, modules, named):
        self.events.append("validate_initial")

    def freeze(self):
        self.events.append("freeze")
        self.frozen = True

    def assign(self, named, factors):
        self.events.append("assign")
        assert self.frozen
        self.adapter_hash = factors


class OrderingTests(unittest.TestCase):
    def run_capture(self, ops=None, teacher=None, load=None):
        ops = ops or FakeOps()
        if teacher is None:
            def teacher(*args):
                ops.events.append("teacher_gate")
                return {"pass_all": True}
        if load is None:
            def load():
                ops.events.append("load_final")
                return FINAL
        return _ordered_capture(ops, list(range(5)), load, teacher, lambda: None,
                                lambda *args: ops.events.append(args[0]))

    def test_teacher_before_install_and_final_access(self):
        ops = FakeOps()
        result = self.run_capture(ops)
        self.assertLess(ops.events.index("teacher_verified"), ops.events.index("install"))
        self.assertLess(ops.events.index("freeze"), ops.events.index("load_final"))
        self.assertEqual(result["metadata"]["decoder_calls"], 40)
        self.assertEqual(result["metadata"]["encoder_calls"], 5)
        self.assertEqual(result["metadata"]["optimizer_updates"], 0)
        self.assertIn(("fuse", "points2", "confidence1"), ops.events)
        self.assertNotIn(("fuse", "points2", "confidence2"), ops.events)

    def test_teacher_failure_prevents_adapter_access(self):
        ops, load = FakeOps(), Mock()
        with self.assertRaisesRegex(ValueError, "Teacher replay"):
            self.run_capture(ops, teacher=lambda *args: {"pass_all": False}, load=load)
        load.assert_not_called()
        self.assertNotIn("install", ops.events)
        self.assertEqual(ops.collections, 1)

    def test_truthy_teacher_result_is_not_acceptance(self):
        for result in (None, True, {"pass_all": 1}, {"pass_all": "yes"}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.run_capture(teacher=lambda *args: result)

    def test_teacher_exception_propagates_without_final_access(self):
        load = Mock()
        with self.assertRaisesRegex(RuntimeError, "preserved mismatch"):
            self.run_capture(teacher=Mock(side_effect=RuntimeError("preserved mismatch")), load=load)
        load.assert_not_called()

    def test_missing_and_excess_encoder_calls_stop(self):
        for count in (4, 6):
            ops = FakeOps()
            ops.n_encoder = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_capture(ops)
            self.assertEqual(ops.collections, 0)

    def test_missing_and_excess_decoder_calls_stop(self):
        for count in (19, 21):
            ops = FakeOps()
            ops.n_decoder = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_capture(ops)
            self.assertNotIn("install", ops.events)

    def test_wrong_original_state_rejected_before_features(self):
        ops = FakeOps()
        ops.original_hash = "changed"
        with self.assertRaisesRegex(ValueError, "pristine model"):
            self.run_capture(ops)
        self.assertEqual(ops.events, [])

    def test_wrong_initial_adapter_rejects_final_access(self):
        ops, load = FakeOps(), Mock()
        ops.adapter_hash = "wrong"
        with self.assertRaisesRegex(ValueError, "initial adapter"):
            self.run_capture(ops, load=load)
        load.assert_not_called()

    def test_wrong_final_adapter_stops_second_capture(self):
        ops = FakeOps()
        with self.assertRaisesRegex(ValueError, "final P3"):
            self.run_capture(ops, load=lambda: "wrong")
        self.assertEqual(ops.collections, 1)

    def test_teacher_modified_features_rejected(self):
        ops, load = FakeOps(), Mock()
        def teacher(*args):
            ops.feature_hash = "modified"
            return {"pass_all": True}
        with self.assertRaisesRegex(ValueError, "cached features"):
            self.run_capture(ops, teacher=teacher, load=load)
        load.assert_not_called()

    def test_original_modification_after_final_capture_rejected(self):
        ops = FakeOps()
        collect = ops.collect
        def changed(features):
            result = collect(features)
            if ops.collections == 2:
                ops.original_hash = "modified"
            return result
        ops.collect = changed
        with self.assertRaisesRegex(ValueError, "original state"):
            self.run_capture(ops)

    def test_counter_restores_method_on_failure_and_blocks_excess(self):
        model = FakeModel()
        original = model._decoder
        with self.assertRaises(ValueError):
            with counted_method(model, "_decoder", 1, lambda: None) as count:
                self.assertEqual(model._decoder(), 2)
                model._decoder()
        self.assertEqual(count["calls"], 1)
        self.assertEqual(model._decoder, original)
        self.assertNotIn("_decoder", model.__dict__)

    def test_guard_failure_restores_existing_instance_method(self):
        model = FakeModel()
        assigned = lambda: 3
        model._decoder = assigned
        with self.assertRaisesRegex(RuntimeError, "cap"):
            with counted_method(model, "_decoder", 20, Mock(side_effect=RuntimeError("cap"))):
                model._decoder()
        self.assertIs(model._decoder, assigned)


class TensorBoundaryTests(unittest.TestCase):
    def test_view_preprocessing_uses_frozen_resize_and_identity(self):
        # Different constant images expose order/normalization errors without
        # duplicating the interpolation implementation in the expected result.
        rgb = np.broadcast_to(np.array([0., .25, .5, .75, 1.], dtype=np.float32)
                              [:, None, None, None], (5, 3, 392, 518)).copy()
        with patch("profiled_runtime.verify_runtime", return_value={}):
            views = make_views(rgb, "a"*64)
        self.assertEqual(len(views), 5)
        for i, view in enumerate(views):
            expected = torch.full((1, 3, 384, 512), [-1., -.5, 0., .5, 1.][i])
            torch.testing.assert_close(view["img"], expected, rtol=0., atol=1e-6)
            self.assertEqual(view["instance"], f"heads_base_{'a'*12}_{i}")
            self.assertEqual(view["true_shape"].dtype, np.int32)

    def test_preprocessing_requires_registered_runtime_first(self):
        with patch("profiled_runtime.verify_runtime", side_effect=RuntimeError("runtime")):
            with self.assertRaisesRegex(RuntimeError, "runtime"):
                make_views(None, "a"*64)

    def pair_fixture(self):
        points = {p: (torch.ones(2, 3, 3), None) for p in PAIRS}
        confidence = {p: torch.full((2, 3), 2.) for p in PAIRS}
        return points, confidence

    def test_cpu_fixture_and_production_device_boundary(self):
        points, confidence = self.pair_fixture()
        validate_pairs(points, confidence, shape=(2, 3), device_type="cpu")
        with self.assertRaisesRegex(ValueError, "registered device"):
            validate_pairs(points, confidence, shape=(2, 3))

    def test_missing_pair_and_changed_dtype_rejected(self):
        points, confidence = self.pair_fixture()
        del points[PAIRS[0]]
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_pairs(points, confidence, shape=(2, 3), device_type="cpu")
        points, confidence = self.pair_fixture()
        confidence[PAIRS[0]] = confidence[PAIRS[0]].double()
        with self.assertRaisesRegex(ValueError, "float32"):
            validate_pairs(points, confidence, shape=(2, 3), device_type="cpu")

    def test_invalid_weight_nonfinite_and_grad_rejected(self):
        for mode in ("weight", "nan", "grad"):
            points, confidence = self.pair_fixture()
            if mode == "weight":
                confidence[PAIRS[0]][0, 0] = .999
            elif mode == "nan":
                points[PAIRS[0]][0][0, 0, 0] = float("nan")
            else:
                points[PAIRS[0]][0].requires_grad_(True)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_pairs(points, confidence, shape=(2, 3), device_type="cpu")

    def test_bfloat_features_digest_detects_content_dtype_and_order(self):
        feat = [(torch.ones(2, 3, dtype=torch.bfloat16), torch.zeros(2, 2), torch.tensor([2, 3]))]
        initial = tensor_sequence_digest(feat)
        self.assertEqual(initial, tensor_sequence_digest(feat))
        changed = [(feat[0][0].float(), feat[0][1], feat[0][2])]
        self.assertNotEqual(initial, tensor_sequence_digest(changed))
        feat[0][0][0, 0] = 2
        self.assertNotEqual(initial, tensor_sequence_digest(feat))

    def test_assign_prevalidates_all_factors_before_mutating(self):
        ops = TorchCaptureOps(None, None)
        named = [("a", torch.nn.Parameter(torch.zeros(2), requires_grad=False)),
                 ("b", torch.nn.Parameter(torch.zeros(2), requires_grad=False))]
        with self.assertRaisesRegex(ValueError, "Invalid final factor"):
            ops.assign(named, {"a": np.ones(2, dtype=np.float32), "b": np.ones(3, dtype=np.float32)})
        self.assertTrue(all(not bool(p.any()) for _, p in named))
        ops.assign(named, {n: np.ones(2, dtype=np.float32) for n, _ in named})
        self.assertTrue(all(bool((p == 1).all()) for _, p in named))


if __name__ == "__main__":
    unittest.main()
