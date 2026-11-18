"""Fixtures for sealed D1 replay, teacher acceptance and Docker resource gates."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from profiled_d1_analysis import CONTRASTS, STATES, analyze_matrices, key
from profiled_d1_capture import counted_method
from run_profiled_d0 import sha
from run_profiled_d1 import exact_raster, load_sealed_arrays, save_arrays
from run_profiled_d1_local import IMAGE, LABEL, MEMORY, inspect_matches


class ControllerTests(unittest.TestCase):
    def test_exact_teacher_raster_detects_single_ulp(self):
        expected = np.ones((5, 3, 4), dtype=np.float32)
        actual = expected.copy()
        self.assertTrue(exact_raster(expected, actual)["equal"])
        actual[2, 0, 1] = np.nextafter(np.float32(1), np.float32(2))
        result = exact_raster(expected, actual)
        self.assertFalse(result["equal"])
        self.assertEqual(result["differing_pixels"], 1)

    def test_only_sealed_arrays_can_be_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sealed.npz"
            save_arrays(path, {"A": np.eye(5)})
            digest = sha(path)
            np.testing.assert_array_equal(load_sealed_arrays(path, digest)["A"], np.eye(5))
            with path.open("ab") as stream:
                stream.write(b"altered")
            with self.assertRaisesRegex(ValueError, "modified"):
                load_sealed_arrays(path, digest)
            with self.assertRaises(FileExistsError):
                save_arrays(path, {})

    def test_every_fold_and_contrast_retained_and_targets_immutable(self):
        matrices = {}
        for fold in (0, 1):
            for index, state in enumerate(STATES):
                values = dict(A=np.eye(5)*(1+index*.01), b=np.ones(5),
                              Q=np.eye(5)*(1+fold*.1+index*.03),
                              d=np.ones(5), ridge=np.array(.1))
                matrices.update({key(fold, state, name): value for name, value in values.items()})
        rows = analyze_matrices(matrices)
        self.assertEqual([(r["fold"], r["contrast"]) for r in rows],
                         [(fold, f"{a}_to_{b}") for fold in (0, 1) for a, b in CONTRASTS])
        self.assertTrue(all(r["response"]["pass_all"] for r in rows))
        self.assertEqual(sum(r["response"]["derivative"] is not None for r in rows), 2)
        matrices[key(1, "F1", "d")] = np.zeros(5)
        with self.assertRaisesRegex(ValueError, "remain fixed"):
            analyze_matrices(matrices)

    def test_actual_invocation_recorder_survives_forward_failure(self):
        class Model:
            def decode(self):
                raise RuntimeError("fixture decoder failed")
        model = Model()
        events = []
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with counted_method(model, "decode", 20, lambda: None, events.append):
                model.decode()
        self.assertEqual(events, [1])
        self.assertNotIn("decode", model.__dict__)

    def test_local_inspection_rejects_wrong_owner_device_and_caps(self):
        cpu = dict(Image=IMAGE, Config={"Labels": {LABEL: "ours"}},
                   HostConfig=dict(NetworkMode="none", Memory=MEMORY, MemorySwap=MEMORY,
                                   DeviceRequests=None))
        self.assertTrue(inspect_matches(cpu, "ours", "prepare"))
        self.assertFalse(inspect_matches(cpu, "other", "prepare"))
        self.assertFalse(inspect_matches(cpu, "ours", "model"))
        gpu = copy.deepcopy(cpu)
        gpu["HostConfig"]["DeviceRequests"] = [{"Capabilities": [["gpu"]]}]
        self.assertTrue(inspect_matches(gpu, "ours", "model"))
        self.assertFalse(inspect_matches(gpu, "ours", "replay"))
        gpu["HostConfig"]["Memory"] += 1
        self.assertFalse(inspect_matches(gpu, "ours", "model"))


if __name__ == "__main__":
    unittest.main()
