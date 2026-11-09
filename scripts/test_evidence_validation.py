"""Negative controls for experiment reporting; fixtures stay in temporary folders."""
import copy
import json
import pathlib
import tempfile
import unittest

from aggregate_midterm import validated_pairs


class EvidenceValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.manifest = {"test/seq-01": [0, 200, 400, 600, 800]}
        self.base = {"status": "complete", "arguments": {"scene": "test/seq-01", "frames": self.manifest["test/seq-01"], "size": 518, "steps": 0, "seed": 42},
                     "base_parameters_modified": [], "input_sha256": {"rgb": "same"}, "reference_sha256": {"pose": "same"},
                     "upstream_commit": "test-source", "upstream_method_sha256": {"method": "same"}, "gpu": "TEST GPU",
                     "versions": {"torch": "test"}, "prior_provenance": "test", "evaluation_only": "test",
                     "checkpoint_download": {"sha256": "same"}, "configuration": {"tco_steps": 0, "lr": 0.001},
                     "metrics": {"accuracy_mean_m": 1.0, "completeness_mean_m": 1.0, "normal_consistency_mean": 0.5, "protocol": "test-only"}}
        self.adapted = copy.deepcopy(self.base)
        self.adapted["arguments"]["steps"] = self.adapted["configuration"]["tco_steps"] = 40

    def tearDown(self):
        self.temp.cleanup()

    def save(self, record):
        steps = record["arguments"]["steps"]
        folder = self.root / f"cloud_test_seq-01_518_s{steps}_seed42"
        folder.mkdir(exist_ok=True)
        (folder / "run.json").write_text(json.dumps(record))

    def test_missing_is_not_zero(self):
        pairs, missing = validated_pairs(self.root, self.manifest)
        self.assertEqual(pairs, [])
        self.assertEqual(len(missing), 2)

    def test_complete_pair(self):
        self.save(self.base)
        self.save(self.adapted)
        pairs, missing = validated_pairs(self.root, self.manifest)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(missing, [])

    def test_timeout_excluded(self):
        self.save(self.base)
        self.adapted["status"] = "aborted_hardware_timeout"
        self.save(self.adapted)
        pairs, missing = validated_pairs(self.root, self.manifest)
        self.assertEqual(pairs, [])
        self.assertEqual(missing[0]["status"], "aborted_hardware_timeout")

    def test_mismatched_inputs_and_protocol_rejected(self):
        for field, value in [("reference_sha256", {"pose": "different"}),
                             ("gpu", "DIFFERENT TEST GPU"),
                             ("gsplat_backend", {"sha256": "different"}),
                             ("base_parameters_modified", ["weight"]),
                             ("configuration", {"tco_steps": 40, "lr": 0.01})]:
            with self.subTest(field=field):
                self.save(self.base)
                changed = copy.deepcopy(self.adapted)
                changed[field] = value
                self.save(changed)
                with self.assertRaises(ValueError):
                    validated_pairs(self.root, self.manifest)

    def test_wrong_seed_and_nonfinite_metric_rejected(self):
        self.save(self.base)
        self.adapted["arguments"]["seed"] = 7
        self.save(self.adapted)
        with self.assertRaises(ValueError):
            validated_pairs(self.root, self.manifest)
        self.adapted["arguments"]["seed"] = 42
        self.adapted["metrics"]["accuracy_mean_m"] = float("nan")
        self.save(self.adapted)
        with self.assertRaises(ValueError):
            validated_pairs(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
