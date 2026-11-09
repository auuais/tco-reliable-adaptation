"""Reject mismatched scientific provenance rather than quietly averaging it."""
import copy
import unittest
from aggregate_baseline_screen import validate_record
from run_baseline_screen import plan


class BaselineScreenValidation(unittest.TestCase):
    def setUp(self):
        self.plan = plan()
        self.job = next(j for j in self.plan["jobs"] if j["condition"] == "lr3e4")
        self.control = {k: "same" for k in ["research_commit", "input_sha256", "reference_sha256",
            "upstream_commit", "upstream_method_sha256", "versions", "gpu", "gpu_identity",
            "gsplat_backend", "checkpoint_download", "image_shape", "evaluation_tensor_sha256", "prior_metadata"]}
        self.control["configuration"] = dict(tco_steps=0, seed=42, tco_lr=.001, lambda_pose=1, lambda_intrinsics=.01)
        self.control["metrics"] = dict(accuracy_mean_m=.01, completeness_mean_m=.02, normal_consistency_mean=.8, protocol="same")
        self.record = copy.deepcopy(self.control)
        j = self.job
        self.record.update(status="complete", research_dirty=False, study_id=self.plan["study_id"], manifest_sha256="frozen",
            base_parameters_modified=[], runner_sha256=self.plan["sources"]["scripts/run_experiment.py"],
            prior_generator_sha256=self.plan["sources"]["scripts/prior_corruptions.py"],
            optimizer_learning_rates=[j["learning_rate"]], inference_and_adaptation_seconds=1,
            arguments=dict(scene=j["scene"], steps=j["steps"], frames=j["frames"], size=j["size"], seed=j["seed"],
                prior_profile=j["profile"], prior_weight=j["weight"], prior_seed=j["prior_seed"], learning_rate=j["learning_rate"]))
        self.record["configuration"].update(tco_steps=j["steps"], seed=j["seed"], tco_lr=j["learning_rate"])

    def check(self, record):
        validate_record(record, self.job, self.plan, "frozen", self.control)

    def test_correct_record_passes(self):
        self.check(self.record)

    def test_wrong_actual_optimizer_rate_rejected(self):
        self.record["optimizer_learning_rates"] = [.001]
        with self.assertRaises(AssertionError): self.check(self.record)

    def test_dirty_or_mismatched_evidence_rejected(self):
        for key, value in [("research_dirty", True), ("research_commit", "different"),
                           ("manifest_sha256", "different"), ("input_sha256", "different"),
                           ("base_parameters_modified", ["backbone.weight"]), ("status", "failed")]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.record); changed[key] = value
                with self.assertRaises(AssertionError): self.check(changed)

    def test_nonfinite_metric_rejected(self):
        self.record["metrics"]["accuracy_mean_m"] = float("nan")
        with self.assertRaises(AssertionError): self.check(self.record)

    def test_expected_complete_design(self):
        jobs = self.plan["jobs"]
        self.assertEqual(len(jobs), 39)
        self.assertEqual(len({j["run_id"] for j in jobs}), 39)
        self.assertEqual(sum(j["steps"] for j in jobs), 1440)
        self.assertEqual({j["seed"] for j in jobs if j["condition"] == "visual"}, {42, 123, 2026})


if __name__ == "__main__": unittest.main()
