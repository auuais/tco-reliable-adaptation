"""Exercise the full-40 replay's coverage, quality boundaries and OOM attribution."""

import copy
import unittest

from replay_tco_full40_recompute import analyze, endpoint


class Full40ReplayTest(unittest.TestCase):
    def setUp(self):
        self.plan = dict(candidate="c4_on", reference="c8_off", chunk_control="c4_off",
                         ten_policies=["c8_off", "c4_off", "c4_on"], jobs=[], bootstrap_samples=250)
        self.evidence = dict(runs=[], rental_compute_usd=0)

    def add_run(self, scene, cohort, policy, seed, errors=(100, 100), verdicts=("completed",), dataset="7scenes"):
        views = 10 if cohort == "ten" else 5
        key = f"{cohort}_{scene}_{policy}_{seed}"
        job = dict(id=key, dataset=dataset, scene=scene, frames=list(range(views)), seed=seed,
                   steps=40, cohort=cohort, policy=policy)
        self.plan["jobs"].append(job)
        attempts = [
            dict(attempt=f"attempt{index:02}", verdict=verdict, stopped=True, job=key,
                 exit_code=0 if verdict == "completed" else 1, log_sha256="f" * 64, rental_compute_usd=0,
                 wall_seconds=100)
            for index, verdict in enumerate(verdicts, 1)
        ]
        result = dict(
            **job, updates=40, rental_compute_usd=0, input_sha256="a" * 64, reference_sha256="b" * 64,
            output_sha256="c" * 64, raster_shape=[views, 3, 392 if dataset == "7scenes" else 350, 518],
            scene_seconds=25, peak_allocated=100, peak_reserved=120, cold_seconds=35, host_peak_bytes=300,
            losses=[1] * 39 + [.5], lora_tensors=384, head_taps=[4, 11, 17, 23],
            metrics=dict(accuracy_mean_m=errors[0], completeness_mean_m=errors[1], valid_points=20),
            runtime=dict(torch="2.7.1", cuda="12.8"),
        ) if verdicts[-1:] == ("completed",) else None
        run = dict(id=key, verdict=verdicts[-1] if verdicts else "unattempted", attempts=attempts, result=result)
        self.evidence["runs"].append(run)
        return run

    def compare(self, scene, candidate_values, reference_values, dataset="7scenes"):
        for index, seed in enumerate((7, 77, 777)):
            self.add_run(scene, "confirmation", "c8_off", seed, reference_values[index], dataset=dataset)
            self.add_run(scene, "confirmation", "c4_on", seed, candidate_values[index], dataset=dataset)

    def confirmation(self, dataset="7scenes"):
        return analyze(self.plan, self.evidence)["five_view"]["confirmation"][dataset]

    def test_both_metrics_and_unrounded_inclusive_margin(self):
        self.assertEqual(endpoint(1.01, 1.009, 1.01), "within_margin")
        self.assertEqual(endpoint(1.01, 1.005, 1.01000001), "inconclusive_margin")
        self.compare("a", [(101, 102)] * 3, [(100, 100)] * 3)
        result = self.confirmation()
        self.assertEqual(result["endpoints"], dict(accuracy_mean_m="within_margin",
                                                   completeness_mean_m="degradation_beyond_margin"))
        self.assertEqual(result["verdict"], "degradation_beyond_margin")

    def test_median_before_ratio_and_equal_scene_weight(self):
        self.compare("a", [(1, 1), (1, 1), (200, 200)], [(1, 1), (10, 10), (100, 100)])
        self.compare("b", [(192, 192)] * 3, [(100, 100)] * 3)
        result = self.confirmation()
        self.assertEqual(result["scenes"]["a"]["ratios"]["accuracy_mean_m"], .1)
        self.assertEqual(result["ratios"]["accuracy_mean_m"], 1.01)
        self.assertEqual(result["verdict"], "inconclusive_margin")
        self.assertEqual(result["bounds"]["leave_one_out"]["a"]["accuracy_mean_m"], 1.92)
        self.assertIn("upper_95", result["paired_scene_and_seed_sensitivity"])

    def test_missing_pair_prevents_pass(self):
        self.compare("a", [(100, 100)] * 3, [(100, 100)] * 3)
        self.evidence["runs"].pop()
        result = self.confirmation()
        self.assertEqual(result["verdict"], "inconclusive_missing_cases")
        self.assertEqual(result["missing_pairs"][0]["seed"], 777)

    def test_duplicate_invalid_updates_and_view_order_are_rejected(self):
        self.compare("a", [(100, 100)] * 3, [(100, 100)] * 3)
        self.evidence["runs"].append(copy.deepcopy(self.evidence["runs"][-1]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.confirmation()
        self.evidence["runs"].pop()
        row = self.evidence["runs"][-1]["result"]
        row["updates"] = 39
        with self.assertRaisesRegex(ValueError, "Wrong work"):
            self.confirmation()
        row["updates"] = 40
        row["frames"] = list(reversed(row["frames"]))
        with self.assertRaisesRegex(ValueError, "differs from registration"):
            self.confirmation()

    def test_ten_view_timeout_does_not_show_oom_necessity(self):
        for seed in (42, 43, 44):
            self.add_run("heads/seq-01", "ten", "c4_on", seed)
        self.add_run("heads/seq-01", "ten", "c8_off", 42, verdicts=("oom", "oom"))
        control = self.add_run("heads/seq-01", "ten", "c4_off", 42, verdicts=("oom", "timeout"))

        def assess():
            output = analyze(self.plan, self.evidence)["ten_view"]
            self.assertTrue(output["heads_fit"])
            self.assertFalse(output["three_window_fit"])
            return output["windows"]["heads/seq-01"]["attribution"]

        self.assertEqual(assess(), "inconclusive_comparator_coverage")
        control["attempts"][-1]["verdict"] = control["verdict"] = "oom"
        self.assertEqual(assess(), "recomputation_enabled_fit_among_tested_policies")
        self.add_run("heads/seq-01", "ten", "c4_off", 43)
        self.assertEqual(assess(), "chunk_only_fits")


if __name__ == "__main__":
    unittest.main()
