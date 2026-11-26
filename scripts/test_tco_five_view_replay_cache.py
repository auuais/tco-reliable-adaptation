"""Check pairing, the exact margin and resource-only selection."""

import unittest

from replay_tco_five_view_replay_cache import ENDPOINTS, choose, grade, paired_bootstrap, quality


def resources(time, peak, fits=True, eligible=True, sites=1):
    return dict(eligible=eligible, peak_at_most_native=fits, sites=sites,
                mean_ratios=dict(scene_seconds=time, peak_allocated=peak))


class DecisionTests(unittest.TestCase):
    def test_margin_equality(self):
        self.assertEqual(grade(1.005, 0.99, 1.01), "pass")
        self.assertEqual(grade(1.005, 0.99, 1.010001), "inconclusive")
        self.assertEqual(grade(1.03, 1.010001, 1.1), "degradation")

    def test_resources_use_only_qualified_arms(self):
        table = {"last": resources(0.91, 0.8), "combined": resources(0.9, 0.91),
                 "uniform": resources(0.80, 0.9), "retention": resources(0.65, 1.15, fits=False)}
        self.assertEqual(choose(table, {"last", "combined", "retention"})[0], "combined")
        self.assertEqual(choose(table, set())[0], None)

    def test_bootstrap_keeps_arms_and_endpoints_paired(self):
        scene = [[(value, value * 10), (value, value * 15)] for value in (1, 5, 100)]
        out = paired_bootstrap([scene, scene], samples=100)
        self.assertEqual(out[ENDPOINTS[0]]["upper"], 1)
        self.assertEqual(out[ENDPOINTS[1]]["lower"], 1.5)

    def test_missing_pairs_are_inconclusive_on_each_dataset(self):
        scenes = [dict(dataset=dataset, scene="paired", width=518, frames=[1, 2, 3, 4, 5])
                  for dataset in ("7scenes", "eth3d")]
        jobs = [dict(**scene, seed=seed, policy=policy, cohort="confirmation", id=f"job_{i}_{seed}_{policy}")
                for i, scene in enumerate(scenes) for seed in (7, 77, 777) for policy in ("native", "last")]
        results = [dict(result=None if job["id"] == "job_1_777_last" else
                        dict(metrics={name: 1 for name in ENDPOINTS}, input_sha256="a", reference_sha256="b",
                             raster_shape=[5], scene_seconds=2, peak_allocated=3, peak_reserved=4)) for job in jobs]
        table = quality(jobs, results, "last", scenes)
        self.assertEqual(table["7scenes"]["verdict"], "pass")
        self.assertEqual(table["eth3d"]["verdict"], "inconclusive")
        self.assertEqual(table["eth3d"]["completed"], 5)


if __name__ == "__main__":
    unittest.main()
