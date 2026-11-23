"""Check scene weighting, quality margins and the equal-time comparator."""

import unittest

from replay_tco_one_percent import FIELDS, matched_control, normalize_scenes, paired_bootstrap, passes_quality_challenge


class OnePercentReplayTest(unittest.TestCase):
    def test_rooms_receive_equal_weight_after_sequence_medians(self):
        rooms = {"wide/seq1": "wide", "wide/seq2": "wide", "tight/seq1": "tight"}
        jobs = []
        rows = {}
        for scene in rooms:
            for arm, steps in (("four", 40), ("cached12", 10)):
                for seed in range(3):
                    key = f"{scene}/{arm}/{seed}"
                    jobs.append(dict(id=key, scene=scene, arm=arm, steps=steps))
                    error = 1 if arm == "four" else (0.96 if scene.startswith("wide") else 1.04)
                    rows[key] = dict(id=key, scene_seconds=1, peak_allocated=1, cold_seconds=1,
                                     peak_reserved=1, host_peak_bytes=1,
                                     metrics=dict(accuracy_mean_m=error, completeness_mean_m=error))
        output = normalize_scenes(jobs, rows, rooms)["cached12_10"]
        self.assertEqual(output["ratios"]["accuracy_mean_m"], 1)
        self.assertEqual(output["rooms_over_1percent"]["accuracy_mean_m"], 1)
        del rows["tight/seq1/cached12/2"]
        self.assertIsNone(normalize_scenes(jobs, rows, rooms)["cached12_10"]["ratios"])

    def test_equal_time_quality_challenge_checks_both_errors(self):
        values = {"four_5": (.35, 1.04, 1.05), "four_10": (.43, .98, 1.02),
                  "cached12_10": (.4, 1.0, 1.07), "zero_0": (.1, .99, 1)}
        stats = {key: {"ratios": dict(zip(FIELDS, (time, 1, accuracy, completeness)))}
                 for key, (time, accuracy, completeness) in values.items()}
        self.assertEqual(matched_control(stats, "cached12_10"), "four_5")
        self.assertFalse(passes_quality_challenge(stats["cached12_10"]["ratios"], stats["four_5"]["ratios"]))

    def test_bootstrap_resamples_whole_rooms(self):
        result = paired_bootstrap({"a": {"metric": .99}, "b": {"metric": 1.02}}, 20261123, 10000)
        self.assertEqual(result["upper_95"]["metric"], 1.02)
        self.assertEqual(result["leave_one_out"]["a"]["metric"], 1.02)


if __name__ == "__main__":
    unittest.main()
