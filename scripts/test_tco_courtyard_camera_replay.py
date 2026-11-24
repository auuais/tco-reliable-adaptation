"""Exercise the decision rules without GPU or reconstruction dependencies."""

import unittest

from replay_tco_courtyard_camera import analyze_locked, screen_verdict
from tco_courtyard_camera_protocol import CAMERA, MARGIN, REFERENCE, locked_jobs, window_jobs, FIVE_ETH3D


class CameraReplayTests(unittest.TestCase):
    def setUp(self):
        windows = window_jobs({scene: [f"{idx}.JPG" for idx in range(5)] for scene in FIVE_ETH3D})
        self.plan = dict(jobs=locked_jobs(windows), bootstrap_seed=20261125, bootstrap_samples=10_000,
                         five_only=True)
        self.rows = {}
        for job in self.plan["jobs"]:
            if job["cohort"] == "ten":
                continue
            error = 1.0
            if job["policy"] == CAMERA:
                error = 8.0 if job["scene"] == "courtyard" else MARGIN
            self.rows[job["id"]] = dict(
                **job, metrics=dict(accuracy_mean_m=error, completeness_mean_m=error),
                scene_seconds=1, peak_allocated=1, peak_reserved=1, cold_seconds=1, host_peak_bytes=1)

    def analyze(self):
        return analyze_locked(self.plan, {}, self.rows, {"gate": "repeatable_two_update_oom"})

    def test_exact_margin_passes_while_courtyard_remains_diagnostic(self):
        decision = self.analyze()
        self.assertTrue(decision["five_view_supported"])
        self.assertEqual(decision["primary"]["eth3d"]["ratios"]["accuracy_mean_m"], MARGIN)
        self.assertEqual(decision["courtyard"]["scenes"]["courtyard"]["ratios"]["accuracy_mean_m"], 8)

    def test_separate_completeness_endpoint_can_fail(self):
        for row in self.rows.values():
            if row["policy"] == CAMERA and row["dataset"] == "7scenes":
                row["metrics"]["completeness_mean_m"] = 1.011
        report = self.analyze()
        self.assertFalse(report["five_view_supported"])
        self.assertEqual(report["primary"]["7scenes"]["endpoints"],
                         dict(accuracy_mean_m="within_margin", completeness_mean_m="degradation_beyond_margin"))

    def test_missing_reference_forces_inconclusive(self):
        key = next(key for key, row in self.rows.items()
                   if row["policy"] == REFERENCE and row["cohort"] == "confirmation" and row["dataset"] == "eth3d")
        del self.rows[key]
        self.assertEqual(self.analyze()["primary"]["eth3d"]["verdict"], "inconclusive_missing_cases")

    def test_backward_oom_blocks_promotion_even_after_forward(self):
        row = dict(verdict="oom", attempts=[dict(verdict="oom", phase="backward")] * 2)
        pilot = dict(verdict="unattempted", attempts=[])
        self.assertEqual(screen_verdict(dict(gate_camera=row, promote_camera=pilot)), "repeatable_two_update_oom")
        row["attempts"][1] = dict(verdict="timeout")
        self.assertEqual(screen_verdict(dict(gate_camera=row, promote_camera=pilot)), "ten_view_fit_unverified")


if __name__ == "__main__":
    unittest.main()
