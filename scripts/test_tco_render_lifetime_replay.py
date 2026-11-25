"""Regression tests for offline render-lifetime decisions."""

import unittest

from replay_tco_render_lifetime import (
    FIVE_ETH, FIVE_SEVEN, MARGIN, PROMOTION_ORDER, SEEDS, TEN_SEEDS,
    analyze_capacity, analyze_five, check_depth_batches, classify, derive_screen, is_qualified,
    validate_locked_roster,
)


def record(*statuses):
    return dict(verdict=statuses[-1] if statuses else "unattempted",
                attempts=[dict(verdict=status, wall_seconds=90) for status in statuses])


def row(job, error):
    return dict(**job, metrics=dict(accuracy_mean_m=error, completeness_mean_m=error, valid_points=30),
                scene_seconds=1, peak_allocated=1, peak_reserved=1, cold_seconds=1, host_peak_bytes=1)


def case(cohort, dataset, scene, seed, frames, policy):
    identifier = "_".join((cohort, dataset, scene, str(seed), policy))
    return dict(id=identifier, cohort=cohort, dataset=dataset, scene=scene, seed=seed, frames=list(frames),
                policy=policy, width=518, steps=40, profile=False, score=True)


class RenderLifetimeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate = "n"
        control = "native"
        cls.plan = dict(candidate=candidate, jobs=[
            case("ten", "7scenes", scene, seed, range(0, 1000, 100), policy)
            for scene, seeds in TEN_SEEDS.items() for seed in seeds for policy in (candidate, control, "camera")
        ] + [
            case("confirmation", dataset, scene, seed, range(start, start + 5 * stride, stride), policy)
            for dataset, windows in (("7scenes", FIVE_SEVEN), ("eth3d", FIVE_ETH))
            for scene, (start, stride) in windows.items() for seed in SEEDS for policy in (candidate, control)
        ])

    def test_fixed_adjoint_without_native_render_derivative_cannot_qualify(self):
        details = dict(qualified=True, fixed_passed=True, real_update_passed=True)
        self.assertFalse(is_qualified(details))
        details["real_renderer_passed"] = True
        self.assertTrue(is_qualified(details))
        details["real_update_passed"] = False
        self.assertFalse(is_qualified(details))

    def test_repeatable_second_update_oom_blocks_promotion(self):
        plan, attempts = dict(jobs=[]), {}
        for policy in PROMOTION_ORDER:
            attempts[f"pilot_{policy}"] = record()
            for replica in (1, 2):
                key = f"screen_{policy}_{replica}"
                plan["jobs"].append(dict(id=key, steps=2, cohort="screen", policy=policy))
                attempts[key] = record("oom") if policy == "n" else record()
        quality = {"n": dict(qualified=True, real_renderer_passed=True, real_update_passed=True)}
        screen = derive_screen(plan, attempts, quality)
        self.assertEqual(screen["arms"]["n"]["gate"], "repeatable_oom")
        self.assertIsNone(screen["promoted_policy"])
        attempts["pilot_n"] = record("completed")
        with self.assertRaisesRegex(ValueError, "bypassed"):
            derive_screen(plan, attempts, quality)

    def test_timeout_is_never_an_oom_and_mixed_fit_is_unstable(self):
        self.assertEqual(classify(record("timeout", "timeout")), "inconclusive")
        self.assertEqual(classify(record("oom", "completed")), "unstable")
        self.assertEqual(classify(record("error", "completed")), "completed")

    def test_native_depth_batches_must_survive(self):
        job = dict(id="native_batches", frames=list(range(10)), policy="ngd")
        check_depth_batches(job, dict(native_depth_batches=[[8, 2]], camera_checkpoint=True))
        with self.assertRaisesRegex(ValueError, "Depth batches"):
            check_depth_batches(job, dict(native_depth_batches=[[1] * 10], camera_checkpoint=True))

    def test_capacity_requires_nine_stable_completions(self):
        validate_locked_roster(self.plan)
        cases = [job for job in self.plan["jobs"] if job["cohort"] == "ten"]
        attempts = {job["id"]: record("completed") if job["policy"] == "n" else record("oom", "oom") for job in cases}
        rows = {job["id"]: row(job, 1.0) for job in cases if job["policy"] == "n"}
        answer = analyze_capacity(self.plan, attempts, rows)
        self.assertTrue(answer["complete"] and answer["native_comparator_supported"])
        self.assertEqual(answer["valid_completions"], 9)
        unstable = next(iter(rows))
        attempts[unstable] = record("oom", "completed")
        answer = analyze_capacity(self.plan, attempts, rows)
        self.assertFalse(answer["complete"])
        self.assertEqual(answer["valid_completions"], 8)

    def test_closed_capacity_blocks_confirmation(self):
        attempts = {job["id"]: record() for job in self.plan["jobs"]}
        self.assertEqual(analyze_five(self.plan, attempts, {}, {"complete": False}), {})
        key = next(job["id"] for job in self.plan["jobs"] if job["cohort"] == "confirmation")
        attempts[key] = record("timeout")
        with self.assertRaisesRegex(ValueError, "capacity gate"):
            analyze_five(self.plan, attempts, {}, {"complete": False})

    def test_exact_margin_and_missing_pair(self):
        cases = [job for job in self.plan["jobs"] if job["cohort"] == "confirmation"]
        attempts = {job["id"]: record("completed") for job in cases}
        rows = {job["id"]: row(job, MARGIN if job["policy"] == "n" else 1.0) for job in cases}
        paired = analyze_five(self.plan, attempts, rows, {"complete": True})
        for dataset in paired.values():
            self.assertEqual(dataset["verdict"], "within_margin")
            self.assertEqual(dataset["ratios"]["accuracy_mean_m"], MARGIN)
        missing = next(job for job in cases if job["dataset"] == "eth3d")
        del rows[missing["id"]]
        attempts[missing["id"]] = record("timeout")
        paired = analyze_five(self.plan, attempts, rows, {"complete": True})
        self.assertEqual(paired["eth3d"]["verdict"], "inconclusive_missing_cases")


if __name__ == "__main__":
    unittest.main()
