"""Regression tests for predeclared TCO evidence decisions."""

from copy import deepcopy
import unittest

from replay_tco_five_view_replay_cache import grade, quality
from replay_tco_native_stalls_quality_capacity import (
    CAPACITY_SEEDS, POLICIES, capacity_verdict, verify_attempts, verify_model_result,
)


SHA = "a" * 64


def result_fields(job):
    views, width, steps = len(job["frames"]), job["width"], job["steps"]
    return dict(**job, updates=steps, native_depth_batches=[[5]] if views == 5 else [[8, 2]],
                lora_tensors=384, head_taps=[4, 11, 17, 23], depth_recompute_spans=[],
                camera_checkpoint=job["policy"] != "native", losses=[1.] * steps,
                optimizer=dict(step_calls=steps, factor_states=384, final_gradients=384,
                               counter_min=steps, counter_max=steps),
                raster_shape=[views, 3, 392, width],
                final_readout=dict(views=views, height=392, width=width, geometry_dtype="torch.float32"),
                input_sha256=SHA, reference_sha256=SHA, output_sha256=SHA,
                cuda_device_total=10 * 2**30, peak_allocated=8 * 2**30, peak_reserved=9 * 2**30,
                scene_seconds=60., runtime=dict(
                    gpu="NVIDIA GeForce RTX 3080", gpu_memory_fraction=0.875,
                    allocator="garbage_collection_threshold:0.75,expandable_segments:True",
                    torch="2.7.1+cu128", cuda="12.8",
                    gsplat_sha256="339c0272e544d6a166814cf49e2d21f500eaf9018e69585f1fcaadb5ed3c940e",
                ))


class ReplayTest(unittest.TestCase):
    def test_threshold_equality(self):
        self.assertEqual(grade(1.01, 0.99, 1.01), "pass")
        self.assertEqual(grade(1.01, 0.99, 1.01001), "inconclusive")
        self.assertEqual(grade(1.02, 1.01001, 1.08), "degradation")

    def test_missing_pair_keeps_dataset_inconclusive(self):
        jobs, rows, windows = [], [], []
        ratios = {"7scenes": 0.97, "eth3d": 1.07}
        for dataset, ratio in ratios.items():
            case = dict(dataset=dataset, scene="s", frames=list(range(5)), width=518, cohort="confirmation")
            windows.append(case)
            for seed in (7, 77, 777):
                for policy, value in (("native", 1.), ("camera", ratio)):
                    jobs.append(dict(case, id=f"{dataset}_{seed}_{policy}", seed=seed, policy=policy))
                    rows.append(dict(result=dict(
                        input_sha256=SHA, reference_sha256=SHA, raster_shape=[5, 3, 392, 518],
                        metrics=dict(accuracy_mean_m=value, completeness_mean_m=value),
                        scene_seconds=1., peak_allocated=1, peak_reserved=1,
                    )))
        separate = quality(jobs, rows, "camera", windows)
        self.assertEqual(separate["7scenes"]["verdict"], "pass")
        self.assertEqual(separate["eth3d"]["verdict"], "degradation")
        rows[-1]["result"] = None
        missing = quality(jobs, rows, "camera", windows)
        self.assertEqual(missing["7scenes"]["verdict"], "pass")
        self.assertEqual(missing["eth3d"]["verdict"], "inconclusive")
        self.assertEqual(missing["eth3d"]["completed"], 5)

    def test_final_loss_row_does_not_certify_adam_update(self):
        job = dict(id="case", policy="camera", steps=40, frames=list(range(10)), width=518)
        complete = result_fields(job)
        verify_model_result(job, complete)
        partial = deepcopy(complete)
        partial["optimizer"]["step_calls"] = 39
        with self.assertRaisesRegex(ValueError, "Changed Adam work"):
            verify_model_result(job, partial)
        wrong_batch = dict(complete, native_depth_batches=[[5]])
        with self.assertRaisesRegex(ValueError, "Wrong depth batches"):
            verify_model_result(job, wrong_batch)

    def test_timeout_is_distinct_from_oom(self):
        attempt = dict(job="case", attempt="attempt01", verdict="timeout", exit_code=None,
                       stopped=True, error="TimeoutExpired: exceeded 1200s", outer_seconds=1200,
                       wall_seconds=1203., log_sha256=SHA, complete_sha256=None, rental_compute_usd=0)
        record = dict(id="case", attempts=[attempt], verdict="timeout", result=None)
        verify_attempts(dict(id="case"), record, 2)
        attempt.update(verdict="oom", exit_code=1)
        with self.assertRaisesRegex(ValueError, "Missing OOM evidence"):
            verify_attempts(dict(id="case"), dict(record, verdict="oom"), 2)

    def test_best_pilot_and_recovered_oom(self):
        records, jobs = {}, {}

        def add(key, complete, oom=False):
            rows = [{"verdict": "oom"}] if oom else []
            if complete:
                rows.append({"verdict": "completed"})
            records[key] = dict(id=key, verdict=rows[-1]["verdict"] if rows else "gated",
                                result={} if complete else None, attempts=rows)

        qualified = dict.fromkeys(POLICIES, True)
        for method in POLICIES:
            for index in (1, 2):
                add(f"screen_{method}_{index}", True)
            add(f"pilot_{method}", method in {"n", "ng"}, oom=method == "ng")
        for scene, seeds in CAPACITY_SEEDS.items():
            for seed in seeds:
                identifier = f"ten_{scene}_{seed}_n"
                jobs[identifier] = dict(id=identifier, cohort="capacity", policy="n", scene=scene, seed=seed,
                                        steps=40)
                add(identifier, True)
        result = capacity_verdict(jobs, records, qualified, dict(candidate="n", ids=list(jobs)))
        self.assertEqual(result["completed"], 9)
        self.assertEqual(result["verdict"], "replicated_capacity")
        add(next(iter(jobs)), True, oom=True)
        self.assertEqual(capacity_verdict(jobs, records, qualified)["verdict"], "intermittent_fit")


if __name__ == "__main__":
    unittest.main()
