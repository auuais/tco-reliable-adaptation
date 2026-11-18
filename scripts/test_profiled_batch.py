"""Portable CPU-only safety fixtures for the local serial research launcher.

No Docker engine, model, dataset, checkpoint, GPU or NumPy is required.
Run: python scripts/test_profiled_batch.py
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import run_profiled_batch as batch


def job(arm="P0"):
    name = f"heads_earlier_{arm}"
    return dict(run_id=name, arm=arm, scene="heads", adaptation_source="earlier",
                adapt_lora=arm != "P0", optimizer_steps=0 if arm == "P0" else 8,
                raw_attempt_parent=f"reports/lora_profiled_v1/cases/{name}",
                public_attempt_parent=f"results/lora_profiled_v1/cases/{name}")


class SealedAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifests/execution.json"
        self.registry = dict(docker_image="sha256:fixture", checkpoint_sha256="checkpoint",
                             source_sha256_lf={"scripts/runner.py": "source"},
                             artifact_sha256={"inputs.json": "input"}, input_sha256={"inputs.json": "input"})
        batch.dump_new(self.manifest, self.registry)
        self.job = job()
        self.raw, self.public = batch.paths_for(self.root, self.job, "fixture01")

    def seal(self):
        self.raw.mkdir(parents=True)
        self.public.mkdir(parents=True)
        (self.raw / "predictions.npz").write_bytes(b"fixture-prediction-bytes")
        batch.dump_new(self.public / "started.json", {"fixture": True})
        record = dict(run_id=self.job["run_id"], attempt="fixture01", job=self.job, pass_all=True,
                      status="predictions_sealed_without_accuracy_scoring", original_state_unchanged=True,
                      accuracy_improvement_established=False, accuracy_scoring_performed=False,
                      optimizer_updates=0, manifest="manifests/execution.json", manifest_sha256=batch.sha(self.manifest),
                      original_state_sha256="original", pre_endpoint_original_state_sha256="original",
                      final_original_state_sha256="original",
                      raw_files={"predictions.npz": batch.sha(self.raw / "predictions.npz")},
                      public_files={"started.json": batch.sha(self.public / "started.json")}, **self.registry)
        batch.dump_new(self.public / "complete.json", record)
        return record

    def verify(self):
        return batch.verify_completed(self.root, self.manifest, self.registry, self.job, "fixture01")

    def test_unstarted_and_valid_seal(self):
        self.assertIsNone(self.verify())
        self.seal()
        self.assertEqual(self.verify()["raw_files_verified"], 1)

    def test_partial_or_failed_attempt_is_never_skipped(self):
        self.raw.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Partial/failed"):
            self.verify()

    def test_failure_marker_disqualifies_otherwise_valid_seal(self):
        self.seal()
        batch.dump_new(self.public / "failure.json", {"pass_all": False})
        with self.assertRaisesRegex(ValueError, "Partial/failed"):
            self.verify()

    def test_raw_tampering_is_detected_without_mtime_selection(self):
        self.seal()
        (self.raw / "predictions.npz").write_bytes(b"changed-predictions")
        with self.assertRaisesRegex(ValueError, "file hash differs"):
            self.verify()

    def test_public_tampering_is_detected(self):
        self.seal()
        (self.public / "started.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file hash differs"):
            self.verify()

    def test_extra_unsealed_file_is_rejected(self):
        self.seal()
        (self.raw / "unsealed.npy").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.verify()

    def test_altered_job_identity_is_rejected(self):
        record = self.seal()
        record["optimizer_updates"] = 8
        (self.public / "complete.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "optimizer_updates"):
            self.verify()

    def test_relative_file_cannot_escape_sealed_directory(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        self.raw.mkdir(parents=True)
        with self.assertRaises(ValueError):
            batch.relative_file(self.raw, "../../../../outside.txt")


class LauncherTests(unittest.TestCase):
    def test_argv_pins_local_engine_image_mounts_and_arithmetic(self):
        argv = batch.docker_argv(job("P1"), "fixture01", "manifests/execution.json", "sha256:fixture")
        self.assertEqual(argv[:3], batch.DOCKER)
        self.assertIn(f"type=bind,source={batch.HOST_REPO},target=/workspace", argv)
        self.assertIn(f"type=bind,source={batch.HOST_WEIGHTS},target=/weights,readonly", argv)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", argv)
        self.assertEqual(argv[argv.index("--image")+1], "sha256:fixture")
        self.assertEqual(argv[argv.index("--network")+1], "none")
        self.assertNotIn("--rm", argv)

    def test_stop_refuses_a_container_owned_by_another_queue(self):
        wrong = {"Config": {"Labels": {batch.LABEL: "someone-else"}}, "State": {"Running": True}}
        with patch.object(batch, "inspect_container", return_value=wrong), patch.object(batch, "command") as command:
            with self.assertRaisesRegex(ValueError, "ownership"):
                batch.stop_owned("other", "fixture01", "heads_earlier_P0")
            command.assert_not_called()

    def test_timeout_stops_only_own_container_and_records_real_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock()
            process.poll.return_value = None
            process.wait.return_value = 143
            stop = {"name": "owned", "running": False, "container_exit_code": 143}
            with patch.object(batch, "command", return_value=""), \
                    patch.object(batch.subprocess, "Popen", return_value=process), \
                    patch.object(batch.time, "monotonic", side_effect=[0., 2.]), \
                    patch.object(batch, "stop_owned", return_value=stop) as stopped:
                result = batch.run_container(["fixture"], Path(directory)/"stdout.log", "owned", "fixture01", "case", 1.)
            self.assertEqual(result["status"], "timed_out")
            self.assertEqual(result["docker_exit_code"], 143)
            stopped.assert_called_once_with("owned", "fixture01", "case")

    def test_docker_failure_before_creation_preserves_exitcode(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(returncode=125)
            process.poll.return_value = 125
            with patch.object(batch, "command", return_value=""), \
                    patch.object(batch.subprocess, "Popen", return_value=process), \
                    patch.object(batch, "inspect_optional", return_value=None):
                result = batch.run_container(["fixture"], Path(directory)/"stdout.log", "owned", "fixture01", "case", float("inf"))
            self.assertEqual(result["status"], "container_not_created")
            self.assertEqual(result["docker_exit_code"], 125)

    def test_zero_exit_without_scientific_seal_stops_before_next_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [job(), job("P1")]
            registry = {"docker_image": "sha256:fixture", "jobs": jobs}
            plan = {"jobs": [{"job": item, "container": item["run_id"], "argv": ["fixture"]} for item in jobs]}
            args = types.SimpleNamespace(attempt_prefix="fixture01", max_wall_hours=1., limit=2, min_free_gib=5.)
            checked = (root / "manifest.json", registry, plan, jobs, {}, {})
            fake_registry = types.ModuleType("register_profiled_execution")
            fake_registry.verify_registry = lambda *args: None

            def run(argv, log, *args):
                log.write_text("fixture stdout", encoding="utf-8")
                return {"status": "exited", "docker_exit_code": 0, "container_exit_code": 0, "oom_killed": False}

            with patch.dict("sys.modules", {"register_profiled_execution": fake_registry}), \
                    patch.object(batch, "ROOT", root), patch.object(batch, "build_plan", return_value=plan), \
                    patch.object(batch, "host_checks", return_value={}), patch.object(batch, "live_gpu_containers", return_value=[]), \
                    patch.object(batch, "verify_completed", return_value=None), \
                    patch.object(batch, "run_container", side_effect=run) as launched, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "without a sealed"):
                    batch.execute(args, checked)
            self.assertEqual(launched.call_count, 1)
            terminal_path, = root.rglob("terminal.json")
            self.assertEqual(batch.read(terminal_path)["status"], "failed")
            exit_path, = root.rglob("*.exit.json")
            self.assertEqual(batch.read(exit_path)["docker_exit_code"], 0)
            self.assertFalse((root / "reports/lora_profiled_v1/batch.lock").exists())

    def test_dry_run_never_executes_or_writes(self):
        selected = job()
        checked = (Path("manifest.json"), {"jobs": [selected]},
                   {"manifest": "manifest.json", "jobs": [{"job": selected, "argv": ["fixture"]}]},
                   [selected], {selected["run_id"]: None}, {})
        with patch("sys.argv", ["run_profiled_batch.py", "--attempt-prefix", "fixture01", "--dry-run"]), \
                patch.object(batch, "preflight", return_value=checked), patch.object(batch, "execute") as execute, \
                patch.object(batch, "dump_new") as write, contextlib.redirect_stdout(io.StringIO()) as output:
            batch.main()
        execute.assert_not_called()
        write.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertFalse(report["gpu_work_performed"])
        self.assertTrue(report["no_writes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
