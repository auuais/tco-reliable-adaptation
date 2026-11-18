"""D0 registration, immutability and independent-arithmetic boundary fixtures."""
import copy
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_profiled_d0 as runner


class D0ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in runner.SOURCES:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        runner.write_new(self.root/runner.DESIGN, {"fixture": True})
        self.checks = "results/synthetic.json"
        runner.write_new(self.root/self.checks, dict(pass_all=True, tests_run=10,
            source_sha256_lf=runner.source_hashes(self.root), runtime=runner.EXPECTED_RUNTIME))

    def prepare(self):
        with patch.object(runner, "runtime_identity", return_value=runner.EXPECTED_RUNTIME), \
             patch("profiled_d0_artifacts.build_inventory", return_value={"synthetic_only": True}):
            return runner.prepare(self.root, self.root, runner.REGISTRY, self.checks)

    def verify(self, commit=False):
        with patch.object(runner, "runtime_identity", return_value=runner.EXPECTED_RUNTIME), \
             patch("profiled_d0_artifacts.verify_inventory") as check:
            result = runner.verify_registration(self.root, self.root, runner.REGISTRY, require_commit=commit)
            check.assert_called_once()
            return result

    def test_prepare_is_exclusive_and_records_no_decode(self):
        registration = self.prepare()
        self.assertFalse(registration["registration_preparation_decoded_tensors"])
        self.assertTrue(registration["committed_registration_required_before_decode"])
        with self.assertRaisesRegex(ValueError, "Preserve"):
            self.prepare()
        self.verify()

    def test_changed_tested_code_blocks_inventory_preparation(self):
        (self.root/runner.SOURCES[0]).write_text("# changed\n", encoding="utf-8")
        with patch.object(runner, "runtime_identity", return_value=runner.EXPECTED_RUNTIME), \
             patch("profiled_d0_artifacts.build_inventory") as inventory:
            with self.assertRaisesRegex(ValueError, "different source"):
                runner.prepare(self.root, self.root, runner.REGISTRY, self.checks)
            inventory.assert_not_called()

    def test_changed_registration_dependencies_rejected(self):
        self.prepare()
        for name in (runner.DESIGN, self.checks, runner.SOURCES[0]):
            path = self.root/name
            old = path.read_bytes()
            path.write_bytes(old+b"\n")
            with self.assertRaises(ValueError):
                self.verify()
            path.write_bytes(old)

    def test_resources_cannot_be_silently_expanded(self):
        registration = self.prepare()
        registration["resources"]["model_forward_calls"] = 1
        (self.root/runner.REGISTRY).write_text(json.dumps(registration), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resource"):
            self.verify()

    def test_git_committed_registration_and_sources_required(self):
        self.prepare()
        command = ["git", "-C", str(self.root)]
        subprocess.check_call(command+["init", "-q"], stdout=subprocess.DEVNULL)
        subprocess.check_call(command+["config", "core.autocrlf", "false"])
        subprocess.check_call(command+["add", "."])
        subprocess.check_call(command+["-c", "user.name=D0 Fixture", "-c", "user.email=d0@example.invalid",
                                        "commit", "-qm", "synthetic fixture"])
        self.verify(commit=True)
        path = self.root/runner.SOURCES[0]
        path.write_text("# uncommitted\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            runner.committed_identity(self.root, runner.REGISTRY, runner.SOURCES)
        path.write_text("# fixture\n", encoding="utf-8")
        path = self.root/runner.REGISTRY
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaises(ValueError):
            runner.committed_identity(self.root, runner.REGISTRY, runner.SOURCES)

    def test_invalid_paths_and_overwrite_are_rejected(self):
        for value in ("../escape", "a/b", "", "C:/escape", "x"*65):
            with self.assertRaises(ValueError):
                runner.attempt_path(self.root, value)
        output = self.root/"fresh.json"
        runner.write_new(output, {"ok": True})
        with self.assertRaises(FileExistsError):
            runner.write_new(output, {"ok": False})
        self.assertEqual(json.loads(output.read_text()), {"ok": True})

    def test_json_refuses_nonfinite_results(self):
        with self.assertRaises(ValueError):
            runner.write_new(self.root/"bad.json", {"value": float("nan")})
        self.assertFalse((self.root/"bad.json").exists())

    def test_direct_ba_independent_check_and_wrong_scale(self):
        from profiled_d0_math import effective_weight_stats
        rng = np.random.default_rng(13)
        factors = {"A": rng.normal(size=(4,7)), "B": rng.normal(size=(9,4))}
        gram = effective_weight_stats(factors["A"], factors["B"], 1.0)["update_frobenius"]
        self.assertTrue(runner._independent_weight_check(factors, gram)["pass_all"])
        with self.assertRaisesRegex(ValueError, "differs"):
            runner._independent_weight_check(factors, gram*4)

    def test_independent_function_scalar_and_shape(self):
        from profiled_d0_math import compare_depths
        base = np.arange(1,13,dtype=np.float64).reshape(3,4)
        change = np.cos(base)
        change -= np.sum(base*change)/np.sum(base*base)*base
        candidate = 1.1*base+change
        xy = np.array([(x,y) for y in range(3) for x in range(4)],dtype=np.int64)
        measured = compare_depths(base,candidate,xy)["scalar_coefficient"]
        self.assertTrue(runner._independent_function_check(base,candidate,xy,measured)["pass_all"])
        with self.assertRaisesRegex(ValueError, "differs"):
            runner._independent_function_check(base,candidate,xy,measured+0.1)
        candidate[0,0] = np.nan
        self.assertIsNone(runner._independent_function_check(base,candidate,xy,None)["pass_all"])

    @unittest.skipUnless(platform.system()=="Linux", "Production uses Linux fcntl lock")
    def test_active_os_lock_and_release(self):
        first = runner.acquire_lock(self.root)
        try:
            with self.assertRaises(BlockingIOError):
                runner.acquire_lock(self.root)
        finally:
            first.close()
        second = runner.acquire_lock(self.root)
        second.close()

    @unittest.skipUnless(platform.system()=="Linux", "Production uses Linux resource/signal")
    def test_failure_retains_partial_measurements(self):
        registration = self.prepare()
        def fail(root, weights, inventory, deadline, progress, emit):
            emit("adapter_00.json", {"synthetic_statistic": 3.0})
            raise ValueError("deliberate fixture failure")
        with patch.object(runner, "verify_registration", return_value=(registration,"fixture")), \
             patch.object(runner, "compute", side_effect=fail):
            with self.assertRaisesRegex(ValueError,"fixture failure"):
                runner.execute(self.root,self.root,runner.REGISTRY,"partial01")
        result = runner.attempt_path(self.root,"partial01")
        self.assertEqual(json.loads((result/"partial/adapter_00.json").read_text())["data"],
                         {"synthetic_statistic":3.0})
        self.assertEqual(json.loads((result/"failure.json").read_text())["status"],"failed")
        self.assertFalse((result/"report.json").exists())

    @unittest.skipUnless(platform.system()=="Linux", "Production uses Linux resource/signal")
    def test_lost_creation_race_does_not_write_into_other_attempt(self):
        registration = self.prepare()
        destination = runner.attempt_path(self.root,"race01")
        def race(*args,**kwargs):
            destination.mkdir(parents=True)
            runner.write_new(destination/"owner.json", {"owner":"other"})
            return registration,"fixture"
        with patch.object(runner,"verify_registration",side_effect=race), \
             patch.object(runner,"compute") as compute:
            with self.assertRaises(FileExistsError):
                runner.execute(self.root,self.root,runner.REGISTRY,"race01")
            compute.assert_not_called()
        self.assertEqual(sorted(p.name for p in destination.iterdir()),["owner.json"])

    @unittest.skipUnless(platform.system()=="Linux", "Production uses Linux resource/signal")
    def test_failed_verification_preserves_original_and_fresh_failure(self):
        registration = self.prepare()
        destination=runner.attempt_path(self.root,"done01")
        runner.write_new(destination/"report.json", {"existing":"untouched"})
        before=(destination/"report.json").read_bytes()
        with patch.object(runner,"verify_registration",return_value=(registration,"fixture")), \
             patch.object(runner,"compute",side_effect=ValueError("verification fixture failure")):
            with self.assertRaisesRegex(ValueError,"fixture failure"):
                runner.execute(self.root,self.root,runner.REGISTRY,"done01",verification_output="results/fresh_verify.json")
        self.assertEqual((destination/"report.json").read_bytes(),before)
        self.assertFalse(json.loads((self.root/"results/fresh_verify.json").read_text())["pass_all"])


if __name__ == "__main__":
    unittest.main()
