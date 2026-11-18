"""No Docker calls: verify the constructed local, CPU-only launch boundary."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_profiled_d0_local as launcher


class LocalD0Tests(unittest.TestCase):
    def test_cpu_local_networkless_readonly_weights_and_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=launcher.create_argv(root,root,"tco-m3-d0-fixture01",["python","fixture.py"])
        self.assertEqual(args[:3],["docker","--host",launcher.ENGINE])
        self.assertNotIn("--gpus",args)
        self.assertIn("CUDA_VISIBLE_DEVICES=",args)
        self.assertIn("PROFILED_D0_IMAGE_ID="+launcher.IMAGE,args)
        self.assertEqual(args[args.index("--network")+1],"none")
        self.assertTrue(any(x.endswith("target=/weights,readonly") for x in args))
        self.assertEqual(args[-3:],[launcher.IMAGE,"python","fixture.py"])
        self.assertLessEqual(launcher.OUTER_SECONDS,1900)

    def test_rejects_unsafe_or_unrelated_container_names(self):
        for name in ("hbbs","../../escape","tco-m3-d0-","tco-m3-d0-x;bad","tco-m3-d0-"+"a"*66):
            with self.assertRaises(ValueError):
                launcher.create_argv(Path.cwd(),Path.cwd(),name,[])

    def test_attach_interruption_still_stops_owned_container(self):
        created=dict(Image=launcher.IMAGE,HostConfig=dict(DeviceRequests=None,NetworkMode="none",Memory=6*1024**3,NanoCpus=2*10**9))
        running=dict(Image=launcher.IMAGE,State=dict(Running=True,ExitCode=0))
        stopped=dict(Image=launcher.IMAGE,State=dict(Running=False,ExitCode=137))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(launcher,"docker_json",side_effect=[[dict(Id=launcher.IMAGE)],[created],[running],[stopped]]), \
                 patch.object(launcher.subprocess,"check_output",return_value="owned-container\n"), \
                 patch.object(launcher.subprocess,"run",side_effect=[KeyboardInterrupt(),subprocess.CompletedProcess([],0)]) as run, \
                 patch("builtins.print"):
                with self.assertRaises(RuntimeError):
                    launcher.launch(root,root,"tco-m3-d0-interrupted01",["python","fixture.py"])
            self.assertIn("stop",run.call_args_list[1].args[0])
            receipt=json.loads((root/"reports/lora_profiled_d0_v1/launches/tco-m3-d0-interrupted01/terminal.json").read_text())
            self.assertTrue(receipt["stopped_verified"])
            self.assertEqual(receipt["error"]["type"],"KeyboardInterrupt")

    def test_stop_timeout_falls_back_to_kill_and_verify(self):
        running=dict(Image=launcher.IMAGE,State=dict(Running=True,ExitCode=0))
        stopped=dict(Image=launcher.IMAGE,State=dict(Running=False,ExitCode=137))
        with patch.object(launcher,"docker_json",side_effect=[[running],[running],[stopped]]), \
             patch.object(launcher.subprocess,"run",side_effect=[subprocess.TimeoutExpired("stop",20),subprocess.CompletedProcess([],0)]) as run:
            state,errors=launcher.ensure_stopped("owned-container")
        self.assertFalse(state["State"]["Running"])
        self.assertEqual(errors[0]["type"],"TimeoutExpired")
        self.assertIn("kill",run.call_args_list[1].args[0])

    def test_unverified_shutdown_remains_unverified(self):
        with patch.object(launcher,"docker_json",side_effect=OSError("daemon unavailable")), \
             patch.object(launcher.subprocess,"run",side_effect=OSError("daemon unavailable")):
            state,errors=launcher.ensure_stopped("owned-container")
        self.assertIsNone(state)
        self.assertTrue(errors)


if __name__=="__main__":
    unittest.main()
