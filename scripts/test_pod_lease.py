import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pod_lease import pod_request, reason_to_stop
from weekly_pod_guard import watch_pod, weekly_window


class LeaseTests(unittest.TestCase):
    def test_expiry_and_concurrent_deadline(self):
        self.assertIsNone(reason_to_stop(119, 300, 0, 120, False))
        self.assertEqual(reason_to_stop(120, 300, 0, 120, False), "client heartbeat expired")
        self.assertEqual(reason_to_stop(300, 300, 299, 120, False), "budget deadline reached")
        self.assertEqual(reason_to_stop(4, 300, 3, 120, True), "client requested shutdown")

    def test_pod_scope_and_stop_endpoint(self):
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "example1"}), patch("pod_lease.build_opener") as opener:
            with self.assertRaises(ValueError):
                pod_request("example2", "example-key", stop=True)
            opener.assert_not_called()
            response = MagicMock()
            response.status = 200
            opener.return_value.open.return_value.__enter__.return_value = response
            self.assertEqual(pod_request("example1", "example-key", stop=True), 200)
            requested = opener.return_value.open.call_args.args[0]
            self.assertEqual(requested.method, "POST")
            self.assertEqual(requested.full_url, "https://rest.runpod.io/v1/pods/example1/stop")

    def test_weekly_window_can_be_primary_or_secondary(self):
        week = {"windowDurationMins": 10080, "usedPercent": 95, "resetsAt": int(time.time()) + 300}
        for name in ("primary", "secondary"):
            self.assertEqual(weekly_window({"rateLimitsByLimitId": {"codex": {name: week}}})["remaining_percent"], 5)
        with self.assertRaises(RuntimeError):
            weekly_window({"rateLimits": {"primary": dict(week, windowDurationMins=300)}})

    def test_remote_lease_is_stopped_at_exact_reserve_or_failed_quota_read(self):
        args = SimpleNamespace(pod_id="example1", host="example.invalid", port=22,
                               deadline=int(time.time())+300, reserve=5, identity=None)
        for readings, raises in (([{"remaining_percent": 6}, {"remaining_percent": 5}], False),
                                 ([RuntimeError("quota unavailable")], True)):
            with patch("weekly_pod_guard.subprocess.run") as run, patch("weekly_pod_guard.time.sleep"), \
                 patch("weekly_pod_guard.read_quota", side_effect=readings), patch("builtins.print"):
                if raises:
                    with self.assertRaises(RuntimeError):
                        watch_pod(args)
                else:
                    watch_pod(args)
            commands = [call.args[0][-1] for call in run.call_args_list]
            self.assertTrue(commands[-1].startswith("touch /tmp/tco-lease-"))
            self.assertTrue(commands[-1].endswith("/stop"))
            self.assertEqual(sum(c.endswith("/heartbeat") for c in commands), 0 if raises else 1)
