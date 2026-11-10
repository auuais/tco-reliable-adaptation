"""Stop only the current RunPod on lease expiry or at its absolute UTC deadline."""
import argparse
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def pod_request(pod, key, stop=False):
    if not re.fullmatch(r"[a-zA-Z0-9]{4,64}", pod) or pod != os.environ.get("RUNPOD_POD_ID"):
        raise ValueError("Expected the ID of this Pod")
    url = "https://rest.runpod.io/v1/pods/" + pod + ("/stop" if stop else "")
    req = Request(url, headers={"Authorization": "Bearer " + key}, method="POST" if stop else "GET")
    with build_opener(NoRedirect()).open(req, timeout=12) as response:
        if stop:
            return response.status
        return json.load(response)


def reason_to_stop(now, deadline, last_heartbeat, lease_seconds, stop_requested):
    if stop_requested:
        return "client requested shutdown"
    if now >= deadline:
        return "budget deadline reached"
    if now - last_heartbeat >= lease_seconds:
        return "client heartbeat expired"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod-id", required=True)
    p.add_argument("--deadline", required=True, type=int)
    p.add_argument("--session", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.session):
        raise ValueError("Provide a unique hexadecimal session ID")
    key = os.environ["RUNPOD_API_KEY"]
    if not key or pod_request(args.pod_id, key).get("id") != args.pod_id:
        raise ValueError("Could not verify the active Pod identity with its API key")
    directory = Path("/tmp") / ("tco-lease-" + args.session)
    directory.mkdir(mode=0o700, exist_ok=False)
    heartbeat = directory / "heartbeat"
    heartbeat.touch()
    (directory / "ready").write_text(args.pod_id)
    # Retain a monotonic hard deadline even if the system clock is moved backwards.
    hard_deadline = time.monotonic() + max(0, args.deadline - time.time())
    last_tick = time.monotonic()
    last_stamp = heartbeat.stat().st_mtime_ns
    while True:
        stamp = heartbeat.stat().st_mtime_ns
        if stamp != last_stamp:
            last_tick, last_stamp = time.monotonic(), stamp
        reason = reason_to_stop(time.monotonic(), hard_deadline, last_tick, 120, (directory / "stop").exists())
        if reason:
            print(reason, flush=True)
            break
        time.sleep(2)
    while True:
        try:
            pod_request(args.pod_id, key, stop=True)
            print("Stop request accepted", flush=True)
            # Remain available to retry if the provider has not stopped the container.
            time.sleep(15)
        except (HTTPError, OSError) as exc:
            print("Stop request failed", type(exc).__name__, getattr(exc, "code", ""), flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
