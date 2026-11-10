"""Read the signed-in Codex weekly window and renew a single remote GPU lease."""
import argparse
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid


def weekly_window(result):
    by_id = result.get("rateLimitsByLimitId") or {}
    snapshot = by_id.get("codex") or result.get("rateLimits") or {}
    windows = [snapshot.get(name) or {} for name in ("primary", "secondary")]
    matches = [w for w in windows if w.get("windowDurationMins") == 7 * 24 * 60]
    if len(matches) != 1:
        raise RuntimeError("No unambiguous Codex weekly usage window")
    used = matches[0].get("usedPercent")
    reset = matches[0].get("resetsAt")
    if not isinstance(used, (int, float)) or not 0 <= used <= 100 or not isinstance(reset, int) or reset <= time.time():
        raise RuntimeError("The weekly usage reading is missing or stale")
    return {"remaining_percent": 100 - used, "reset_utc_epoch": reset}


def read_quota():
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen([shutil.which("codex") or "codex", "app-server", "--listen", "stdio://"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding="utf-8", creationflags=flags)
    messages = queue.Queue()

    def collect():
        for line in proc.stdout:
            try:
                messages.put(json.loads(line))
            except ValueError:
                continue

    threading.Thread(target=collect, daemon=True).start()

    def send(data):
        proc.stdin.write(json.dumps(data) + "\n")
        proc.stdin.flush()

    def receive(request_id):
        end = time.monotonic() + 15
        while time.monotonic() < end:
            try:
                data = messages.get(timeout=1)
            except queue.Empty:
                continue
            if data.get("id") == request_id:
                if "result" not in data:
                    raise RuntimeError("Codex usage request was unavailable")
                return data["result"]
        raise TimeoutError("Timed out reading Codex usage")

    try:
        send(dict(id=1, method="initialize", params=dict(clientInfo=dict(name="tco-budget", version="1"))))
        receive(1)
        send(dict(method="initialized"))
        send(dict(id=2, method="account/rateLimits/read", params={}))
        return weekly_window(receive(2))
    finally:
        proc.terminate()
        try:
            proc.wait(3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def watch_pod(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", args.host) or not re.fullmatch(r"[a-zA-Z0-9]{4,64}", args.pod_id):
        raise ValueError("Invalid SSH host or Pod ID")
    if not 1 <= args.port <= 65535 or args.deadline <= time.time():
        raise ValueError("A live absolute budget deadline and SSH port are required")
    remote = "root@" + args.host
    session = uuid.uuid4().hex
    folder = "/tmp/tco-lease-" + session
    program = "/tmp/tco-guard-" + session + ".py"
    options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8"]
    if args.identity:
        options.extend(["-i", str(args.identity)])
    ssh = ["ssh", "-p", str(args.port), *options, remote]
    subprocess.run(["scp", "-P", str(args.port), *options, str(Path(__file__).with_name("pod_lease.py")),
                    remote + ":" + program], check=True, timeout=25)
    command = f"nohup python3 {program} --pod-id {args.pod_id} --deadline {args.deadline} --session {session} </dev/null >{program}.log 2>&1 &"
    subprocess.run(ssh + [command], check=True, timeout=15)
    # A disconnected client cannot extend the two-minute lease.
    ready = f"for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do if test -f {folder}/ready && grep -Fxq {args.pod_id} {folder}/ready; then exit 0; fi; sleep 1; done; exit 1"
    subprocess.run(ssh + [ready], check=True, timeout=28)
    print("Remote lease armed for", args.pod_id, "until UTC epoch", args.deadline, flush=True)
    try:
        while time.time() < args.deadline:
            status = read_quota()
            print(json.dumps(status), flush=True)
            if status["remaining_percent"] <= args.reserve:
                break
            subprocess.run(ssh + ["touch " + folder + "/heartbeat"], check=True, timeout=15)
            time.sleep(30)
    finally:
        subprocess.run(ssh + ["touch " + folder + "/stop"], check=True, timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=float, default=5.)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    watch = sub.add_parser("watch-pod")
    watch.add_argument("--host", required=True)
    watch.add_argument("--port", type=int, required=True)
    watch.add_argument("--pod-id", required=True)
    watch.add_argument("--deadline", type=int, required=True)
    watch.add_argument("--identity", type=Path)
    args = parser.parse_args()
    if not 0 <= args.reserve <= 100:
        parser.error("reserve must be a percentage")
    if args.mode == "status":
        status = read_quota()
        print(json.dumps(dict(**status, at_reserve=status["remaining_percent"] <= args.reserve)))
    else:
        watch_pod(args)


if __name__ == "__main__":
    main()
