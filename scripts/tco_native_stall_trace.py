"""Collect host and CUDA-stream progress for diagnostic TCO runs."""

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import inspect
from itertools import count
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import patch


TRACE_FILE = "phase_trace.jsonl"
OLD_STUDY = "tco_five_view_replay_cache_v1"
LOSS_ROW = re.compile(r"(?m)^\s*(\d+)\s*\|")
STACK_TOP = re.compile(
    r'Thread [^\n]*\(most recent call first\):\s+File "([^"]+)", line (\d+) in ([^\n]+)'
)
FINAL_PHASES = {"transformer", "camera_head", "depth_head", "depth.refinenet1.out_conv"}
INVENTORY_PHASES = {"update_entry", "render.target_normals", "render.rgb", "backward", "adam"}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    """A killed process can leave one partial final line."""
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.endswith("\n"):
                rows.append(json.loads(line))
    return rows


class PhaseTrace:
    def __init__(self, stream, torch, inventory):
        self.stream = stream
        self.torch = torch
        self.inventory = inventory
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sequence = count()
        self.pending = deque()
        self.state = {}
        self.steps = None
        self.monitor = None
        self.thread = threading.Thread(target=self.poll, name="trace-completions", daemon=True)
        self.thread.start()

    def mark(self, phase, edge, **values):
        with self.lock:
            if not self.stream.closed:
                row = dict(phase=phase, edge=edge, elapsed_seconds=time.monotonic() - self.started,
                           utc=utc(), updates_done=self.state.get("updates"), **values)
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.stream.flush()

    def poll(self):
        last_heartbeat = self.started
        try:
            while not self.stop.is_set() or self.pending:
                while self.pending:
                    phase, serial, update, begin, end = self.pending[0]
                    if not end.query():
                        break
                    self.pending.popleft()
                    self.mark(phase, "cuda_complete", serial=serial, update_at_entry=update,
                              cuda_elapsed_ms=begin.elapsed_time(end))
                if self.stop.is_set():
                    return
                now = time.monotonic()
                if now - last_heartbeat >= 2:
                    last_heartbeat = now
                    self.mark("progress", "heartbeat", pending_cuda_events=len(self.pending))
                self.stop.wait(0.2)
        except Exception as exc:
            self.mark("progress", "cuda_poll_error", exception=type(exc).__name__)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)
        self.mark("progress", "trace_closed", pending_cuda_events=len(self.pending),
                  poller_stopped=not self.thread.is_alive())

    def cuda_state(self, phase):
        runtime = self.torch.cuda
        statistics = dict(allocated=runtime.memory_allocated(), reserved=runtime.memory_reserved(),
                          peak_allocated=runtime.max_memory_allocated(), peak_reserved=runtime.max_memory_reserved())
        if self.inventory and self.monitor is not None and phase in INVENTORY_PHASES:
            statistics["inventory"] = self.monitor.inventory()
        return statistics

    @contextmanager
    def phase(self, name, *, cuda=True):
        update = self.state.get("updates")
        phase = "final." + name if update == self.steps and self.steps is not None and name in FINAL_PHASES else name
        serial = next(self.sequence)
        self.mark(phase, "enter", serial=serial)
        begin = None
        if cuda and self.torch.cuda.is_initialized():
            begin = self.torch.cuda.Event(enable_timing=True)
            begin.record()
            self.mark(phase, "cuda_begin_submitted", serial=serial, **self.cuda_state(phase))
        try:
            yield
        except Exception as exc:
            exceptional = "early_stop" if type(exc).__name__ == "_StopRecomputationError" else "error"
            self.mark(phase, exceptional, serial=serial, exception=type(exc).__name__)
            # Checkpoint replay intentionally unwinds here after submitting useful work.
            if exceptional == "early_stop":
                self.submit_end(phase, serial, update, begin)
            raise
        else:
            self.mark(phase, "exit", serial=serial)
            self.submit_end(phase, serial, update, begin)

    def submit_end(self, phase, serial, update, begin):
        if begin is not None:
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            self.mark(phase, "cuda_end_submitted", serial=serial, **self.cuda_state(phase))
            self.pending.append((phase, serial, update, begin, end))


class NoInventory:
    """Keep the source overlay's probes while avoiding saved-tensor hooks."""

    def remember(self, *_):
        pass


def wrap(function, trace, name, *, cuda=False):
    def traced(*args, **kwargs):
        with trace.phase(name, cuda=cuda):
            return function(*args, **kwargs)

    return traced


def instrument_interface(original):
    """Place diagnostic scopes around the upstream interface's actual transfers."""
    source = textwrap.dedent(inspect.getsource(original))
    h2d_first = "    imgs ="
    dtype_first = "    dtype ="
    d2h = "    return global_points.cpu().float().numpy()"
    if any(source.count(anchor) != 1 for anchor in (h2d_first, dtype_first, d2h)):
        raise ValueError("Pinned inference transfer anchors changed")
    start, end = source.index(h2d_first), source.index(dtype_first)
    source = (source[:start] + "    with _native_stall_scope('input_h2d'):\n"
              + textwrap.indent(source[start:end], "    ") + source[end:])
    source = source.replace(d2h, "    with _native_stall_scope('pipeline_d2h'):\n    " + d2h, 1)
    namespace = {}
    exec(compile(source, "TCO inference interface (stall trace)", "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


@contextmanager
def instrument_lifetime(lifetime, out, *, inventory=False):
    """Reuse the pinned lifetime runner with stream events instead of barrier timings."""
    with (out / TRACE_FILE).open("x", encoding="utf-8") as stream, ExitStack() as stack:
        trace = PhaseTrace(stream, lifetime.torch, inventory)
        stack.callback(trace.close)
        native_timeline = lifetime.LifetimeTimeline
        native_setup = lifetime.setup_execution

        def timeline(profile_stream, state, steps, monitor):
            trace.state, trace.steps, trace.monitor = state, steps, monitor
            profile_stream.write(json.dumps(dict(event_log=TRACE_FILE)) + "\n")
            profile_stream.flush()
            return trace

        def setup(model, job, state, audit, scope, monitor):
            return native_setup(model, job, state, audit, scope, monitor if inventory else None)

        def write(path, content):
            phase = "write_receipt" if path.name == "complete.json" else "write_json"
            with trace.phase(phase, cuda=False):
                return native_write(path, content)

        native_write = lifetime.write_new
        stack.enter_context(patch.multiple(lifetime, LifetimeTimeline=timeline, setup_execution=setup,
                                           write_new=write))
        if not inventory:
            stack.enter_context(patch.object(lifetime, "LifetimeMonitor", NoInventory))
        model_class = lifetime.base.tco_module.TCO_VGGT_LoRA
        stack.enter_context(patch.object(model_class, "from_pretrained",
                                         wrap(model_class.from_pretrained, trace, "load_checkpoint")))
        stack.enter_context(patch.object(lifetime.full40, "warm_device",
                                         wrap(lifetime.full40.warm_device, trace, "cuda_warmup")))
        stack.enter_context(patch.object(lifetime, "export",
                                         wrap(lifetime.export, trace, "prediction_d2h", cuda=True)))
        stack.enter_context(patch.object(lifetime, "unproject_depth_to_world_points_torch", wrap(
            lifetime.unproject_depth_to_world_points_torch, trace, "canonical_unprojection", cuda=True)))
        stack.enter_context(patch.object(lifetime.base, "evaluate",
                                         wrap(lifetime.base.evaluate, trace, "score_alignment_icp")))
        stack.enter_context(patch.object(lifetime.np, "savez_compressed",
                                         wrap(lifetime.np.savez_compressed, trace, "serialize_prediction")))
        interface = lifetime.base.infer_mv_pointclouds_w_priors_v2
        stack.enter_context(patch.dict(interface.__globals__, _native_stall_scope=trace.phase))
        stack.enter_context(patch.object(lifetime.base, "infer_mv_pointclouds_w_priors_v2",
                                         instrument_interface(interface)))
        # Check that the inherited runner still exposes the expected callback.
        if not callable(getattr(native_timeline, "phase", None)):
            raise ValueError("Pinned lifetime timeline changed")
        with trace.phase("case", cuda=False):
            yield trace


WINDOWS_TELEMETRY = r"""
$ErrorActionPreference = 'Stop'
while ($true) {
    $sampleStart = [DateTimeOffset]::UtcNow
    $gpu = $null
    $adapter = $null
    $hostMemory = $null
    $errors = @()
    try {
        $query = 'index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory'
        $gpu = @(nvidia-smi "--query-gpu=$query" --format=csv,noheader,nounits)
        if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi returned an error' }
    } catch { $errors += 'nvidia_smi' }
    try {
        $adapter = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
            Select-Object Name,DedicatedUsage,SharedUsage)
    } catch { $errors += 'wddm_adapter_memory' }
    try {
        $hostMemory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory |
            Select-Object AvailableMBytes,CommittedBytes,CommitLimit,PagesInputPersec,PagesOutputPersec
    } catch { $errors += 'host_memory' }
    $processes = @(Get-Process -Name 'vmmemWSL','com.docker.backend' -ErrorAction SilentlyContinue |
        Select-Object Id,ProcessName,WorkingSet64)
    $sampleEnd = [DateTimeOffset]::UtcNow
    [ordered]@{ utc=$sampleStart.ToString('o'); sample_seconds=($sampleEnd-$sampleStart).TotalSeconds;
        nvidia_smi=$gpu; wddm_adapter_memory=$adapter; host_memory=$hostMemory; processes=$processes;
        unavailable=$errors } | ConvertTo-Json -Depth 4 -Compress
    $remaining = 2000 - [int]([DateTimeOffset]::UtcNow-$sampleStart).TotalMilliseconds
    if ($remaining -gt 0) { Start-Sleep -Milliseconds $remaining }
}
"""


@contextmanager
def host_telemetry(destination):
    """Sample the Windows host during a diagnostic attempt, with no system changes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        if sys.platform != "win32":
            stream.write(json.dumps(dict(utc=utc(), unavailable=["windows_telemetry"])) + "\n")
            yield
            return
        with subprocess.Popen(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_TELEMETRY],
            stdout=stream, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
        ) as sampler:
            try:
                yield
            finally:
                sampler.terminate()
                try:
                    sampler.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    sampler.kill()


def summarize_trace(case_dir):
    trace = case_dir / TRACE_FILE
    if not trace.is_file():
        return None
    rows = json_lines(trace)
    outstanding = {}
    completed = defaultdict(list)
    for row in rows:
        if row["edge"] == "enter":
            outstanding[row["serial"]] = row
        elif row["edge"] in {"exit", "error", "early_stop"}:
            outstanding.pop(row["serial"], None)
        elif row["edge"] == "cuda_complete":
            completed[row["phase"]].append(row)
    last = {}
    for field in ("enter", "exit", "cuda_end_submitted", "cuda_complete", "error", "cuda_poll_error", "heartbeat"):
        last[field] = next((row for row in reversed(rows) if row["edge"] == field), None)
    last["open_host_phases"] = list(outstanding.values())
    adam = [row for row in rows if row["phase"] == "adam" and row["edge"] == "cuda_complete"]
    last["last_device_confirmed_adam_update"] = max((row["update_at_entry"] + 1 for row in adam), default=0)
    last["row_count"], last["trace_sha256"] = len(rows), digest(trace)
    last["elapsed_sec"] = rows[-1]["elapsed_seconds"] if rows else None
    last["cuda_phase_timings"] = {
        name: dict(count=len(spans), median_ms=statistics.median(row["cuda_elapsed_ms"] for row in spans),
                   slowest=max(spans, key=lambda row: row["cuda_elapsed_ms"]))
        for name, spans in sorted(completed.items())
    }
    inventories = [row for row in rows if "inventory" in row and row["phase"] in INVENTORY_PHASES]
    last["second_update_inventory"] = [row for row in inventories if row["updates_done"] == 1]
    return last


def attempt_history(directory, cases):
    terminal_path = directory / "terminal.json"
    terminal = read_json(terminal_path)
    log_path = directory / "stdout.log"
    if terminal["log_sha256"] != digest(log_path):
        raise ValueError(f"Changed historical log for {terminal['job']}")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    printed = [int(row) for row in LOSS_ROW.findall(log)]
    stack_samples = [
        dict(file=file.removeprefix("/workspace/").split(" (registered")[0], line=int(line), function=function)
        for file, line, function in STACK_TOP.findall(log)
    ]
    case_dir = cases / terminal["job"] / terminal["attempt"] / "case"
    complete = case_dir / "complete.json"
    result = None
    if terminal["verdict"] == "completed":
        if terminal["complete_sha256"] != digest(complete):
            raise ValueError(f"Changed historical completion for {terminal['job']}")
        result = read_json(complete)
    profile = case_dir / "profile.jsonl"
    events = json_lines(profile)
    return dict(
        job=terminal["job"], attempt=terminal["attempt"], verdict=terminal["verdict"],
        end_utc=terminal["utc"], wall_seconds=terminal["wall_seconds"],
        deadline_seconds=terminal["outer_seconds"], stopped=terminal["stopped"],
        log_sha256=terminal["log_sha256"], terminal_sha256=digest(terminal_path),
        complete_sha256=terminal["complete_sha256"], last_zero_based_loss_row=printed[-1] if printed else None,
        updates_verified=result["updates"] if result is not None else None,
        scene_seconds=result["scene_seconds"] if result is not None else None,
        last_stack_samples=stack_samples[-3:], profile_last_event=events[-1] if events else None,
        profile_sha256=digest(profile) if profile.is_file() else None,
    )


def historical(root):
    archive = root / "reports" / OLD_STUDY
    names = [f"dev_{scene}_native_{seed}" for scene in ("chess", "heads", "courtyard") for seed in (42, 43, 44)]
    names.extend(f"dev_{scene}_repeat_{seed}" for scene in ("chess", "courtyard") for seed in (42, 43, 44))
    names.extend(f"profile_{scene}_native" for scene in ("heads", "courtyard"))
    rows = [attempt_history(receipt.parent, archive / "cases") for name in names
            for receipt in sorted((archive / "launches" / name).glob("*/terminal.json"))]
    if {row["job"] for row in rows} != set(names):
        raise ValueError("Missing historical investigation slots")
    return dict(study=OLD_STUDY, attempts=rows, rental_compute_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    old = subparsers.add_parser("historical")
    old.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    trial = subparsers.add_parser("summarize")
    trial.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    result = historical(args.root) if args.command == "historical" else summarize_trace(args.case_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
