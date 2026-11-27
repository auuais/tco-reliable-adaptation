"""Replay the original study with the recorded allocator accounting correction."""

from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

import replay_tco_five_view_replay_cache as registered
from run_profiled_d0 import read, sha, write_new


PREFIX = "Changed GPU configuration for "
ALLOCATOR = "garbage_collection_threshold:0.75,expandable_segments:True"
GSPLAT = "339c0272e544d6a166814cf49e2d21f500eaf9018e69585f1fcaadb5ed3c940e"


def independent_allocator_check(identifier):
    matches = []
    for phase in registered.PHASES:
        path = registered.PUBLIC / f"{phase}_evidence.json"
        if path.is_file():
            matches.extend((phase, row) for row in read(path)["runs"]
                           if row["id"] == identifier and row["result"] is not None)
    if len(matches) != 1:
        raise ValueError(f"Expected one allocator receipt for {identifier}")
    phase, record = matches[0]
    result = record["result"]
    runtime = result["runtime"]
    limit = result["cuda_device_total"] * 0.875
    if runtime != {**runtime, "gpu": "NVIDIA GeForce RTX 3080", "gpu_memory_fraction": 0.875,
                   "allocator": ALLOCATOR, "torch": "2.7.1+cu128", "cuda": "12.8", "gsplat_sha256": GSPLAT}:
        raise ValueError(f"Changed runtime in {identifier}")
    allocated, reserved = result["peak_allocated"], result["peak_reserved"]
    if not 0 < allocated <= limit < reserved <= result["cuda_device_total"]:
        raise ValueError(f"Unexpected allocator statistics in {identifier}")
    return dict(id=identifier, phase=phase, complete_sha256=record["attempts"][-1]["complete_sha256"],
                peak_allocated=allocated, peak_reserved=reserved, fraction_limit_bytes=limit,
                reserved_above_limit_bytes=reserved-limit)


@contextmanager
def corrected_replay(exceptions):
    original = registered.require

    def require(condition, message):
        if not condition and message.startswith(PREFIX):
            identifier = message.removeprefix(PREFIX)
            exceptions[identifier] = independent_allocator_check(identifier)
            return
        original(condition, message)

    with patch.object(registered, "require", require):
        yield


def main():
    command, *options = sys.argv[1:]
    exceptions = {}
    with corrected_replay(exceptions), patch.object(sys, "argv", [sys.argv[0], *options]):
        if command == "replay":
            registered.main()
        elif command == "resources":
            from report_five_view_resources import main as report
            report()
        elif command == "lock":
            from extend_five_view_design import extend
            extend("lock")
        elif command == "archive":
            from seal_five_view_archive import main as archive
            archive()
        elif command == "allocator-receipt":
            for phase in registered.PHASES:
                registered.verified(phase)
            result = dict(
                rule="Verify effective allocator configuration, runtime pins, and peak allocated bytes. "
                     "Publish reserved bytes and retain any above-limit reserved counter.",
                original_source_sha256_lf=sha(Path(registered.__file__), text=True),
                correction_source_sha256_lf=sha(Path(__file__), text=True),
                affected_cases=list(exceptions.values()), rental_compute_usd=0,
            )
            path = registered.PUBLIC / "allocator_reservation_replay.json"
            if path.is_file():
                if read(path) != result:
                    raise ValueError("The allocator correction receipt differs")
            else:
                write_new(path, result)
        else:
            raise ValueError(f"Unknown release command: {command}")
    print(f"Verified {len(exceptions)} allocator accounting exception(s)")


if __name__ == "__main__":
    main()
