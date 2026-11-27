"""Seal the five-view study using byte hashes for artifacts of every file type.

Usage matches archive_tco_five_view_replay_cache.py (plan, copy, verify).
"""

from pathlib import Path
from unittest.mock import patch

import archive_tco_five_view_replay_cache as study
from profiled_artifacts import resolve_logical
from replay_tco_courtyard_camera import payload_digest
from run_profiled_d0 import read, sha


def complete_markers(markers, validate_terminals):
    core = study.archive
    # The inherited planner has already included every file in the copy list.
    # Supply the remaining digests to its receipt validator and post-copy guards.
    for name in sorted(core.inventory() - markers.keys()):
        path = resolve_logical(core.ROOT, name, must_exist=True)
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while sealing {name}")
        markers[name] = digest

    targets = validate_terminals(markers)
    prefix = core.PRIVATE.relative_to(core.ROOT).as_posix() + "/launches"
    expected = {}
    for phase in study.PHASES:
        ledger = read(core.PUBLIC / f"{phase}_evidence.json")
        for case in ledger["runs"]:
            for receipt in case["attempts"]:
                name = f"{prefix}/{case['id']}/{receipt['attempt']}/terminal.json"
                if name in expected:
                    raise ValueError(f"One physical attempt occurs twice in the public ledgers: {name}")
                expected[name] = payload_digest(receipt)
    actual = {row["terminal"]: row["sha256"] for row in targets}
    if actual != expected:
        raise ValueError("Raw terminal receipt identities or bytes differ from the five phase ledgers")
    return targets


def main():
    core = study.archive
    validate_terminals = core.terminal_targets
    def validate(markers):
        return complete_markers(markers, validate_terminals)

    extension = core.ROOT / "results" / study.STUDY / "deterministic_extension.json"
    sources = read(extension)["source_sha256_lf"] if extension.is_file() else {}
    for name, digest in sources.items():
        if sha(core.ROOT / name, text=True) != digest:
            raise ValueError(f"Amended protocol source changed: {name}")
    tools = (*core.TOOLS, Path(__file__).resolve(), core.ROOT / "scripts/report_five_view_resources.py",
             core.ROOT / "scripts/replay_five_view_release.py",
             *(core.ROOT / name for name in sources))
    with patch.multiple(core, terminal_targets=validate, TOOLS=tools):
        study.main()


if __name__ == "__main__":
    main()
