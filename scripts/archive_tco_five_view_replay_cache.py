"""Copy and verify five-view experiment evidence on the second physical disk."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import archive_tco_render_lifetime as archive
from replay_tco_five_view_replay_cache import PHASES, PUBLIC, ROOT, STUDY, authority, replay


def plan_decision():
    return {"attempt_counts": replay()["terminal_attempt_counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "copy", "verify"))
    parser.add_argument("--snapshot", default="study-v1")
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    if args.check_source and args.action != "verify":
        parser.error("Use --check-source with verify")
    tools = (*archive.TOOLS, Path(__file__).resolve(), *(authority(phase) for phase in PHASES[1:-1]))
    with patch.multiple(archive, STUDY=STUDY, PRIVATE=ROOT / "reports" / STUDY, PUBLIC=PUBLIC,
                        INITIAL=authority("initial"), LOCKED=authority("locked"), replay=plan_decision,
                        DESTINATION=Path("E:/ResearchData/TCO/research_results") / STUDY,
                        RECEIPT=PUBLIC / "archive_receipt.json", TOOLS=tools):
        result = archive.archive(args.action, args.snapshot, args.check_source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
