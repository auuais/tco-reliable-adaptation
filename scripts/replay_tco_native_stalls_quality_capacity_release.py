"""Replay the study with the registered receipt-validation correction applied."""

from unittest.mock import patch

import replay_tco_native_stalls_quality_capacity as registered
from tco_native_stalls_quality_capacity_protocol import receipt_correction


verify_registered = registered.verify


def verify(plan, ledger, initial_sha, expected_windows=None):
    correction = receipt_correction(plan)
    changes = correction.get("source_changes", {})
    effective_sources = {**plan["source_sha256_lf"], **{name: row["after"] for name, row in changes.items()}}
    return verify_registered({**plan, "source_sha256_lf": effective_sources}, ledger, initial_sha, expected_windows)


def replay(stage):
    with patch.multiple(registered, verify=verify, receipt_correction=effective_receipt_correction):
        return registered.replay(stage)


def effective_receipt_correction(plan):
    # The registered reader checks the correction against the original source map.
    from tco_native_stalls_quality_capacity_protocol import INITIAL, read

    return receipt_correction(read(INITIAL))


if __name__ == "__main__":
    with patch.multiple(registered, verify=verify, receipt_correction=effective_receipt_correction):
        registered.main()
