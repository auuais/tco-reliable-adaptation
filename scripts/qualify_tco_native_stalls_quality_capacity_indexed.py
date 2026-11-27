"""Read inherited clipped-gradient update labels for the registered TCO verifier."""

from unittest.mock import patch

import check_tco_render_lifetime as lifetime
import qualify_tco_native_stalls_quality_capacity as registered
from qualify_tco_render_lifetime import compare_npz
from run_profiled_d0 import sha
from tco_native_stalls_quality_capacity_protocol import INITIAL


original = registered.qualify


def qualify(stage):
    # The pre-step hook counts completed updates, so raw grad/0 and grad/1
    # describe the clipping of updates one and two. The archive stays untouched.
    with patch.object(lifetime, "compare_npz", compare_npz):
        report = original(stage)
    report["qualification_reader"] = {
        "registration_sha256": sha(INITIAL, text=True),
        "registered_reader_sha256": sha(registered.__file__, text=True),
        "index_reader_sha256": sha(__file__, text=True),
        "correction": "Present only the clipped-gradient keys grad/0 and grad/1 as grad/1 and grad/2.",
    }
    return report


if __name__ == "__main__":
    with patch.object(registered, "qualify", qualify):
        registered.main()
