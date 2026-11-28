"""Run the pinned timing bridge, checking locally vendored TCO against its own Git repository."""
import argparse
from pathlib import Path
import subprocess
from unittest.mock import patch

import run_tco_two_track_a_local as launcher
import tco_native_harness_bridge as bridge
from tco_two_track_protocol import PUBLIC, ROOT, digest, read


def invoke(phase):
    upstream = ROOT/"third_party/TCO"
    pins = read(PUBLIC/"runtime.json")
    outer = launcher.committed

    def verify(*paths):
        local, upstream_names = [], []
        for path in paths:
            path = path.resolve()
            if upstream in path.parents:
                upstream_names.append(path.relative_to(upstream).as_posix())
            else:
                local.append(path)
        outer(Path(__file__),PUBLIC/"runtime.json",*local)
        revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=upstream,text=True).strip()
        if revision != pins["upstream_commit"]:
            raise ValueError("Different registered upstream checkout")
        subprocess.run(["git","ls-files","--error-unmatch","--",*upstream_names],cwd=upstream,check=True,capture_output=True)
        exceptions = pins["upstream_modified_files"]
        clean = [name for name in upstream_names if name not in exceptions]
        if clean:
            subprocess.run(["git","diff","--exit-code","HEAD","--",*clean],cwd=upstream,check=True,capture_output=True)
        if any(digest(upstream/name)!=expected for name,expected in exceptions.items()):
            raise ValueError("Local compatibility patch no longer matches its pinned bytes")

    # The original bridge still verifies every scientific file's registered hash and
    # the entire completed core roster. Only Git index ownership is resolved here.
    with patch.object(launcher,"committed",verify):
        bridge.execute(phase)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("heads","fire"))
    invoke(parser.parse_args().phase)
