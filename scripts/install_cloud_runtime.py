"""Install a checksum-verified local wheel/deb bundle without network access."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=pathlib.Path, required=True)
parser.add_argument("--record", type=pathlib.Path, required=True)
args = parser.parse_args()
started = time.monotonic()
manifest = json.loads((args.runtime / "manifest.json").read_text())
assert sys.version_info[:2] == (3, 11)
assert importlib.metadata.version("torch") == "2.7.1+cu128"
for name, expected in manifest["sha256"].items():
    file = args.runtime / name
    assert hashlib.sha256(file.read_bytes()).hexdigest() == expected, name
env = dict(os.environ, DEBIAN_FRONTEND="noninteractive", PIP_NO_INDEX="1")
debs = sorted(map(str, (args.runtime / "debs").glob("*.deb")))
assert debs
subprocess.run(["dpkg", "--force-confold", "-i", *debs], check=True, env=env)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--require-hashes", "--find-links", str(args.runtime / "wheelhouse"),
                "-r", str(args.runtime / "requirements-hashed.txt")], check=True, env=env)
normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
actual = {normalize(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}
expected = manifest["expected_distributions"]
mismatch = {name: {"expected": version, "actual": actual.get(name)}
            for name, version in expected.items() if actual.get(name) != version}
assert not mismatch, mismatch
subprocess.run([sys.executable, "-c", "import open3d,cv2,torch,gsplat; from gsplat import csrc; print('Precompiled renderer:', csrc.__file__)"], check=True)
args.record.parent.mkdir(parents=True, exist_ok=True)
args.record.write_text(json.dumps({"status": "passed", "install_seconds": time.monotonic() - started,
                                   "verified_distributions": len(expected),
                                   "runtime_manifest_sha256": hashlib.sha256((args.runtime / "manifest.json").read_bytes()).hexdigest(),
                                   "network_required_for_dependencies": False,
                                   "compiled_cuda_architectures": manifest["cuda_architectures"]}, indent=2) + "\n")
print(args.record.read_text())
