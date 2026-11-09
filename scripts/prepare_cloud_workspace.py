"""Verify transfer hashes and unpack into a fresh workspace, without overwriting runs."""
import argparse
import hashlib
import json
import pathlib
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=pathlib.Path, default=pathlib.Path("/workspace"))
parser.add_argument("--transfers", type=pathlib.Path, default=pathlib.Path("/workspace"))
args = parser.parse_args()
manifest = json.loads((args.transfers / "transfer_manifest.json").read_text())
marker = args.workspace / "tco-research/cloud/prepared-transfer.json"
if marker.exists():
    assert json.loads(marker.read_text()) == manifest, "Different bundle already installed; preserve existing experiment and select a new workspace."
    print("Identical workspace already prepared; retained existing evidence.")
    raise SystemExit(0)
for directory in ["tco-research", "tco-runtime", "tco-assets/data"]:
    assert not (args.workspace / directory).exists(), f"Refusing to overwrite {directory}"
for name, metadata in manifest["archives"].items():
    assert pathlib.Path(name).name == name
    path = args.transfers / name
    assert path.stat().st_size == metadata["bytes"], name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == metadata["sha256"], name
    print(f"Verified {name}", flush=True)
for name, metadata in manifest["archives"].items():
    destination = args.workspace / metadata["extract_into"]
    assert destination.resolve().is_relative_to(args.workspace.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.transfers / name) as archive:
        archive.extractall(destination, filter="data")
marker.write_text(json.dumps(manifest, indent=2) + "\n")
print("Prepared verified code, runtime and selected public data.")
