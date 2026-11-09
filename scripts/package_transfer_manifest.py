"""Seal the three finished cloud packages for checked transfer."""
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
archives = {}
for name, destination in [("tco_project.tar.gz", "."), ("tco_runtime.tar.gz", "."),
                          ("midterm_inputs.tar.gz", "tco-assets")]:
    path = root / "cloud" / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    archives[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest(), "extract_into": destination}
manifest = {"archives": archives, "total_bytes": sum(a["bytes"] for a in archives.values())}
(root / "cloud/transfer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
