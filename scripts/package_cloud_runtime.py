"""Create an offline dependency archive without Torch/CUDA already in the base."""
import hashlib
import json
import pathlib
import tarfile
import zipfile
from email.parser import BytesParser

root = pathlib.Path(__file__).resolve().parents[1]
lock = json.loads((root / "environment/cloud-dependency-lock.json").read_text())
files = []
hashes = {}
requirements = []
found = {}
for path in sorted((root / "cloud/wheelhouse").glob("*.whl")):
    with zipfile.ZipFile(path) as wheel:
        metadata_path = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(metadata_path) == 1
        metadata = BytesParser().parsebytes(wheel.read(metadata_path[0]))
        name = metadata["Name"].lower().replace("_", "-").replace(".", "-")
        version = metadata["Version"]
        assert name in lock["delta"] and version == lock["delta"][name], (name, version)
        assert name not in found, name
        found[name] = version
        if name == "gsplat":
            assert any(n.endswith("/csrc.so") for n in wheel.namelist()), "Missing compiled CUDA extension"
    relative = "wheelhouse/" + path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[relative] = digest
    requirements.append(f"{name}=={version} --hash=sha256:{digest}")
    files.append((path, relative))
assert found == lock["delta"], sorted(set(lock["delta"]) - found.keys())
for path in sorted((root / "cloud/debs").glob("*.deb")):
    relative = "debs/" + path.name
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append((path, relative))
hashed = root / "cloud/requirements-hashed.txt"
hashed.write_text("\n".join(sorted(requirements)) + "\n")
hashes[hashed.name] = hashlib.sha256(hashed.read_bytes()).hexdigest()
files.append((hashed, hashed.name))
manifest = {"sha256": hashes, "expected_distributions": lock["expected_distributions"],
            "cuda_architectures": ["sm_86", "sm_89"], "gsplat_version": "1.5.3",
            "torch_version": "2.7.1+cu128", "python": "3.11",
            "build": "Official gsplat 1.5.3 source distribution; default optimized setup.py build; no source modification"}
manifest_path = root / "cloud/runtime_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
files.append((manifest_path, "manifest.json"))
archive = root / "cloud/tco_runtime.tar.gz"
with tarfile.open(archive, "w:gz", compresslevel=1) as output:
    for path, relative in files:
        output.add(path, arcname="tco-runtime/" + relative)
record = {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "wheels": len(found), "debs": len(list((root / "cloud/debs").glob("*.deb"))),
          "cuda_architectures": manifest["cuda_architectures"]}
(root / "cloud/runtime_archive.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
