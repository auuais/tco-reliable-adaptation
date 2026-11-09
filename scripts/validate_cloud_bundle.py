"""Check the packaged source can be restored and patched in clean Linux paths."""
import hashlib
import json
import pathlib
import subprocess
import tarfile
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
archive = root / "cloud/tco_project.tar.gz"
manifest = json.loads((root / "cloud/code_manifest.json").read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
with tempfile.TemporaryDirectory() as temporary:
    temp = pathlib.Path(temporary)
    with tarfile.open(archive) as stream:
        stream.extractall(temp, filter="data")
    project = temp / "tco-research"
    upstream = project / "third_party/TCO"
    subprocess.run(["git", "clone", str(project / "cloud/TCO.bundle"), str(upstream)], check=True)
    subprocess.run(["git", "-C", str(upstream), "apply", str(project / "patches/0001-bfloat16-numpy-export.patch")], check=True)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "65387333877723cf99086f7f29089379dd50c59a"
    hashes = {}
    for name in ["tco_vggt_lora.py", "tco_loss.py", "objective.py", "constraints.py", "peft/lora.py", "utils/interfaces.py"]:
        restored = (upstream / name).read_text()
        local = (root / "third_party/TCO" / name).read_text()
        assert restored == local, name
        hashes[name] = hashlib.sha256(restored.encode()).hexdigest()
    subprocess.run(["bash", "-n", str(project / "scripts/cloud_bootstrap.sh")], check=True)
    subprocess.run(["bash", "-n", str(project / "cloud/pod_start_command.txt")], check=True)
record = {"status": "passed", "source_archive_sha256": manifest["sha256"],
          "restored_commit": commit, "patched_method_matches_local_source": True,
          "normalized_source_sha256": hashes,
          "limitation": "Cloud provisioning and execution remain untested until the funded GPU session."}
(root / "environment/cloud_bundle_validation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
