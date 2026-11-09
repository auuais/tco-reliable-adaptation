"""Bundle only public source and experiment scripts; no credentials or run data."""
import hashlib
import json
import pathlib
import subprocess
import tarfile

root = pathlib.Path(__file__).resolve().parents[1]
bundle = root / "cloud/TCO.bundle"
subprocess.run(["git", "-C", str(root / "third_party/TCO"), "bundle", "create", str(bundle), "HEAD"], check=True)
subprocess.run(["git", "-C", str(root / "third_party/TCO"), "bundle", "verify", str(bundle)], check=True)
files = [root / "README.md", root / "environment/Dockerfile", bundle]
files += [root / "environment/cloud-dependency-lock.json", root / "environment/requirements-cloud-delta.txt"]
for directory in ["scripts", "configs", "manifests", "patches", "third_party/protocol"]:
    files.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
files.extend(root / "cloud" / name for name in ["README.md", "job_plan.json", "inputs_manifest.json", "pod_start_command.txt", "runtime_archive.json"])
archive = root / "cloud/tco_project.tar.gz"
with tarfile.open(archive, "w:gz") as output:
    for path in sorted(files):
        output.add(path, arcname="tco-research/" + path.relative_to(root).as_posix())
record = {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "included_files": [p.relative_to(root).as_posix() for p in sorted(files)],
          "contains_credentials": False, "contains_private_dataset": False}
(root / "cloud/code_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "included_files"}, indent=2))
