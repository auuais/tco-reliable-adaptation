"""Fetch the pinned upstream tree and apply the documented export-only patch."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "65387333877723cf99086f7f29089379dd50c59a"
upstream = ROOT / "third_party/TCO"
if not upstream.exists():
    upstream.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/cvlab-stonybrook/TCO.git", str(upstream)], check=True)
    subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", REVISION], check=True)
actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
if actual != REVISION:
    raise RuntimeError("Existing upstream is at a different revision; refusing to change it")
subprocess.run(["git", "-C", str(upstream), "submodule", "update", "--init", "--recursive"], check=True)
patches = list((ROOT / "patches").glob("*.patch"))
if len(patches) != 1:
    raise RuntimeError("Expected one reviewed compatibility patch")
patch = str(patches[0])
already = subprocess.run(["git", "-C", str(upstream), "apply", "--reverse", "--check", patch], capture_output=True)
if already.returncode:
    subprocess.run(["git", "-C", str(upstream), "apply", "--check", patch], check=True)
    subprocess.run(["git", "-C", str(upstream), "apply", patch], check=True)
print("Verified upstream revision and export compatibility patch:", actual)
