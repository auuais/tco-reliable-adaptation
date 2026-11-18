"""Record the pinned local runtime and immutable sources for profiled LoRA."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "third_party/mast3r": "f5209afc300cec36239a7ac992263f36847bbba0",
    "third_party/mast3r/dust3r": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
    "third_party/mast3r/dust3r/croco": "d7de0705845239092414480bd829228723bf20de",
}
IMAGE = "sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50"
CHECKPOINT = "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(directory), *args], text=True).strip()


def verify_tree(directory):
    command = ["git", "-c", "safe.directory=*", "-C", str(directory)]
    listing = subprocess.check_output(command + ["ls-tree", "-rz", "HEAD"])
    entries = []
    for row in listing.split(b"\0"):
        if not row:
            continue
        info, name = row.split(b"\t", 1)
        mode, kind, blob = info.decode().split()
        if kind == "blob":
            entries.append((name.decode(), blob))
    request = ("\n".join(blob for _, blob in entries) + "\n").encode()
    batch = subprocess.run(command + ["cat-file", "--batch"], input=request,
                           stdout=subprocess.PIPE, check=True).stdout
    cursor, crlf_only = 0, []
    for name, blob in entries:
        boundary = batch.index(b"\n", cursor)
        returned, kind, size = batch[cursor:boundary].decode().split()
        if returned != blob or kind != "blob":
            raise ValueError("Unexpected Git object response")
        cursor = boundary + 1
        expected = batch[cursor:cursor + int(size)]
        cursor += int(size) + 1
        actual = (directory / name).read_bytes()
        if actual != expected:
            if actual.replace(b"\r\n", b"\n") != expected:
                raise ValueError(f"Tracked upstream content differs from pinned Git blob: {name}")
            crlf_only.append(name)
    return dict(tracked_blobs_verified=len(entries), crlf_only_paths=crlf_only,
                content_matches_pinned_commit=True)


def run(weights, image):
    if image != IMAGE:
        raise ValueError("Local Docker image differs from the registered runtime")
    if sha(weights / "model.safetensors") != CHECKPOINT:
        raise ValueError("Checkpoint changed")
    revisions = {name: git(ROOT / name, "rev-parse", "HEAD") for name in EXPECTED}
    if revisions != EXPECTED:
        raise ValueError("Upstream source revision changed")
    source_status = {name: verify_tree(ROOT / name) for name in EXPECTED}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the pinned local container")
    properties = torch.cuda.get_device_properties(0)
    probe = torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64, device="cuda")
    solve = torch.linalg.cholesky(probe)
    torch.testing.assert_close(solve @ solve.T, probe)
    result = dict(scope="Runtime/input verification only; no accuracy experiment",
                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=properties.name, total_vram_bytes=properties.total_memory,
                  docker_image=image, checkpoint_sha256=CHECKPOINT,
                  upstream_revisions=revisions, upstream_blob_checks=source_status,
                  execution_commit=git(ROOT, "rev-parse", "HEAD"),
                  source_sha256=sha(Path(__file__)), cuda_float64_cholesky_pass=True,
                  rental_compute_usd=0, pass_all=True)
    target = ROOT / "results/lora_profiled_v1/environment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("Preserve the existing environment evidence")
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("/weights"))
    parser.add_argument("--image", required=True, help="Docker image ID inspected by the launching host")
    args = parser.parse_args()
    run(args.weights, args.image)
