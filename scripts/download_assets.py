"""Fetch only the pinned checkpoint and selected public scene archives.

No credentials or external experiment tracking are used. Existing complete
files are retained. A resumable .part file is renamed after transfer finishes.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
DATA_BASE = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/"


def fetch(url, dest):
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    metadata = dest.with_suffix(dest.suffix + ".download.json")
    if dest.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["bytes"] == dest.stat().st_size and saved["url"] == url:
            print(f"Already downloaded: {dest.name}", flush=True)
            return saved
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "TCO-academic-reproduction/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0
                expected = int(response.headers.get("Content-Length", "0")) + offset
                mode = "ab" if offset else "wb"
                transferred = offset
                last_log = time.monotonic()
                print(f"Downloading {dest.name}: expected {expected / 1e9:.3f} GB", flush=True)
                with partial.open(mode) as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        transferred += len(chunk)
                        if time.monotonic() - last_log > 30:
                            print(f"{dest.name}: {transferred / 1e9:.3f}/{expected / 1e9:.3f} GB", flush=True)
                            last_log = time.monotonic()
                if expected and transferred != expected:
                    raise IOError(f"Incomplete: {transferred} of {expected} bytes")
            digest = hashlib.sha256()
            with partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            partial.replace(dest)
            record = {"url": url, "bytes": dest.stat().st_size, "sha256": digest.hexdigest()}
            metadata.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Finished {dest.name}: {record['sha256']}", flush=True)
            return record
        except Exception as exc:
            print(f"Download attempt {attempt + 1} failed for {dest.name}: {exc}", flush=True)
            if attempt == 3:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["heads"],
                        choices=["heads", "chess", "fire", "office", "pumpkin", "redkitchen", "stairs"])
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()
    tasks = []
    if not args.skip_checkpoint:
        for name in ("config.json", "model.safetensors"):
            tasks.append((f"https://huggingface.co/facebook/VGGT-1B/resolve/{REVISION}/{name}",
                          args.root / "checkpoints" / "VGGT-1B" / name))
    if not args.checkpoint_only:
        tasks.extend((DATA_BASE + scene + ".zip", args.root / "archives" / (scene + ".zip"))
                     for scene in args.scenes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: fetch(*pair), tasks))


if __name__ == "__main__":
    main()
