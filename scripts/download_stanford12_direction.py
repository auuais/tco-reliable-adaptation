"""Acquire the three publisher archives using checked, resumable HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


ENVIRONMENTS = ("apt1", "apt2", "office2")
BASE = "https://graphics.stanford.edu/projects/reloc/data"
CHUNK = 16 * 1024 * 1024


def fetch(url, path, byte_range):
    headers = subprocess.check_output(
        ["curl", "-fsSL", "--retry", "2", "--max-time", "90", "--range",
         byte_range, "-D", "-", "-o", str(path), url], text=True)
    match = re.search(r"(?im)^Content-Range: bytes (\d+)-(\d+)/(\d+)", headers)
    if not match:
        raise ValueError(f"Missing HTTP Content-Range for {url}")
    start, end, total = map(int, match.groups())
    if path.stat().st_size != end - start + 1:
        raise ValueError(f"Wrong byte count for {url}")
    return start, end, total


def download(root, environment, pool):
    archive = root / f"{environment}.zip"
    if zipfile.is_zipfile(archive):
        return
    chunks = root / f"{environment}.parts"
    chunks.mkdir(exist_ok=True)
    url = f"{BASE}/{environment}.zip"
    _, _, total = fetch(url + "?direction-meta=1", chunks / "probe", "-65536")

    def get(start):
        end = min(start + CHUNK, total) - 1
        final = chunks / str(start)
        if final.exists() and final.stat().st_size == end - start + 1:
            return final
        temp = final.with_suffix(".part")
        if fetch(f"{url}?direction-byte={start}", temp, f"{start}-{end}") != (start, end, total):
            raise ValueError(f"Unexpected range for {environment}:{start}")
        temp.replace(final)
        return final

    combined = root / f"{environment}.assembling"
    with combined.open("wb") as out:
        for index, part in enumerate(pool.map(get, range(0, total, CHUNK)), 1):
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, out)
            if index % 20 == 0:
                print(environment, index * CHUNK, "/", total, flush=True)
    if combined.stat().st_size != total or not zipfile.is_zipfile(combined):
        raise ValueError(f"Incomplete ZIP for {environment}")
    combined.replace(archive)
    print(environment, total, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    args = parser.parse_args()
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as chunks, ThreadPoolExecutor(max_workers=3) as archives:
        jobs = [archives.submit(download, args.archive_dir, env, chunks) for env in ENVIRONMENTS]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
