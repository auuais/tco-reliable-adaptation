"""Save completed training, predictions and logs on a second physical Windows disk."""
import argparse
import json
from pathlib import Path
import subprocess

from archive_mast3r_crossover import copy_checked
from plan_mast3r_crossover import MANIFEST, sha256
from score_partner_shape import RAW, PUBLIC, ROOT, read, dump, check_seal


def archive(destination):
    check_seal(local=True)
    for name in ("scores.json", "headroom.json", "bias.json", "surface/scores.json", "README.md"):
        if not (PUBLIC/name).is_file():
            raise FileNotFoundError(PUBLIC/name)
    disks = []
    for root in (ROOT, destination.resolve()):
        letter = root.drive.rstrip(":")
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError("Expected a local Windows drive")
        disk = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
            f"(Get-Partition -DriveLetter {letter}).DiskNumber"], text=True)
        disks.append(int(disk))
    if disks[0] == disks[1]:
        raise ValueError("Archive requires a physically different disk")
    logs = RAW/"logs"
    logs.mkdir(exist_ok=True)
    names = subprocess.check_output(["docker", "ps", "-a", "--filter", "name=tco-partner-", "--format", "{{.Names}}"], text=True).splitlines()
    sessions = []
    for name in sorted(names):
        info = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        if info["State"]["Running"] or info["Image"] != read(MANIFEST)["image_sha256"]:
            raise ValueError(f"Unexpected container identity or state: {name}")
        with (logs/(name+".log")).open("wb") as log:
            subprocess.run(["docker", "logs", name], stdout=log, stderr=subprocess.STDOUT, check=True)
        sessions.append(dict(name=name, command=info["Config"]["Cmd"], image=info["Image"],
            gpu=bool(info["HostConfig"]["DeviceRequests"]), log_sha256=sha256(logs/(name+".log")),
            **{k:info["State"][k] for k in ("StartedAt","FinishedAt","ExitCode","OOMKilled")}))
    if sum(r["gpu"] and r["ExitCode"] == 0 for r in sessions) != 6:
        raise ValueError("Incomplete six-fold GPU execution record")
    dump(PUBLIC/"execution.json", dict(containers=sessions, additional_runpod_compute_usd=0))
    files = [f for folder in (PUBLIC, RAW) for f in folder.rglob("*") if f.is_file() and f.name != "archive.json"]
    files.append(ROOT/"docs/MAST3R_PARTNER_SHAPE.md")
    files.append(ROOT/"docs/MAST3R_VIEW_BIAS_SURFACE.md")
    copied = [copy_checked(f, destination/f.relative_to(ROOT)) for f in sorted(files)]
    summary = PUBLIC/"archive.json"
    dump(summary, dict(physical_disks=disks, files=copied, total_bytes=sum(r["bytes"] for r in copied)))
    copy_checked(summary, destination/summary.relative_to(ROOT))
    print("Verified", len(copied), "files on disks", disks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    archive(parser.parse_args().destination)
