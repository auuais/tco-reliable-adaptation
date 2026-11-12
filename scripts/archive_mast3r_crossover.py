"""Copy the local crossover evidence to a second physical drive and verify every file."""
import argparse
from pathlib import Path
import shutil
import subprocess

from evaluation_stability import dump
from plan_mast3r_crossover import ROOT, MANIFEST, read, require, sha256


PUBLIC = ROOT / "results/mast3r_geometry_matching_v1"
RAW = ROOT / "reports/mast3r_geometry_matching_v1"


def copy_checked(source, target):
    digest = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    require(sha256(target) == digest, f"Backup differs: {target}")
    return dict(file=source.relative_to(ROOT).as_posix(), bytes=source.stat().st_size, sha256=digest)


def archive(destination, final):
    disks = []
    for path in (ROOT, destination.resolve()):
        drive = path.drive.rstrip(":")
        require(len(drive) == 1 and drive.isalpha(), "Local Windows drive required")
        disk = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
                                        f"(Get-Partition -DriveLetter {drive}).DiskNumber"], text=True).strip()
        disks.append(int(disk))
    require(len(set(disks)) == 2, "Two different physical drives required")
    completed = sorted((PUBLIC / "cases").glob("*/complete.json"))
    if final:
        require(len(completed) == 48, "Full experiment has incomplete cases")
        for name in ("predictions.json", "associations.json", "scores.json", "match_audit.json"):
            require((PUBLIC / name).is_file(), f"Final evidence missing: {name}")
    sources = [MANIFEST]
    for completion in completed:
        case = completion.parent.name
        sources.extend(p for p in (RAW / "cases" / case).rglob("*") if p.is_file())
        sources.extend(p for p in completion.parent.rglob("*") if p.is_file())
    for folder in (RAW / "attempts", RAW / "diagnosis", PUBLIC / "attempts", PUBLIC / "diagnosis"):
        sources.extend(p for p in folder.rglob("*") if p.is_file())
    if final:
        sources.extend(p for folder in (RAW, PUBLIC) for p in folder.rglob("*")
                       if p.is_file() and p.name != "archive.json")
    records = [copy_checked(source, destination / source.relative_to(ROOT)) for source in sorted(set(sources))]
    record = dict(physical_disks=disks, completed_cases=[p.parent.name for p in completed],
                  final=final, files=records, total_bytes=sum(r["bytes"] for r in records))
    output = PUBLIC / "archive.json" if final else ROOT / "cloud/mast3r_crossover_checkpoint.json"
    dump(output, record)
    copy_checked(output, destination / output.relative_to(ROOT))
    print(f"Verified {len(records)} files for {len(completed)} cases on two physical drives.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    archive(args.destination, args.final)
