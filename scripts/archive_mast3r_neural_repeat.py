"""Preserve the completed repeat and surface evaluation on a second physical disk."""
import argparse
from pathlib import Path
import shutil
import subprocess
from replay_mast3r_neural_repeat import ROOT, PUBLIC, RAW, read, sha, dump


def archive(destination):
    disks = []
    for path in (RAW.resolve(), destination.resolve()):
        drive = path.drive.rstrip(':')
        if len(drive) != 1 or not drive.isalpha():
            raise ValueError('Requires local Windows drive paths')
        disks.append(int(subprocess.check_output(['powershell.exe', '-NoProfile', '-Command',
            f'(Get-Partition -DriveLetter {drive}).DiskNumber'], text=True).strip()))
    assert len(set(disks)) == 2
    result = read(PUBLIC/'result.json')
    for name, digest in result['files'].items():
        assert sha(RAW/name) == digest
    assert len(list((RAW/'evaluation').rglob('checked.json'))) == 6
    records = []
    for label, folder in (('raw', RAW), ('public', PUBLIC)):
        for source in sorted(p for p in folder.rglob('*') if p.is_file()):
            if label == 'public' and source == PUBLIC/'archive.json':
                continue
            relative = Path(label)/source.relative_to(folder)
            target = destination/relative
            digest = sha(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            assert sha(target) == digest, 'Existing backup differs: '+str(relative)
            records.append(dict(file=relative.as_posix(), bytes=source.stat().st_size, sha256=digest))
    report = dict(verified_physical_drives=2, files=records, total_bytes=sum(r['bytes'] for r in records),
        repeat_result_sha256=sha(PUBLIC/'result.json'), reference_evaluations=6)
    dump(PUBLIC/'archive.json', report)
    print('Verified', len(records), 'repeat/evaluation files on two physical disks.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    archive(p.parse_args().destination)
