"""Verify complete MASt3R prediction and native-output copies on distinct Windows disks."""
import argparse
from pathlib import Path
import shutil
import subprocess

from score_mast3r_joint_priors import PUBLIC,RAW,read,sha,dump


def archive(destination):
    destination=destination.resolve()
    primary=RAW.resolve()
    disks=[]
    for path in (primary,destination):
        drive=path.drive.rstrip(':')
        if len(drive)!=1 or not drive.isalpha():
            raise ValueError('This archive requires two local Windows drives')
        disk=subprocess.check_output(['powershell.exe','-NoProfile','-Command',
             f'(Get-Partition -DriveLetter {drive}).DiskNumber'],text=True).strip()
        disks.append(int(disk))
    if disks[0]==disks[1]:
        raise ValueError('Backup resolves to the same physical disk')
    seal=read(PUBLIC/'predictions.json')
    assert len(seal['rows'])==21
    points=sorted({primary/'points'/(r['prediction']+'.npz') for r in seal['rows']})
    captures=sorted(p for p in primary.rglob('*.npz') if 'points' not in p.parts)
    inventory=[]
    for source in points+captures:
        relative=source.relative_to(primary)
        target=destination/relative
        expected=sha(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copyfile(source,target)
        if sha(target)!=expected:
            raise ValueError('Backup checksum differs: '+str(relative))
        inventory.append(dict(file=relative.as_posix(),bytes=source.stat().st_size,sha256=expected))
    point_records=[dict(file=Path(v['file']).name,bytes=v['bytes'],sha256=v['sha256'])
                   for v in inventory if v['file'].startswith('points/')]
    dump(PUBLIC/'archive.json',dict(verified_physical_drives=2,files=point_records,
         total_bytes=sum(v['bytes'] for v in point_records),native_and_probe_files=[v for v in inventory if not v['file'].startswith('points/')]))
    print('Verified',len(points),'point maps and',len(captures),'native/probe archives on two physical disks.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    archive(p.parse_args().destination)
