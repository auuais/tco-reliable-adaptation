"""Incrementally preserve completed neural-transfer cases on a distinct physical disk."""
import argparse
from pathlib import Path
import shutil
import subprocess
from verify_mast3r_neural_plan import PUBLIC,ROOT
from evaluation_stability import dump,sha
from score_saved_geometry import read

RAW=ROOT/'reports/mast3r_neural_transfer'


def archive(destination,complete=False):
    disks=[]
    for path in (RAW.resolve(),destination.resolve()):
        drive=path.drive.rstrip(':')
        if len(drive)!=1 or not drive.isalpha():raise ValueError('Requires local Windows drive paths')
        disks.append(int(subprocess.check_output(['powershell.exe','-NoProfile','-Command',
            f'(Get-Partition -DriveLetter {drive}).DiskNumber'],text=True).strip()))
    assert len(set(disks))==2,'The backup must be on a distinct physical disk'
    def copy(source,target,expected=None):
        h=sha(source)
        if expected is not None:assert h==expected
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
        assert sha(target)==h,'Existing backup differs: '+str(target)
        return dict(file=source.name,bytes=source.stat().st_size,sha256=h)
    cases=[]
    for done_path in sorted((PUBLIC/'cases').glob('*/complete.json')):
        done=read(done_path);case=done['job']['case']
        entries=[]
        for name,h in done['files'].items():entries.append(copy(RAW/'cases'/case/name,destination/'cases'/case/name,h))
        for name,h in done['metadata_sha256'].items():copy(done_path.parent/name,destination/'metadata'/case/name,h)
        copy(done_path,destination/'metadata'/case/'complete.json')
        cases.append(dict(case=case,files=entries,completion_sha256=sha(done_path)))
    for name in ('plan.json','pose_packets.npz','zero_adapter_bridge.json'):
        copy(PUBLIC/name,destination/'registration'/name)
    report=dict(verified_physical_drives=2,completed_cases=cases)
    if complete:
        plan=read(PUBLIC/'plan.json');seal=read(PUBLIC/'predictions.json')
        assert {c['case'] for c in cases}=={j['case'] for j in plan['jobs']}|{'two_pair_smoke'}
        assert len(seal['rows'])==105
        hashes=sorted({r['prediction'] for r in seal['rows']})
        files=[copy(RAW/'points'/(h+'.npz'),destination/'points'/(h+'.npz')) for h in hashes]
        report.update(files=files,total_bytes=sum(e['bytes'] for e in files),planned_reconstructions=105)
        dump(PUBLIC/'archive.json',report)
    else:
        dump(ROOT/'cloud/neural_transfer_checkpoint.json',report)
    print('Verified',len(cases),'completed case archives on two physical disks.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--complete',action='store_true')
    args=p.parse_args();archive(args.destination,args.complete)
