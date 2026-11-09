"""Package a fully validated screen; never mark incomplete runs as completed."""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,'scripts/aggregate_repeated_screen.py'],cwd=ROOT,check=True,
               stdout=subprocess.DEVNULL)
summary=json.loads((ROOT/'reports/diagnostics_v3_repeats/summary.json').read_text())
assert summary['status']=='complete' and summary['completed_jobs']==24
archive=pathlib.Path('/workspace/tco_diagnostics_v3_repeats.tar.gz')
assert not archive.exists(),'Use a new archive name instead of overwriting evidence'
paths=sorted(ROOT.glob('runs/diag3_*'))+sorted(ROOT.glob('runs/logs/diag3_*'))
paths += [ROOT/'runs/logs/diagnostics_v3_repeats.log',ROOT/'runs/diagnostic_inputs',
          ROOT/'reports/diagnostics_v3_repeats',ROOT/'manifests/diagnostics_v3_repeats.json',
          ROOT/'environment/diagnostics_v3_launch.json',ROOT/'environment/diagnostics_v2_gpu.csv']
paths += [ROOT/'runs'/f'diag2_{scene}_bridge_clean_s0' for scene in ['heads_seq-01','chess_seq-03']]
paths += [ROOT/'scripts'/name for name in ['run_repeated_screen.py','aggregate_repeated_screen.py',
          'package_repeated_screen.py','run_diagnostic.py','prior_corruptions.py']]
paths += [ROOT/'configs/diagnostic_profiles.json',ROOT/'configs/7scenes_baseline.yaml']
with tarfile.open(archive,'w:gz') as t:
    for p in paths:
        assert p.exists(),p
        t.add(p,arcname=str(p.relative_to(ROOT)))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.sha256').write_text(digest+'  '+archive.name+'\n')
print(json.dumps(dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size),indent=2))
