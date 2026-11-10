"""Safely extract and independently validate a collected v5 evidence archive."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=pathlib.Path, required=True)
parser.add_argument('--checksum', type=pathlib.Path, required=True)
parser.add_argument('--extract', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
expected = args.checksum.read_text().split()[0]
digest = hashlib.sha256()
with args.archive.open('rb') as source:
    for block in iter(lambda: source.read(8*1024*1024), b''):
        digest.update(block)
assert digest.hexdigest() == expected
assert not args.extract.exists(), 'Never overwrite collected evidence'
args.extract.mkdir(parents=True)
with tarfile.open(args.archive) as archive:
    members = archive.getmembers()
    for member in members:
        name = pathlib.PurePosixPath(member.name)
        assert not name.is_absolute() and '..' not in name.parts
        assert ':' not in member.name and '\\' not in member.name
        assert member.isdir() or member.isfile()
        resolved = (args.extract/member.name).resolve()
        assert resolved == args.extract.resolve() or args.extract.resolve() in resolved.parents
    archive.extractall(args.extract)
assert (args.extract/'environment/prior_ablation_v5_exit_code.txt').read_text().strip() == '0'
assert (args.extract/'manifests/prior_ablation_v5.json').read_bytes() == (ROOT/'manifests/prior_ablation_v5.json').read_bytes()
batch = json.loads((args.extract/'reports/prior_ablation_v5/batch_status.json').read_text())
assert len(batch) == 39 and all(j['status'] == 'complete' for j in batch)
calibration = json.loads((args.extract/'reports/prior_ablation_v5/calibration_remote.json').read_text())
local = json.loads((ROOT/'results/calibration_v5/summary.json').read_text())
for key in ['findings', 'checks', 'audit_sha256', 'upstream_commit']:
    assert calibration[key] == local[key], key
source_map = json.loads((ROOT/'results/calibration_v5/source_line_endings.json').read_text())
assert set(calibration['source_sha256']) == set(local['source_sha256'])
for name, remote_hash in calibration['source_sha256'].items():
    assert local['source_sha256'][name] == source_map['files'][name]['local_raw_sha256']
    assert remote_hash == source_map['files'][name]['git_blob_lf_sha256']
for remote_frame, local_frame in zip(calibration['frames'], local['frames']):
    for key in ['scene', 'frame', 'pose_sha256', 'registered_depth_sha256']:
        assert remote_frame[key] == local_frame[key], key
    assert remote_frame['exact_inverse_roundtrip_max_pixels'] < .001
assert len(calibration['frames']) == len(local['frames']) == 15
subprocess.run([sys.executable, str(ROOT/'scripts/aggregate_prior_ablation.py'),
                '--runs', str(args.extract/'runs'), '--output', str(args.output)], check=True)
summary = json.loads((args.output/'summary.json').read_text())
remote_summary = json.loads((args.extract/'reports/prior_ablation_v5/summary.json').read_text())
assert summary == remote_summary
records = [json.loads((args.extract/'runs'/j['run_id']/'run.json').read_text()) for j in batch]
for record in records:
    for field in ['research_commit', 'gpu_identity', 'gpu', 'versions', 'checkpoint_download']:
        assert record[field] == records[0][field], field
record = dict(status='passed', planned_and_validated_runs=39, adaptation_updates=1440,
              archive_sha256=expected, archive_bytes=args.archive.stat().st_size,
              extraction_members=len(members), code_commit=summary['code_commit'],
              local_remote_aggregates_identical=True, calibration_sources_and_inputs_match=True,
              same_physical_gpu_all_runs=True, gpu=records[0]['gpu'])
(args.output/'local_validation.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
