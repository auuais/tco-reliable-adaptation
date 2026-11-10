"""Compare numerical evidence exactly, accounting only for verified CRLF/LF bytes."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
actual = json.loads((ROOT/'reports/calibration_v5/parameterization.json').read_text())
expected = json.loads((ROOT/'results/calibration_v5/parameterization.json').read_text())
mapping = json.loads((ROOT/'results/calibration_v5/source_line_endings.json').read_text())
actual_hashes, expected_hashes = actual.pop('source_sha256'), expected.pop('source_sha256')
assert actual == expected, 'Numerical or interpretation evidence changed'
assert set(actual_hashes) == set(expected_hashes)
for name in actual_hashes:
    info = mapping['files'][name]
    assert expected_hashes[name] == info['local_raw_sha256']
    assert actual_hashes[name] in [info['local_raw_sha256'], info['git_blob_lf_sha256']]
print('Exact numerical evidence and explicitly mapped upstream source hashes verified')
