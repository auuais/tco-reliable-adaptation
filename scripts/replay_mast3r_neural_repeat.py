"""Publish and independently replay the fixed fresh-process neural repeat comparison."""
import argparse
import hashlib
import shutil
import numpy as np
from verify_mast3r_neural_plan import ROOT, PUBLIC as PRIMARY
from evaluation_stability import dump, sha
from score_saved_geometry import read

PUBLIC = ROOT/'results/mast3r_neural_repeat'
RAW = ROOT/'reports/mast3r_neural_repeat'


def publish(source, target, expected):
    assert sha(source) == expected
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    assert sha(target) == expected


def replay(export=False):
    plan = read(PUBLIC/'plan.json')
    result = read(PUBLIC/'result.json')
    case = plan['original_case']
    original = read(PRIMARY/'cases'/case/'complete.json')
    assert sha(PRIMARY/'plan.json') == plan['primary_plan_sha256']
    assert sha(PRIMARY/'cases'/case/'complete.json') == plan['original_completion_sha256']
    source = ROOT/'scripts/repeat_mast3r_neural_case.py'
    assert hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest() == plan['runner_sha256']
    assert result['plan_sha256'] == sha(PUBLIC/'plan.json')
    assert result['original_state_matches'] == (result['original_state_sha256'] == original['original_state_sha256'])
    training = read(PUBLIC/'training.json')
    initial = read(PRIMARY/'cases'/case/'training.json')
    assert len(training['rows']) == len(initial['rows']) == plan['updates'] == 40
    assert result['initial_gauge_and_support_exact'] == all(training[n] == initial[n] for n in ('gauges', 'units', 'sample_counts'))
    expected = {'adapters.npz', 'gauge_support.npz', 'ordinary.npz', 'KD.npz', 'KDP_clean.npz'}
    assert set(result['files']) == expected
    for name, digest in result['files'].items():
        if export:
            publish(RAW/name, PUBLIC/'outputs'/name, digest)
        assert sha(PUBLIC/'outputs'/name) == digest
    adapter = {}
    with np.load(PUBLIC/'outputs/adapters.npz') as new, np.load(PRIMARY/'adapter_states'/(case+'.npz')) as old:
        assert set(new.files) == set(old.files) and len(new.files) == 288
        assert sum(new[n].size for n in new.files) == 1032192
        for name in new.files:
            assert np.isfinite(new[name]).all()
            adapter[name] = dict(exact=bool(np.array_equal(new[name], old[name])), max_difference=float(np.abs(new[name]-old[name]).max()))
    assert adapter == result['adapter_comparison']
    with np.load(PUBLIC/'outputs/gauge_support.npz') as new, np.load(PRIMARY/'gauge_support'/(case+'.npz')) as old:
        assert set(new.files) == set(old.files)
        for name in new.files:
            np.testing.assert_array_equal(new[name], old[name])
    native = {}
    for post in ('ordinary', 'KD', 'KDP_clean'):
        fields = {}
        with np.load(PUBLIC/'outputs'/(post+'.npz')) as new, np.load(PRIMARY/'native_outputs'/(case+'_'+post+'.npz')) as old:
            assert set(new.files) == set(old.files) == {'depth', 'intrinsics', 'c2w'}
            for name in new.files:
                assert new[name].shape == old[name].shape and np.isfinite(new[name]).all()
                delta = np.abs(new[name].astype(float)-old[name].astype(float))
                fields[name] = dict(exact=bool(np.array_equal(new[name], old[name])), max_difference=float(delta.max()),
                    mean_difference=float(delta.mean()), close=bool(np.allclose(new[name], old[name], rtol=plan['native_rtol'], atol=plan['native_atol'])))
        native[post] = fields
    assert native == result['native_comparison']
    assert result['all_native_exact'] == all(v['exact'] for fields in native.values() for v in fields.values())
    assert result['all_native_close'] == all(v['close'] for fields in native.values() for v in fields.values())
    metadata_names = ('plan.json', 'result.json', 'training.json', 'ordinary.json', 'KD.json', 'KDP_clean.json', 'KD_trace.json', 'KDP_clean_trace.json')
    metadata = {name:sha(PUBLIC/name) for name in metadata_names}
    seal = dict(metadata_sha256=metadata, outputs=result['files'])
    if export and not (PUBLIC/'publication.json').exists():
        dump(PUBLIC/'publication.json', seal)
    assert read(PUBLIC/'publication.json') == seal
    if (PUBLIC/'archive.json').exists():
        archive = read(PUBLIC/'archive.json')
        assert archive['verified_physical_drives'] == 2 and archive['reference_evaluations'] == 6
        assert archive['repeat_result_sha256'] == sha(PUBLIC/'result.json')
        records = archive['files']
        assert len({r['file'] for r in records}) == len(records)
        assert archive['total_bytes'] == sum(r['bytes'] for r in records)
        inventory = {r['file']:r for r in records}
        for name, digest in result['files'].items():
            assert inventory['raw/'+name]['sha256'] == digest
        for name, digest in metadata.items():
            assert inventory['public/'+name]['sha256'] == digest
        assert all(r['bytes'] > 0 and len(r['sha256']) == 64 for r in records)
    print('Replayed one fresh-process numerical repeat: 288 adapter tensors, fixed gauges/support and nine native arrays.')
    print('All native arrays exact:', result['all_native_exact'], '; within registered tolerance:', result['all_native_close'])
    print('Nonexact outputs require surface-metric comparison; this is not a population repeatability estimate.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', action='store_true')
    replay(p.parse_args().export)
