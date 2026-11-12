"""Seal and score the three nonexact repeat outputs with the unchanged surface protocol."""
import argparse
import hashlib
import numpy as np
from replay_mast3r_neural_repeat import PUBLIC, RAW, PRIMARY, ROOT, read, sha, dump
from score_mast3r_transfer import resample
from neighbor_sparse_depth import point_map, tensor_digest, evaluate_points, REFS
from prepare_dense_focal_pilot import write_arrays


def points(post):
    with np.load(PUBLIC/'outputs'/(post+'.npz')) as z:
        depth, intrinsic = resample(z['depth'], z['intrinsics'].astype(float))
        extrinsic = np.linalg.inv(z['c2w'].astype(float))[:, :3, :]
    return point_map(depth, intrinsic, extrinsic, np.ones(len(depth)))


def predict():
    result = read(PUBLIC/'result.json')
    assert not result['all_native_exact'], 'Exact outputs do not need repeated surface evaluation'
    (RAW/'points').mkdir(parents=True, exist_ok=True)
    rows = []
    for post in ('ordinary', 'KD', 'KDP_clean'):
        path = PUBLIC/'outputs'/(post+'.npz')
        assert sha(path) == result['files'][post+'.npz']
        cloud = points(post)
        digest = tensor_digest(cloud)
        write_arrays(RAW/'points'/(digest+'.npz'), dict(points=cloud))
        rows.append(dict(post=post, prediction=digest, native_sha256=sha(path)))
    seal = dict(repeat_result_sha256=sha(PUBLIC/'result.json'), primary_seal_sha256=sha(PRIMARY/'predictions.json'),
        source_sha256=hashlib.sha256((ROOT/'scripts/score_mast3r_neural_repeat.py').read_text(encoding='utf-8').encode()).hexdigest(), rows=rows)
    if (PUBLIC/'predictions.json').exists():
        assert read(PUBLIC/'predictions.json') == seal
    dump(PUBLIC/'predictions.json', seal)
    print('Sealed all three repeat reconstructions before surface scoring.')


def compare():
    seal = read(PUBLIC/'predictions.json')
    assert seal['repeat_result_sha256'] == sha(PUBLIC/'result.json')
    assert seal['primary_seal_sha256'] == sha(PRIMARY/'predictions.json')
    assert seal['source_sha256'] == hashlib.sha256((ROOT/'scripts/score_mast3r_neural_repeat.py').read_text(encoding='utf-8').encode()).hexdigest()
    actual = read(PUBLIC/'metrics.json')
    original = read(PRIMARY/'metrics.json')['heads_base']
    first = {r['post']:r for r in read(PRIMARY/'predictions.json')['rows'] if r['case']=='heads_centered_J'}
    comparisons = {}
    for row in seal['rows']:
        assert sha(PUBLIC/'outputs'/(row['post']+'.npz')) == row['native_sha256']
        assert tensor_digest(points(row['post'])) == row['prediction']
        fields = {}
        for ref in REFS:
            a, b = original[first[row['post']]['prediction']][ref], actual[row['prediction']][ref]
            fields[ref] = dict(original=a, repeat=b, relative_change={m:b[m]/a[m]-1 for m in ('accuracy_mean_m', 'completeness_mean_m')},
                combined_relative_change=float(np.mean([b[m]/a[m] for m in ('accuracy_mean_m', 'completeness_mean_m')]))-1)
            observed = read(PUBLIC/'observations.json')[row['prediction']][ref]
            assert observed['identity']['prediction'] == row['prediction'] and observed['identity']['reference'] == ref
            assert observed['metrics'] == b
        comparisons[row['post']] = fields
    return dict(predictions_sha256=sha(PUBLIC/'predictions.json'), comparisons=comparisons,
        scope='One same-seed fresh-process pair on one development case. Changes are descriptive, not a population repeatability estimate or confidence interval.')


def score():
    seal = read(PUBLIC/'predictions.json')
    _, measured = evaluate_points('heads_base', {r['prediction'] for r in seal['rows']}, RAW)
    dump(PUBLIC/'metrics.json', measured)
    observed = {h:{ref:read(RAW/'evaluation/heads_base'/h/ref/'checked.json') for ref in REFS} for h in measured}
    dump(PUBLIC/'observations.json', observed)
    if (PRIMARY/'metrics.json').exists():
        dump(PUBLIC/'metric_comparison.json', compare())
    print('Scored all three repeat reconstructions against both references.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('predict', 'score', 'compare', 'verify'))
    action = p.parse_args().action
    if action == 'predict':
        predict()
    elif action == 'score':
        score()
    elif action == 'compare':
        dump(PUBLIC/'metric_comparison.json', compare())
    else:
        assert read(PUBLIC/'metric_comparison.json') == compare()
        print('Replayed all repeat point-map identities and six paired surface comparisons.')
