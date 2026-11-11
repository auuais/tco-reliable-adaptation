"""Verify that independent checks, model repeats and normal-score limitations are reported consistently."""
import numpy as np

from evaluation_stability import sha
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from score_saved_geometry import read


def verify():
    seal=PUBLIC/'predictions.json'
    scores=PUBLIC/'scores.json'
    independent=read(PUBLIC/'independent_surface.json')
    assert independent['prediction_sha256']==sha(seal) and independent['score_sha256']==sha(scores)
    assert set(independent['cases'])==set(SEQUENCES)
    normal_failures=0
    primary_passes=0
    flags=[]
    predictions=read(seal)['batches']
    for scene,cases in independent['cases'].items():
        assert set(cases)=={'frozen','centered_guarded','shifted_guarded'}
        for arm,case in cases.items():
            assert case['prediction']==predictions[scene]['hashes'][arm]
            computed={}
            for metric in ('accuracy_mean_m','completeness_mean_m'):
                computed[metric]=bool(np.isclose(case['observed'][metric],case['canonical'][metric],rtol=.002,atol=.00002))
            computed['normal_consistency_mean']=bool(abs(case['observed']['normal_consistency_mean']-case['canonical']['normal_consistency_mean'])<=.002)
            assert case['checks']==computed and case['pass_all']==all(computed.values())
            flags.append(case['pass_all'])
            primary_passes+=int(computed['accuracy_mean_m'] and computed['completeness_mean_m'])
            normal_failures+=int(not computed['normal_consistency_mean'])
    assert independent['pass_all']==all(flags)
    repeat=read(PUBLIC/'repeat_capture.json')
    assert repeat['score_sha256']==sha(scores) and repeat['input_index_sha256']==sha(PUBLIC/'input_index.json')
    exact=[]
    for scene,r in repeat['sequences'].items():
        assert r['original_archive_sha256']==read(PUBLIC/'captures'/(scene+'.json'))['archive_sha256']
        assert r['repeat_archive_sha256']==read(PUBLIC/'repeat_capture/captures'/(scene+'.json'))['archive_sha256']
        assert set(r['arrays'])=={'depth','intrinsics','extrinsics','native'}
        for record in r['arrays'].values():
            is_exact=record['original_tensor_sha256']==record['repeated_tensor_sha256']
            assert record['exact']==is_exact
            if is_exact:
                assert record['max_absolute_difference']==record['mean_absolute_difference']==record['different_values']==0
            exact.append(is_exact)
    assert repeat['all_exact']==all(exact)
    for filename in ('normal_factorial.json','normal_solver_factorial.json','normal_neighborhoods.json'):
        diagnostic=read(PUBLIC/filename)
        assert diagnostic['prediction_sha256']==sha(seal)
        assert diagnostic['original_observations_sha256']==sha(PUBLIC/'observations.json')
        assert diagnostic['independent_sha256']==sha(PUBLIC/'independent_surface.json')
        assert set(diagnostic['cases'])==set(SEQUENCES)
        for scene,r in diagnostic['cases'].items():
            assert r['prediction']==predictions[scene]['hashes']['frozen']
            for alignment in ('canonical_alignment','float64_alignment'):
                measurement=r['measurements'][alignment]
                for metric in ('accuracy_mean_m','completeness_mean_m','normal_consistency_mean'):
                    a,b=(measurement[backend][metric] for backend in ('scipy','open3d_core'))
                    assert np.isfinite([a,b]).all()
                    np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10)
            if filename=='normal_neighborhoods.json':
                neighborhood=r['measurements']['normal_neighborhoods']
                assert neighborhood['sample_count']==4096
                for metric in ('mean_neighbor_overlap','changed_neighborhood_fraction','mean_absolute_normal_dot','normal_dot_below_099_fraction'):
                    assert 0<=neighborhood[metric]<=1
                assert 0<=neighborhood['mean_point_displacement_m']<=neighborhood['max_point_displacement_m']
    print(f'Verified reporting of {primary_passes}/9 independent distance checks, {normal_failures} normal discrepancies, and {sum(exact)}/{len(exact)} exact repeated tensors.')


if __name__=='__main__':
    verify()
