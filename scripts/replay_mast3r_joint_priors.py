"""Replay native-depth projection and all MASt3R joint-prior comparison tables."""
import argparse
import hashlib
import numpy as np
from replay_mast3r_transfer import independently_resize
from score_mast3r_joint_priors import PUBLIC,RAW,ROOT,OLD,REFS,read,sha,dump,write_arrays,summarize
from neighbor_sparse_depth import tensor_digest
from motion_reliability_surface import verify_map_archive


def replay(export=False):
    seal=read(PUBLIC/'predictions.json')
    plan=read(PUBLIC/'alignment_plan.json')
    assert seal['plan_sha256']==sha(PUBLIC/'alignment_plan.json')
    assert len(plan['jobs'])==len(seal['rows'])==21
    assert [r['run_id'] for r in seal['rows']]==[j['run_id'] for j in plan['jobs']]
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==digest
    assert seal['historical_scores_sha256']==sha(OLD/'scores.json')
    assert seal['historical_observations_sha256']==sha(OLD/'observations.json')
    assert seal['scoring_source_sha256']==hashlib.sha256((ROOT/'scripts/score_mast3r_joint_priors.py').read_text(encoding='utf-8').encode()).hexdigest()
    assert read(PUBLIC/'disabled_hook_bridge.json')['pass_all']
    assert read(PUBLIC/'joint_gradient_probe.json')['gradient_and_output_check_passed']
    samples={}
    inventory={}
    for row in seal['rows']:
        case,key=row['run_id'],row['input_key']
        folder=PUBLIC/'full'/case
        completion=read(folder/'complete.json')
        capture=read(folder/(key+'.json'))
        trace=read(folder/'prior_trace.json')
        assert completion['plan_sha256']==seal['plan_sha256']
        assert completion['capture_sha256']==sha(folder/(key+'.json'))
        assert completion['trace_sha256']==sha(folder/'prior_trace.json')
        assert capture['input_sha256']==plan['rgb_sha256'][row['scene']]
        assert capture['pairs']==20 and capture['alignment_settings']['niter1']==300 and capture['alignment_settings']['niter2']==300
        assert trace['job']==completion['job']
        if row['arm']=='standard':
            assert trace['rows']==[]
        else:
            assert len(trace['rows'])==300 and len(trace['native_sample_counts'])==5
            assert min(trace['native_sample_counts'])>=8
            assert [v['iteration'] for v in trace['rows']]==list(range(300))
            assert all(v['finite_gradient'] and v['received_gradient']==1 for v in trace['rows'])
        native=PUBLIC/'native_outputs'/(case+'.npz')
        assert sha(native)==row['native_sha256']==completion['native_sha256']==capture['output_sha256']
        with np.load(native) as z:
            if row['arm']=='standard':
                with np.load(OLD/'native_outputs'/(key+'.npz')) as old:
                    repeated={name:bool(np.array_equal(z[name],old[name])) for name in z}
                assert repeated==seal['baseline_exact_repeat'][key] and all(repeated.values())
            np.testing.assert_array_equal((z['depth']<=0).sum(axis=(1,2)),capture['nonpositive_depth_pixels'])
            depth,k=independently_resize(z['depth'],z['intrinsics'])
            c2w=z['c2w'].astype(float)
        y,x=np.indices(depth.shape[1:],dtype=float)
        pixels=np.stack([x,y,np.ones_like(x)],axis=-1)[::9,::9]
        points=[]
        for i in range(len(depth)):
            rays=pixels@np.linalg.inv(k[i]).T
            camera=rays*depth[i,::9,::9,None]
            points.append(camera@c2w[i,:3,:3].T+c2w[i,:3,3])
        coords=np.stack(points)
        if export:
            with np.load(RAW/'points'/(row['prediction']+'.npz')) as z:
                full=z['points']
            assert tensor_digest(full)==row['prediction']
            samples[case]=full[:,::9,::9]
            expected=samples[case]
        else:
            with np.load(PUBLIC/'replay_points.npz') as z:
                expected=z[case]
        np.testing.assert_allclose(coords,expected,rtol=1e-5,atol=2e-5)
        inventory.setdefault(key,{})[case]=row['prediction']
    if export:
        write_arrays(PUBLIC/'replay_points.npz',samples)
    verify_map_archive(PUBLIC/'archive.json',inventory)
    observations=read(PUBLIC/'observations.json')
    metrics=read(PUBLIC/'metrics.json')
    for key,hashes in inventory.items():
        assert set(observations[key])==set(metrics[key])==set(hashes.values())
        for h,refs in observations[key].items():
            assert set(refs)==set(REFS)
            for ref,v in refs.items():
                assert v['identity']['prediction']==h and v['identity']['reference']==ref
                assert v['metrics']==metrics[key][h][ref]
    scores=read(PUBLIC/'scores.json')
    assert scores['seal_sha256']==sha(PUBLIC/'predictions.json')
    assert scores['grids']==summarize(seal,metrics,read(OLD/'scores.json'))
    print('Replayed 21 native reconstructions, 42 surface records and both joint-prior decisions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    replay(p.parse_args().export)
