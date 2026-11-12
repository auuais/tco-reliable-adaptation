"""Replay all neural-transfer evidence, pair gauges and both camera-condition decisions."""
import argparse
import hashlib
import shutil
import numpy as np
from verify_mast3r_neural_plan import verify,PUBLIC,ROOT
from score_mast3r_neural_transfer import RAW,OLD,REFS,read,sha,dump,write_arrays,summarize
from replay_mast3r_transfer import independently_resize
from neighbor_sparse_depth import tensor_digest
from motion_reliability_surface import verify_map_archive


def publish(source,target,expected):
    assert sha(source)==expected
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copyfile(source,target)
    assert sha(target)==expected


def check_training(job,completion,export):
    case=job['case'];folder=PUBLIC/'cases'/case
    if job['arm']=='zero':return
    record=read(folder/'training.json')
    assert record['job']==job and len(record['rows'])==40
    plan=read(PUBLIC/'plan.json')
    iw,pw=plan['arms'][job['arm']]
    for step,(row,triplet) in enumerate(zip(record['rows'],plan['schedule'])):
        i,j,k=triplet
        assert row['step']==step and row['pairs']==[[i,j],[i,k]]
        assert np.isfinite(row['gradient_norm']) and row['gradient_norm']>0
        value=row['depth']+.1*row['consistency']+iw*row['intrinsic']+pw*row['bearing']
        np.testing.assert_allclose(value,row['loss'],rtol=2e-6,atol=2e-6)
    for name,destination in (('adapters.npz',PUBLIC/'adapter_states'/(case+'.npz')),
                             ('gauge_support.npz',PUBLIC/'gauge_support'/(case+'.npz'))):
        if export:publish(RAW/'cases'/case/name,destination,completion['files'][name])
        assert sha(destination)==completion['files'][name]
    with np.load(PUBLIC/'adapter_states'/(case+'.npz')) as z:
        assert len(z.files)==288 and sum(v.size for v in z.values())==1032192
        assert all(np.isfinite(v).all() for v in z.values())
        assert all('lora_A' in name or 'lora_B' in name for name in z.files)
        assert any(np.any(z[name]!=0) for name in z.files if 'lora_B' in name)
    packet='neighbor_sparse_depth' if job['grid']=='centered' else 'shifted_sparse_grid'
    with np.load(ROOT/'results'/packet/'measurements.npz') as measurements,np.load(PUBLIC/'gauge_support'/(case+'.npz')) as z:
        for i,frame in enumerate((25,225,425,625,825)):
            values=measurements[f'{job["scene"]}_{frame}']
            xy=np.rint((values[:,:2]+.5)*[512/518,384/392]-.5).astype(int)
            nearest={}
            for pixel,row in zip(xy,values):
                x,y=pixel
                if 0<=x<512 and 0<=y<384:
                    nearest[(x,y)]=min(nearest.get((x,y),float('inf')),row[2])
            actual=z[f'xy_{i}'].T;measured=z[f'measured_{i}']
            assert len(actual)==len(measured)==record['sample_counts'][i]>=8
            np.testing.assert_array_equal(measured,np.array([nearest[tuple(p)] for p in actual],dtype=np.float32))
            unit=np.sort(measured)[(len(measured)-1)//2]
            np.testing.assert_allclose(unit,record['units'][i],rtol=2e-6)
            for j in range(5):
                if i==j:continue
                initial=z[f'initial_{i}_{j}']
                assert np.isfinite(initial).all() and (initial>0).all()
                ratios=measured/initial
                scale=np.sort(ratios)[(len(ratios)-1)//2]
                np.testing.assert_allclose(scale,record['gauges'][f'{i}_{j}'],rtol=2e-6)
    assert record['original_state_sha256']==completion['original_state_sha256']
    return record


def replay(export=False):
    plan=verify();seal=read(PUBLIC/'predictions.json')
    assert seal['plan_sha256']==sha(PUBLIC/'plan.json')
    assert seal['historical_scores_sha256']==sha(OLD/'scores.json')
    assert seal['historical_observations_sha256']==sha(OLD/'observations.json')
    assert seal['scoring_source_sha256']==hashlib.sha256((ROOT/'scripts/score_mast3r_neural_transfer.py').read_text(encoding='utf-8').encode()).hexdigest()
    expected={(j['case'],post) for j in plan['jobs'] for post in j['post']}
    assert len(seal['rows'])==105 and {(r['case'],r['post']) for r in seal['rows']}==expected
    base_hash=read(PUBLIC/'zero_adapter_bridge.json')['original_state_hash']
    initial_gauges={};completions={}
    for job in plan['jobs']:
        folder=PUBLIC/'cases'/job['case'];done=read(folder/'complete.json')
        assert done['job']==job and done['plan_sha256']==seal['plan_sha256'] and done['original_state_sha256']==base_hash
        for name,h in done['metadata_sha256'].items():assert sha(folder/name)==h
        record=check_training(job,done,export)
        if record:
            group=(job['scene'],job['grid'])
            initial={n:record[n] for n in ('gauges','units','sample_counts')}
            assert initial_gauges.setdefault(group,initial)==initial
        completions[job['case']]=done
    coords={};inventory={}
    for row in seal['rows']:
        tag,key=row['tag'],row['input_key']
        path=PUBLIC/'native_outputs'/(tag+'.npz')
        assert sha(path)==row['native_sha256']==completions[row['case']]['files'][row['post']+'.npz']
        meta=read(PUBLIC/'cases'/row['case']/(row['post']+'.json'))
        assert meta['input_sha256']==plan['rgb_sha256'][row['scene']] and meta['output_sha256']==sha(path)
        assert meta['inference_precision']=='float32' and meta['coarse_iterations']==meta['fine_iterations']==300
        with np.load(path) as z:
            np.testing.assert_array_equal((z['depth']<=0).sum(axis=(1,2)),meta['nonpositive_depth_pixels'])
            if row['arm']=='zero' and row['post']=='ordinary':
                with np.load(ROOT/'results/mast3r_transfer/native_outputs'/(key+'.npz')) as old:
                    equal={n:bool(np.array_equal(z[n],old[n])) for n in z}
                assert equal==seal['baseline_exact'][row['scene']] and all(equal.values())
            depth,k=independently_resize(z['depth'],z['intrinsics']);c2w=z['c2w'].astype(float)
        y,x=np.indices(depth.shape[1:],dtype=float)
        pixels=np.stack([x,y,np.ones_like(x)],axis=-1)[::9,::9]
        point=np.stack([(pixels@np.linalg.inv(k[i]).T*depth[i,::9,::9,None])@c2w[i,:3,:3].T+c2w[i,:3,3] for i in range(5)])
        if export:
            with np.load(RAW/'points'/(row['prediction']+'.npz')) as z:full=z['points']
            assert tensor_digest(full)==row['prediction']
            reference=full[:,::9,::9];coords[tag]=reference
        else:
            with np.load(PUBLIC/'replay_points.npz') as z:reference=z[tag]
        np.testing.assert_allclose(point,reference,rtol=1e-5,atol=2e-5)
        inventory.setdefault(key,{})[tag]=row['prediction']
    if export:write_arrays(PUBLIC/'replay_points.npz',coords)
    verify_map_archive(PUBLIC/'archive.json',inventory)
    observations=read(PUBLIC/'observations.json');metrics=read(PUBLIC/'metrics.json')
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
    print('Replayed 105 reconstructions, 24 neural states, 960 updates and both grids/pose-condition decisions.')
    print('This checks stored evidence and geometry; it is not an independent model or dense-evaluator rerun.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',action='store_true')
    replay(p.parse_args().export)
