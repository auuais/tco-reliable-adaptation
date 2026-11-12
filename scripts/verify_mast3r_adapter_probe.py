"""Check recorded neural-adapter feasibility and independently recompute its sparse loss."""
import hashlib
import numpy as np
from prepare_dense_focal_pilot import ROOT
from evaluation_stability import sha
from score_saved_geometry import read


def verify():
    public=ROOT/'results/mast3r_adapter_probe'
    for attempt in ('decoder_checkpoint01','decoder_checkpoint02'):
        plan=read(public/(attempt+'_plan.json'))
        for name,digest in plan['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==digest
        assert plan['measurements_sha256']==sha(ROOT/'results/neighbor_sparse_depth/measurements.npz')
        assert plan['size']==[384,512] and plan['rank']==4 and plan['updates']==1
    failure=read(public/'decoder_checkpoint01_failure.json')
    assert failure['error_type']=='TypeError' and 'numpy.ndarray' in failure['message']
    assert failure['plan_sha256']==sha(public/'decoder_checkpoint01_plan.json')
    record=read(public/'decoder_checkpoint02.json')
    assert record['plan_sha256']==sha(public/'decoder_checkpoint02_plan.json')
    assert record['original_state_sha256_before']==record['original_state_sha256_after']
    assert len(record['original_state_sha256_before'])==64
    assert len(record['zero_update_exact'])==8 and all(record['zero_update_exact'].values())
    suffixes=('attn.qkv','attn.proj','cross_attn.projq','cross_attn.projk','cross_attn.projv','cross_attn.proj')
    expected={f'{branch}.{i}.{suffix}' for branch in ('dec_blocks','dec_blocks2') for i in range(12) for suffix in suffixes}
    assert set(record['adapter_modules'])==expected and record['trainable_parameters']==24*(4*3072+5*4*1536)
    gradients=record['gradients']
    assert len(gradients)==288 and all(v['finite'] and np.isfinite(v['norm']) for v in gradients.values())
    assert any(v['present'] and v['norm']>0 for v in gradients.values())
    source=public/'decoder_checkpoint02_points.npz'
    assert sha(source)==record['point_archive_sha256']
    with np.load(source) as z:
        before,after=z['before'],z['after']
    assert before.shape==after.shape==(1,384,512,3)
    assert np.isfinite(before).all() and np.isfinite(after).all()
    difference=np.abs(after-before)
    np.testing.assert_allclose([difference.max(),difference.mean()],
                               [record['max_point_change'],record['mean_point_change']],atol=1e-9,rtol=2e-6)
    assert difference.max()>0
    with np.load(ROOT/'results/neighbor_sparse_depth/measurements.npz') as z:
        samples=z['heads_25']
    xy=np.rint((samples[:,:2]+.5)*[512/518,384/392]-.5).astype(int)
    best={}
    for j,((x,y),row) in enumerate(zip(xy,samples)):
        if 0<=x<512 and 0<=y<384 and ((x,y) not in best or row[2]<samples[best[(x,y)],2]):
            best[(x,y)]=j
    x,y=np.array(list(best)).T
    measured=samples[list(best.values()),2].astype(np.float32)
    depths=before[0,y,x,2]
    valid=np.isfinite(depths)&(depths>0)
    measured,depths=measured[valid],depths[valid]
    assert len(depths)==record['sparse_count']==66
    lower_median=lambda a:np.sort(a)[(len(a)-1)//2]
    scale=lower_median(measured/depths)
    unit=lower_median(measured)
    np.testing.assert_allclose([scale,unit],[record['metric_scale'],record['unit_m']],rtol=2e-6)
    residual=(float(scale)*depths.astype(float)-measured)/(.02*float(unit))
    absolute=np.abs(residual)
    loss=np.where(absolute<1,.5*residual**2,absolute-.5).mean()
    np.testing.assert_allclose(loss,record['initial_sparse_loss'],atol=2e-6,rtol=2e-6)
    archive=read(public/'archive.json')
    assert archive['verified_physical_drives']==2 and archive['sha256']==sha(source)
    assert archive['bytes']==source.stat().st_size and record['feasibility_pass']
    print('Verified one preserved setup failure, exact adapter fidelity, finite gradients, sparse loss and changed outputs.')
    print('This checks recorded evidence and output arithmetic, not an independent model rerun or reconstruction quality.')


if __name__=='__main__':
    verify()
