"""Verify the registered denominator, pose-channel perturbations and source identities."""
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_dense_focal_pilot import ROOT
from evaluation_stability import sha
from score_saved_geometry import read

PUBLIC=ROOT/'results/mast3r_neural_transfer'


def verify():
    plan=read(PUBLIC/'plan.json')
    for name,h in plan['sources'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==h
    assert plan['pose_packets_sha256']==sha(PUBLIC/'pose_packets.npz')
    assert plan['original_poses_sha256']==sha(ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz')
    assert len(plan['jobs'])==33 and sum(len(j['post']) for j in plan['jobs'])==plan['reconstructions']==105
    assert sum(j['arm']!='zero' for j in plan['jobs'])==24
    assert len({j['case'] for j in plan['jobs']})==33
    assert len(plan['schedule'])==40
    assert all(len(set(row))==3 and all(0<=i<5 for i in row) for row in plan['schedule'])
    first=[tuple(row[:2]) for row in plan['schedule']]
    assert set(first)=={(i,j) for i in range(5) for j in range(5) if i!=j}
    assert all(first.count(pair)==2 for pair in set(first))
    assert plan['arms']==dict(D=[0.,0.],K=[.01,0.],J=[.01,.01],N=[.01,.01])
    np.testing.assert_allclose(plan['intrinsic'],[[470.4,0,255.9],[0,470.4,191.9],[0,0,1]],atol=1e-10,rtol=0)
    for slug in ('heads','chess','fire'):
        assert plan['rgb_sha256'][slug]==read(ROOT/'results/mast3r_transfer'/(slug+'_base.json'))['input_sha256']
        for grid in ('centered','shifted'):
            jobs=[j for j in plan['jobs'] if j['scene']==slug and j['grid']==grid]
            assert {j['arm'] for j in jobs}=={'zero','D','K','J','N'}
            for j in jobs:
                expected=(['KD','KDP_clean','KDP_noisy'] if j['arm']=='zero' else
                          ['ordinary','KD','KDP_clean','KDP_noisy'] if j['arm'] in ('D','K') else
                          ['ordinary','KD','KDP_clean'] if j['arm']=='J' else ['ordinary','KD','KDP_noisy'])
                assert j['post']==expected
    paths={'centered':'neighbor_sparse_depth','shifted':'shifted_sparse_grid'}
    for grid,folder in paths.items():assert plan['packets'][grid]==sha(ROOT/'results'/folder/'measurements.npz')
    with np.load(PUBLIC/'pose_packets.npz') as z,np.load(ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz') as original:
        for slug in ('heads','chess','fire'):
            clean=z[slug+'_clean'];noisy=z[slug+'_noisy']
            np.testing.assert_array_equal(clean,original[slug+'_base'])
            np.testing.assert_allclose(clean[0],noisy[0],atol=1e-12,rtol=0)
            c,n=np.linalg.inv(clean),np.linalg.inv(noisy)
            diameter=np.linalg.norm(c[:,None,:3,3]-c[None,:,:3,3],axis=-1).max()
            np.testing.assert_allclose(np.linalg.norm(n[1:,:3,3]-c[1:,:3,3],axis=1),.02*diameter,rtol=1e-10,atol=1e-12)
            # The supplied registration matrices are not exactly orthogonal.
            # Recover the applied right perturbation with an inverse, not a transpose.
            relative=np.linalg.solve(c[1:,:3,:3],n[1:,:3,:3])
            np.testing.assert_allclose(Rotation.from_matrix(relative).magnitude(),np.deg2rad(5),rtol=1e-6,atol=1e-7)
    bridge=read(PUBLIC/'zero_adapter_bridge.json')
    assert bridge['pass_all'] and all(bridge['training_pair_exact'].values())
    assert all(v['exact'] and v['max_difference']==0 for v in bridge['final_inference_difference'].values())
    print('Verified 33 cases, 24 neural states, 105 planned reconstructions and the fixed camera-error protocol.')
    return plan


if __name__=='__main__':verify()
