"""Validate sealed TUM inference inputs and disjoint target/source depth identities."""
import numpy as np

from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from prepare_tum_sparse_packet import GRIDS,camera
from evaluation_stability import sha
from score_saved_geometry import read


def verify():
    selection=read(PUBLIC/'inference_selection.json')
    evaluation=read(PUBLIC/'evaluation_selection.json')
    record=read(PUBLIC/'inputs.json')
    assert record['selection_sha256']==sha(PUBLIC/'inference_selection.json')
    assert record['packet_sha256']==sha(PUBLIC/'measurements.npz')
    assert set(selection)==set(evaluation)==set(record['sequences'])==set(SEQUENCES)
    _,intrinsic,_=camera()
    total=0
    with np.load(PUBLIC/'measurements.npz') as arrays:
        expected=set()
        for sequence,entry in selection.items():
            saved=record['sequences'][sequence]
            assert entry['download_sha256']==sha(PUBLIC/(sequence+'_download.json'))
            targets=entry['targets']
            assert len(targets)==len(evaluation[sequence])==len(saved['views'])==5
            label_files={r['depth_file'] for r in evaluation[sequence]}
            sensor_files={s['depth_file'] for t in targets for s in t['sources']}
            rgb_files={t['rgb_file'] for t in targets}
            assert sensor_files.isdisjoint(label_files)
            assert set(saved['files'])==sensor_files|rgb_files
            assert all(len(v)==64 for v in saved['files'].values())
            for suffix in ('poses','intrinsics'):
                expected.add(sequence+'_'+suffix)
            poses=arrays[sequence+'_poses']
            assert poses.shape==(5,4,4)
            for i,t in enumerate(targets):
                np.testing.assert_allclose(poses[i]@np.asarray(t['c2w']),np.eye(4),atol=1e-12)
            np.testing.assert_array_equal(arrays[sequence+'_intrinsics'],np.repeat(intrinsic[None],5,axis=0))
            for i,t in enumerate(targets):
                assert len(t['sources'])==2
                assert {s['association_index'] for s in t['sources']}=={t['association_index']-4,t['association_index']+4}
                for grid,origin in GRIDS.items():
                    key=f'{sequence}_{grid}_{i}'
                    expected.add(key)
                    v=arrays[key]
                    assert v.shape==(saved['views'][i]['counts'][grid],7) and len(v)>=8
                    assert np.isfinite(v).all() and (v[:,2]>0).all()
                    assert ((v[:,:2]>=0)&(v[:,:2]<[518,392])).all()
                    assert set(v[:,3]).issubset({s['depth_index'] for s in t['sources']})
                    assert (v[:,4]%80==origin[0]).all() and (v[:,5]%60==origin[1]).all()
                    total+=len(v)
            assert all(0<=r['rgb_depth_delta_seconds']<=.02+1e-9 for r in evaluation[sequence])
        assert set(arrays.files)==expected
    print('Verified fifteen TUM targets, thirty source depths, two disjoint grids and',total,'usable projected samples.')


if __name__=='__main__':
    verify()
