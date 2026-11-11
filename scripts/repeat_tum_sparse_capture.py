"""Repeat all TUM frozen-model captures in a new process and compare every output tensor."""
import argparse
from pathlib import Path

import numpy as np

from capture_heldout_sparse_depth import capture
from evaluation_stability import dump,sha
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from prepare_tum_sparse_packet import RAW
from score_saved_geometry import read
from collect_matched_focal_heads import tensor_digest


def run(weights):
    if not (PUBLIC/'scores.json').exists():
        raise FileNotFoundError('Complete primary TUM evaluation first')
    repeat_public=PUBLIC/'repeat_capture'
    repeat_raw=RAW/'repeat_capture'
    index=read(PUBLIC/'input_index.json')
    index_path=repeat_public/'input_index.json'
    if index_path.exists() and read(index_path)!=index:
        raise ValueError('Repeat input index differs')
    dump(index_path,index)
    capture(None,weights,public=repeat_public,raw=repeat_raw,rgb_directory=RAW/'inference')
    sequences={}
    for sequence in SEQUENCES:
        original_path=RAW/(sequence+'.npz')
        repeat_path=repeat_raw/(sequence+'.npz')
        assert sha(original_path)==read(PUBLIC/'captures'/(sequence+'.json'))['archive_sha256']
        assert sha(repeat_path)==read(repeat_public/'captures'/(sequence+'.json'))['archive_sha256']
        arrays={}
        with np.load(original_path) as original,np.load(repeat_path) as repeated:
            assert set(original.files)==set(repeated.files)
            for name in original:
                a,b=original[name],repeated[name]
                assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(b).all()
                difference=np.abs(a.astype(float)-b.astype(float))
                arrays[name]=dict(original_tensor_sha256=tensor_digest(a),repeated_tensor_sha256=tensor_digest(b),
                                  exact=bool(np.array_equal(a,b)),max_absolute_difference=float(difference.max()),
                                  mean_absolute_difference=float(difference.mean()),different_values=int(np.count_nonzero(a!=b)))
        sequences[sequence]=dict(original_archive_sha256=sha(original_path),repeat_archive_sha256=sha(repeat_path),arrays=arrays)
    result=dict(score_sha256=sha(PUBLIC/'scores.json'),input_index_sha256=sha(PUBLIC/'input_index.json'),sequences=sequences,
                all_exact=all(v['exact'] for s in sequences.values() for v in s['arrays'].values()))
    dump(PUBLIC/'repeat_capture.json',result)
    print('Fresh-process TUM inference bitwise repeat:',result['all_exact'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    run(p.parse_args().weights)
