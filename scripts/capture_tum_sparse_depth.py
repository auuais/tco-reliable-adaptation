"""Prepare TUM frozen VGGT inference using the held-out collector and the same model settings."""
import argparse
from pathlib import Path

from capture_heldout_sparse_depth import capture
from evaluation_stability import dump,sha
from prepare_dense_focal_pilot import ROOT
from prepare_tum_sparse_packet import RAW
from prepare_tum_sparse_depth import PUBLIC
from score_saved_geometry import read
from verify_tum_sparse_inputs import verify


def prepare():
    verify()
    source=read(PUBLIC/'inputs.json')['sequences']
    targets=read(PUBLIC/'inference_selection.json')
    sequences={}
    for scene,entry in source.items():
        sequences[scene]=dict(targets=[t['association_index'] for t in targets[scene]['targets']],
            rgb_file=entry['rgb_file'],rgb_sha256=entry['rgb_sha256'],rgb_tensor_sha256=entry['rgb_tensor_sha256'])
    record=dict(inference_inputs_sha256=sha(PUBLIC/'inputs.json'),sequences=sequences)
    path=PUBLIC/'input_index.json'
    if path.exists() and read(path)!=record:
        raise ValueError('Changed frozen inference index')
    dump(path,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    prepare()
    if args.run:
        scores=ROOT/'results/heldout_sparse_depth/scores.json'
        if not scores.exists():
            p.error('Held-out scoring is still running. Prepared inference inputs are ready.')
        if not all(v['advance'] for v in read(scores)['grids'].values()):
            p.error('The frozen method failed its held-out advancement checks. Retain the prepared inputs for a redesign.')
        if args.weights is None:
            p.error('--weights is required for model inference')
        capture(None,args.weights,public=PUBLIC,raw=RAW,rgb_directory=RAW/'inference')
    else:
        print('TUM model inference prepared. Both held-out grid decisions are required before --run.')
