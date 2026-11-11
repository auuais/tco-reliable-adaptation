"""Check all reserved windows and seal source RGB, poses and neighboring depth identities."""
import argparse
from pathlib import Path

from evaluation_stability import dump,sha
from prepare_dense_focal_pilot import ROOT
from score_saved_geometry import read


def stage(assets):
    policy=read(ROOT/'manifests/holdout_policy_v1.json')
    protocol=ROOT/'third_party/protocol/7scenes_kf200.json'
    windows=read(protocol)
    sequences={}
    for scene,frames in windows.items():
        if scene.split('/')[0] not in policy['heldout_scenes']:
            continue
        targets=[f+25 for f in frames]
        sources=sorted({f+d for f in targets for d in (-4,4)})
        if set(sources)&set(targets):
            raise ValueError('Target/source overlap')
        observations={}
        for frame in sorted(set(targets)|set(sources)):
            kinds=('color.png','pose.txt') if frame in targets else ('depth.proj.png','pose.txt')
            observations[str(frame)]={suffix:sha(assets/scene/f'frame-{frame:06d}.{suffix}') for suffix in kinds}
        sequences[scene]=dict(targets=targets,sources=sources,files=observations)
    assert {s.split('/')[0] for s in sequences}==set(policy['heldout_scenes'])
    output=ROOT/'results/heldout_sparse_depth/input_index.json'
    result=dict(protocol_sha256=sha(protocol),policy_sha256=sha(ROOT/'manifests/holdout_policy_v1.json'),sequences=sequences)
    if output.exists() and read(output)!=result:
        raise ValueError('Held-out inference input identity changed')
    dump(output,result)
    print(len(sequences),'sequences,',sum(len(r['targets']) for r in sequences.values()),'target views')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    stage(p.parse_args().assets)
