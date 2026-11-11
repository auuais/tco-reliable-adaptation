"""Publish the fixed supplied held-out trajectory, checking original input identities."""
import argparse
from pathlib import Path

import numpy as np
from run_heldout_sparse_depth import PUBLIC,ROOT,read,sha,write_arrays


def export(assets):
    calibration=np.asarray(read(ROOT/'manifests/sensor_frame_v14.json')['calibration'])
    values={}
    for scene,entry in read(PUBLIC/'input_index.json')['sequences'].items():
        poses=[]
        for frame in entry['targets']:
            path=assets/scene/f'frame-{frame:06d}.pose.txt'
            assert sha(path)==entry['files'][str(frame)]['pose.txt']
            poses.append(calibration@np.linalg.inv(np.loadtxt(path)))
        values[scene.replace('/','_')]=np.stack(poses)
    write_arrays(PUBLIC/'supplied_poses.npz',values)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    export(p.parse_args().assets)
