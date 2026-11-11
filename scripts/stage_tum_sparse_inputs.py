"""Freeze TUM inference timestamps and supplied poses without opening target depth images."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from evaluation_stability import dump,sha
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES


def index_file(path):
    result=[]
    for line in path.read_text().splitlines():
        fields=line.split()
        if fields and not fields[0].startswith('#'):
            result.append((float(fields[0]),fields[1]))
    if any(result[i][0]>=result[i+1][0] for i in range(len(result)-1)):
        raise ValueError('Non-increasing timestamps')
    return result


def associations(rgb,depth):
    times=np.array([r[0] for r in depth])
    candidates=[]
    for i,(time,_) in enumerate(rgb):
        for j in range(np.searchsorted(times,time-.02),np.searchsorted(times,time+.02,side='right')):
            candidates.append((abs(time-times[j]),i,j))
    used_rgb,used_depth=set(),set()
    pairs=[]
    for delta,i,j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            pairs.append((i,j,delta))
            used_rgb.add(i)
            used_depth.add(j)
    return sorted(pairs)


def interpolate(trajectory,time):
    timestamps=trajectory[:,0]
    right=np.searchsorted(timestamps,time,side='right')
    if right==0 or right==len(timestamps):
        return None
    left=right-1
    span=timestamps[right]-timestamps[left]
    if span<=0 or span>.05:
        return None
    alpha=(time-timestamps[left])/span
    pose=np.eye(4)
    pose[:3,3]=(1-alpha)*trajectory[left,1:4]+alpha*trajectory[right,1:4]
    pose[:3,:3]=Slerp(timestamps[[left,right]],Rotation.from_quat(trajectory[[left,right],4:8]))([time]).as_matrix()[0]
    return pose


def stage(destination):
    inference,evaluation={},{}
    for sequence in SEQUENCES:
        folder=destination/('rgbd_dataset_freiburg3_'+sequence)
        recorded=PUBLIC/(sequence+'_download.json')
        if not recorded.exists():
            raise FileNotFoundError('Archive validation pending for '+sequence)
        for name,digest in json.loads(recorded.read_text())['metadata_sha256'].items():
            if sha(folder/name)!=digest:
                raise ValueError('Timestamp metadata changed')
        rgb=index_file(folder/'rgb.txt')
        depth=index_file(folder/'depth.txt')
        trajectory=np.loadtxt(folder/'groundtruth.txt')
        if np.any(np.diff(trajectory[:,0])<=0):
            raise ValueError('Trajectory timestamps are not ordered')
        eligible=[]
        for i,j,delta in associations(rgb,depth):
            rgb_pose=interpolate(trajectory,rgb[i][0])
            depth_pose=interpolate(trajectory,depth[j][0])
            if rgb_pose is not None and depth_pose is not None:
                eligible.append(dict(rgb_index=i,depth_index=j,delta=delta,rgb_pose=rgb_pose,depth_pose=depth_pose))
        if len(eligible)<50:
            raise ValueError('Insufficient timestamp coverage')
        low,high=4,len(eligible)-5
        targets=np.rint(low+(high-low)*np.array([.1,.3,.5,.7,.9])).astype(int).tolist()
        sources=sorted({f+d for f in targets for d in (-4,4)})
        if set(sources)&set(targets) or len(set(targets))!=5:
            raise ValueError('Target and source identities overlap')
        def source(f):
            row=eligible[f]
            j=row['depth_index']
            return dict(association_index=f,depth_index=j,depth_time=depth[j][0],depth_file=depth[j][1],c2w=row['depth_pose'].tolist())
        target_info=[]
        labels=[]
        for f in targets:
            row=eligible[f]
            i,j=row['rgb_index'],row['depth_index']
            target_info.append(dict(association_index=f,rgb_index=i,rgb_time=rgb[i][0],rgb_file=rgb[i][1],
                                    c2w=row['rgb_pose'].tolist(),sources=[source(f-4),source(f+4)]))
            labels.append(dict(association_index=f,depth_file=depth[j][1],depth_time=depth[j][0],c2w=row['depth_pose'].tolist(),
                               rgb_depth_delta_seconds=row['delta']))
        inference[sequence]=dict(download_sha256=sha(recorded),eligible_associations=len(eligible),targets=target_info)
        evaluation[sequence]=labels
    dump(PUBLIC/'inference_selection.json',inference)
    dump(PUBLIC/'evaluation_selection.json',evaluation)
    print('Selected fifteen RGB targets and thirty neighboring source depths without reading depth images.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    stage(p.parse_args().destination)
