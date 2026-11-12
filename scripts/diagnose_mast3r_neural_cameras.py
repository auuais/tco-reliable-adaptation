"""Report nominal calibration and gauge-invariant relative-camera diagnostics after sealing."""
import csv
import numpy as np
from scipy.spatial.transform import Rotation
from score_mast3r_neural_transfer import PUBLIC,read,dump,sha


def proper_rotation(matrix):
    u,_,vh=np.linalg.svd(matrix)
    correction=np.eye(3);correction[2,2]=np.linalg.det(u@vh)
    return u@correction@vh


def relative_errors(predicted,reference):
    predicted=np.linalg.inv(predicted[0])@predicted
    reference=np.linalg.inv(reference[0])@reference
    rotation=[];direction=[]
    for p,r in zip(predicted[1:],reference[1:]):
        a,b=proper_rotation(p[:3,:3]),proper_rotation(r[:3,:3])
        rotation.append(float(np.rad2deg(Rotation.from_matrix(a.T@b).magnitude())))
        pn,rn=np.linalg.norm(p[:3,3]),np.linalg.norm(r[:3,3])
        if min(pn,rn)>1e-8:
            cosine=np.clip(np.dot(p[:3,3],r[:3,3])/(pn*rn),-1,1)
            direction.append(float(np.rad2deg(np.arccos(cosine))))
    return dict(relative_rotation_mean_deg=float(np.mean(rotation)),
                translation_direction_mean_deg=float(np.mean(direction)) if direction else None,
                valid_translation_directions=len(direction),total_relative_cameras=len(rotation))


def run():
    seal=read(PUBLIC/'predictions.json');plan=read(PUBLIC/'plan.json')
    assert seal['plan_sha256']==sha(PUBLIC/'plan.json')
    k=np.asarray(plan['intrinsic']);rows=[]
    with np.load(PUBLIC/'pose_packets.npz') as reference:
        for row in seal['rows']:
            path=PUBLIC/'native_outputs'/(row['tag']+'.npz')
            assert sha(path)==row['native_sha256']
            with np.load(path) as z:
                c2w=z['c2w'].astype(float);intrinsics=z['intrinsics'].astype(float)
            focal=intrinsics[:,[0,1],[0,1]]
            target=k[[0,1],[0,1]]
            entry=dict(tag=row['tag'],scene=row['scene'],grid=row['grid'],arm=row['arm'],post=row['post'],
                mean_absolute_relative_focal_percent=float(100*np.mean(np.abs(focal/target-1))),
                mean_principal_error_px=float(np.linalg.norm(intrinsics[:,:2,2]-k[:2,2],axis=1).mean()))
            for condition in ('clean','noisy'):
                metrics=relative_errors(c2w,np.linalg.inv(reference[row['scene']+'_'+condition]))
                entry.update({condition+'_'+name:value for name,value in metrics.items()})
            rows.append(entry)
    assert len(rows)==105
    dump(PUBLIC/'camera_diagnostics.json',dict(seal_sha256=sha(PUBLIC/'predictions.json'),
        rotation_convention='Relative to camera zero; each rotation projected to SO(3) for angle calculation only.',
        translation_convention='Relative-center direction; invariant to positive model scale; degenerate directions counted explicitly.',
        scope='Nominal benchmark matrices and assigned calibration, not certified physical sensor calibration or SLAM accuracy.',rows=rows))
    with (PUBLIC/'camera_diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Reported calibration and both relative-camera targets for all 105 sealed reconstructions.')


if __name__=='__main__':run()
