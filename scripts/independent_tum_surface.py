"""Independently recompute nine fixed TUM surface checks using float64 alignment and Open3D core KNN."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from evaluation_stability import dump,sha
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from prepare_tum_sparse_packet import RAW
from score_saved_geometry import read
from collect_matched_focal_heads import tensor_digest

ARMS=('frozen','centered_guarded','shifted_guarded')


def similarity(source,target):
    x=np.asarray(source,dtype=float)
    y=np.asarray(target,dtype=float)
    cx,cy=x.mean(axis=0),y.mean(axis=0)
    xc,yc=x-cx,y-cy
    u,s,vt=np.linalg.svd(yc.T@xc/len(x))
    sign=np.ones(3)
    sign[-1]=np.sign(np.linalg.det(u@vt))
    rotation=(u*sign)@vt
    scale=(s*sign).sum()/np.mean(np.sum(xc*xc,axis=1))
    return scale*(x@rotation.T)+(cy-scale*cx@rotation.T)


def check(sequence):
    import open3d as o3d
    o3d.utility.random.seed(42)
    seal=read(PUBLIC/'predictions.json')
    refs=read(PUBLIC/'references.json')['sequences'][sequence]
    path=RAW/'references'/refs['file']
    assert sha(path)==refs['sha256']
    with np.load(path) as z:
        truth=z['points']
        mask=z['valid_mask']
    target=truth[mask].astype(float)
    expected=read(PUBLIC/'observations.json')[sequence]
    records={}
    for arm in ARMS:
        h=seal['batches'][sequence]['hashes'][arm]
        with np.load(RAW/'points'/(h+'.npz')) as z:
            points=z['points']
        assert tensor_digest(points)==h
        aligned=similarity(points[mask],target)
        predicted_cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
        truth_cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
        registered=o3d.pipelines.registration.registration_icp(predicted_cloud,truth_cloud,.1,np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        predicted_cloud.transform(registered.transformation)
        predicted_cloud.estimate_normals()
        truth_cloud.estimate_normals()
        predicted=np.asarray(predicted_cloud.points)
        pn,tn=np.asarray(predicted_cloud.normals),np.asarray(truth_cloud.normals)
        measured=[]
        normals=[]
        for dataset,query,data_normals,query_normals in ((target,predicted,tn,pn),(predicted,target,pn,tn)):
            index=o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(dataset,dtype=o3d.core.Dtype.Float64))
            index.knn_index()
            neighbors,squared=index.knn_search(o3d.core.Tensor(query,dtype=o3d.core.Dtype.Float64),1)
            ids=neighbors.numpy().ravel()
            measured.append(float(np.sqrt(squared.numpy()).mean()))
            normals.append(float(np.abs(np.einsum('ij,ij->i',data_normals[ids],query_normals)).mean()))
        observed=dict(accuracy_mean_m=measured[0],completeness_mean_m=measured[1],normal_consistency_mean=float(np.mean(normals)))
        reference=expected[h]['metrics']
        checks={m:bool(np.isclose(observed[m],reference[m],rtol=.002,atol=.00002)) for m in ('accuracy_mean_m','completeness_mean_m')}
        checks['normal_consistency_mean']=bool(abs(observed['normal_consistency_mean']-reference['normal_consistency_mean'])<=.002)
        records[arm]=dict(prediction=h,observed=observed,canonical={m:reference[m] for m in observed},checks=checks,
                          pass_all=all(checks.values()),icp_rmse=float(registered.inlier_rmse))
        print(sequence,arm,checks,flush=True)
    return sequence,records


def run():
    if not (PUBLIC/'scores.json').exists():
        raise FileNotFoundError('Complete the cross-dataset study before independent verification')
    with ProcessPoolExecutor(3) as pool:
        records=dict(pool.map(check,SEQUENCES))
    output=dict(score_sha256=sha(PUBLIC/'scores.json'),prediction_sha256=sha(PUBLIC/'predictions.json'),
                cases=records,pass_all=all(r['pass_all'] for cases in records.values() for r in cases.values()))
    dump(PUBLIC/'independent_surface.json',output)
    print('Independent surface verification:',output['pass_all'],flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
