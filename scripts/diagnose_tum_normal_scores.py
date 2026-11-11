"""Separate normal-score sensitivity to Sim(3) precision from nearest-neighbor backend effects."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import sys

import numpy as np
from scipy.spatial import cKDTree

from independent_tum_surface import similarity
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from prepare_tum_sparse_packet import RAW
from evaluation_stability import dump,sha
from score_saved_geometry import read
from collect_matched_focal_heads import tensor_digest
from focal_zoom_observations import UPSTREAM


def metrics(points,target,pn,tn,backend):
    import open3d as o3d
    distances=[]
    normals=[]
    for dataset,query,data_normals,query_normals in ((target,points,tn,pn),(points,target,pn,tn)):
        if backend=='scipy':
            distance,indices=cKDTree(dataset).query(query,workers=2)
        else:
            index=o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(dataset,dtype=o3d.core.Dtype.Float64))
            index.knn_index()
            ids,squared=index.knn_search(o3d.core.Tensor(query,dtype=o3d.core.Dtype.Float64),1)
            indices=ids.numpy().ravel()
            distance=np.sqrt(squared.numpy()).ravel()
        distances.append(float(distance.mean()))
        normals.append(float(np.abs(np.einsum('ij,ij->i',data_normals[indices],query_normals)).mean()))
    return dict(accuracy_mean_m=distances[0],completeness_mean_m=distances[1],normal_consistency_mean=float(np.mean(normals)))


def check(sequence):
    import open3d as o3d
    sys.path.insert(0,str(UPSTREAM))
    from evaluation.mv_recon.utils import umeyama
    seal=read(PUBLIC/'predictions.json')
    ref=read(PUBLIC/'references.json')['sequences'][sequence]
    h=seal['batches'][sequence]['hashes']['frozen']
    with np.load(RAW/'points'/(h+'.npz')) as z:
        points=z['points']
    assert tensor_digest(points)==h
    path=RAW/'references'/ref['file']
    assert sha(path)==ref['sha256']
    with np.load(path) as z:
        truth,mask=z['points'],z['valid_mask']
    target=truth[mask].astype(float)
    scale,rotation,translation=umeyama(points[mask].T,truth[mask].T)
    canonical=(scale*np.einsum('nhwj,ij->nhwi',points,rotation)+translation.T)[mask]
    double=similarity(points[mask],target)
    result={}
    for mode,aligned in [('canonical_alignment',canonical),('float64_alignment',double)]:
        o3d.utility.random.seed(42)
        predicted=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
        reference=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
        registered=o3d.pipelines.registration.registration_icp(predicted,reference,.1,np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        predicted.transform(registered.transformation)
        predicted.estimate_normals()
        reference.estimate_normals()
        xyz=np.asarray(predicted.points)
        pn,tn=np.asarray(predicted.normals),np.asarray(reference.normals)
        scores={backend:metrics(xyz,target,pn,tn,backend) for backend in ('scipy','open3d_core')}
        result[mode]=scores
        print(sequence,mode,{b:round(v['normal_consistency_mean'],7) for b,v in scores.items()},flush=True)
    expected=read(PUBLIC/'observations.json')[sequence][h]['metrics']
    result['original']={m:expected[m] for m in ('accuracy_mean_m','completeness_mean_m','normal_consistency_mean')}
    return sequence,dict(prediction=h,measurements=result)


def run():
    with ProcessPoolExecutor(3) as pool:
        records=dict(pool.map(check,SEQUENCES))
    dump(PUBLIC/'normal_factorial.json',dict(prediction_sha256=sha(PUBLIC/'predictions.json'),
        original_observations_sha256=sha(PUBLIC/'observations.json'),independent_sha256=sha(PUBLIC/'independent_surface.json'),
        cases=records))


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
