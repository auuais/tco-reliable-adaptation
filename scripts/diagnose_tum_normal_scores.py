"""Separate normal-score sensitivity to Sim(3) precision from nearest-neighbor backend effects."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
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


def check(sequence,slow=False,neighborhoods=False):
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
    geometry={}
    for mode,aligned in [('canonical_alignment',canonical),('float64_alignment',double)]:
        o3d.utility.random.seed(42)
        predicted=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))
        reference=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
        registered=o3d.pipelines.registration.registration_icp(predicted,reference,.1,np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        predicted.transform(registered.transformation)
        predicted.estimate_normals(fast_normal_computation=not slow)
        reference.estimate_normals(fast_normal_computation=not slow)
        xyz=np.asarray(predicted.points)
        pn,tn=np.asarray(predicted.normals),np.asarray(reference.normals)
        scores={backend:metrics(xyz,target,pn,tn,backend) for backend in ('scipy','open3d_core')}
        result[mode]=scores
        if neighborhoods:
            geometry[mode]=(xyz.copy(),pn.copy())
        print(sequence,mode,{b:round(v['normal_consistency_mean'],7) for b,v in scores.items()},flush=True)
    if neighborhoods:
        modes=('canonical_alignment','float64_alignment')
        ids=np.linspace(0,len(target)-1,min(4096,len(target)),dtype=int)
        neighbors=[]
        boundary_ties=[]
        for mode in modes:
            xyz,normals=geometry[mode]
            cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            tree=o3d.geometry.KDTreeFlann(cloud)
            selected=np.empty((len(ids),30),dtype=np.int64)
            ties=[]
            for j,index in enumerate(ids):
                count,indices,distances=tree.search_knn_vector_3d(xyz[index],31)
                assert count==31
                selected[j]=np.asarray(indices)[:30]
                d=np.asarray(distances)
                ties.append(abs(d[30]-d[29])<=1e-12*max(d[30],1e-9))
            neighbors.append(selected)
            boundary_ties.append(float(np.mean(ties)))
        overlap=np.any(neighbors[0][:,:,None]==neighbors[1][:,None,:],axis=2).sum(axis=1)/30
        first,second=(geometry[m] for m in modes)
        displacement=np.linalg.norm(first[0][ids]-second[0][ids],axis=1)
        dot=np.abs(np.einsum('ij,ij->i',first[1][ids],second[1][ids]))
        result['normal_neighborhoods']=dict(sample_count=len(ids),mean_neighbor_overlap=float(overlap.mean()),
            changed_neighborhood_fraction=float(np.mean(overlap<1)),boundary_tie_fraction=dict(zip(modes,boundary_ties)),
            mean_point_displacement_m=float(displacement.mean()),max_point_displacement_m=float(displacement.max()),
            mean_absolute_normal_dot=float(dot.mean()),normal_dot_below_099_fraction=float(np.mean(dot<.99)))
    expected=read(PUBLIC/'observations.json')[sequence][h]['metrics']
    result['original']={m:expected[m] for m in ('accuracy_mean_m','completeness_mean_m','normal_consistency_mean')}
    return sequence,dict(prediction=h,measurements=result)


def run(slow=False,neighborhoods=False):
    with ProcessPoolExecutor(3) as pool:
        records=dict(pool.map(partial(check,slow=slow,neighborhoods=neighborhoods),SEQUENCES))
    name='normal_neighborhoods.json' if neighborhoods else ('normal_solver_factorial.json' if slow else 'normal_factorial.json')
    dump(PUBLIC/name,dict(normal_solver='iterative_eigen' if slow else 'default_fast',prediction_sha256=sha(PUBLIC/'predictions.json'),
        original_observations_sha256=sha(PUBLIC/'observations.json'),independent_sha256=sha(PUBLIC/'independent_surface.json'),
        cases=records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slow-normals',action='store_true')
    parser.add_argument('--neighborhoods',action='store_true')
    args=parser.parse_args()
    run(args.slow_normals,args.neighborhoods)
