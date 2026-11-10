"""Frozen image-only DISK/LightGlue support; no adaptation or reference geometry."""
import hashlib
import itertools
import numpy as np


def split_components(pairs):
    """Use the same half-pixel identity and component-hash split as v6."""
    parents={}
    def root(node):
        parents.setdefault(node,node)
        while parents[node]!=node:
            parents[node]=parents[parents[node]];node=parents[node]
        return node
    def node(view,xy):return (view,int(round(xy[0]*2)),int(round(xy[1]*2)))
    for pair in pairs:
        pair["nodes"]=[(node(pair["i"],x),node(pair["j"],y)) for x,y in zip(pair["xy_i"],pair["xy_j"])]
        for a,b in pair["nodes"]:
            a,b=root(a),root(b)
            if a!=b:parents[max(a,b)]=min(a,b)
    components={}
    for key in parents:components.setdefault(root(key),[]).append(key)
    folds={key:int(hashlib.sha256(repr(sorted(value)).encode()).hexdigest(),16)%2 for key,value in components.items()}
    for pair in pairs:pair["fold"]=np.asarray([folds[root(a)] for a,b in pair.pop("nodes")],dtype=np.int8)
    return len(components)


def coverage(pairs,views=5):
    """Coverage depends only on fixed image support, not any predicted geometry."""
    result=[]
    for fold in [0,1]:
        graph={i:set() for i in range(views)};total=0;edges=[]
        for pair in pairs:
            count=int(np.sum(pair["fold"]==fold))
            if count<10:continue
            i,j=pair["i"],pair["j"];graph[i].add(j);graph[j].add(i);total+=count
            edges.append(dict(i=i,j=j,count=count))
        reached={0};pending=[0]
        while pending:
            for j in graph[pending.pop()]-reached:reached.add(j);pending.append(j)
        result.append(dict(fold=fold,connected_views=len(reached),matches=total,eligible=len(reached)==views and total>=100,edges=edges))
    return result


def build(images,extractor,matcher):
    import cv2
    import torch
    assert images.shape==(5,3,392,518) and np.isfinite(images).all() and 0<=images.min() and images.max()<=1
    cv2.setNumThreads(1);features=[];pairs=[];attempts=[]
    with torch.inference_mode():
        for image in images:features.append(extractor.extract(torch.from_numpy(image),resize=None))
        for i,j in itertools.combinations(range(len(images)),2):
            result=matcher(dict(image0=features[i],image1=features[j]));matches=result["matches"][0].cpu().numpy()
            record=dict(i=i,j=j,raw_matches=len(matches),in_bounds_matches=0,ransac_inliers=0,retained=False)
            attempts.append(record)
            if len(matches)<8:continue
            a=features[i]["keypoints"][0].cpu().numpy()[matches[:,0]]
            b=features[j]["keypoints"][0].cpu().numpy()[matches[:,1]]
            # Reject unsupported coordinates; never clip them into the image.
            valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
            for xy in [a,b]:valid &= (xy[:,0]>=0)&(xy[:,0]<=517)&(xy[:,1]>=0)&(xy[:,1]<=391)
            a,b=a[valid].astype(np.float32),b[valid].astype(np.float32);record["in_bounds_matches"]=len(a)
            if len(a)<8:continue
            cv2.setRNGSeed(20261111+i*100+j)
            _,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,.999,5000)
            if mask is None:continue
            a,b=a[mask.ravel().astype(bool)],b[mask.ravel().astype(bool)];record["ransac_inliers"]=len(a)
            if len(a)<20:continue
            record["retained"]=True;pairs.append(dict(i=i,j=j,xy_i=a,xy_j=b))
    components=split_components(pairs)
    return pairs,dict(feature_counts=[f["keypoints"].shape[1] for f in features],attempted_pairs=attempts,
                      track_components=components,folds=coverage(pairs),reference_inputs_used=False)
