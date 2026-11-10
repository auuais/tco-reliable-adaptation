"""Image-only fixed correspondences and gauge-invariant pointmap diagnostics.

This is a conventional consistency baseline, not a novel method or an online
accuracy oracle. No reference geometry or camera prior enters these functions.
"""
import hashlib
import itertools
import numpy as np


def image_digest(images):
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()


def build_support(images):
    import cv2
    assert images.ndim==4 and images.shape[1]==3 and np.isfinite(images).all()
    assert 0<=images.min() and images.max()<=1
    cv2.setNumThreads(1);cv2.setRNGSeed(20261111)
    detector=cv2.SIFT_create(nfeatures=2000)
    features=[]
    for image in images:
        rgb=np.rint(image.transpose(1,2,0)*255).astype(np.uint8)
        features.append(detector.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None))
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    pairs=[];parents={}
    def root(node):
        parents.setdefault(node,node)
        while parents[node]!=node:
            parents[node]=parents[parents[node]];node=parents[node]
        return node
    def unite(a,b):
        a,b=root(a),root(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    def node(view,xy):return (view,int(round(xy[0]*2)),int(round(xy[1]*2)))
    for i,j in itertools.combinations(range(len(images)),2):
        ki,di=features[i];kj,dj=features[j]
        if di is None or dj is None or min(len(di),len(dj))<2:continue
        def ratio(a,b):
            return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
                    if len(pair)==2 for m,n in [pair] if m.distance<.8*n.distance}
        forward,backward=ratio(di,dj),ratio(dj,di)
        matches=sorted((a,b) for a,b in forward.items() if backward.get(b)==a)
        if len(matches)<8:continue
        a=np.asarray([ki[x].pt for x,y in matches],dtype=np.float32)
        b=np.asarray([kj[y].pt for x,y in matches],dtype=np.float32)
        cv2.setRNGSeed(20261111+i*100+j)
        _,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,.999,5000)
        if mask is None:continue
        a,b=a[mask.ravel().astype(bool)],b[mask.ravel().astype(bool)]
        if len(a)<20:continue
        nodes=[(node(i,x),node(j,y)) for x,y in zip(a,b)]
        for x,y in nodes:unite(x,y)
        pairs.append(dict(i=i,j=j,xy_i=a,xy_j=b,nodes=nodes))
    components={}
    for node_key in parents:components.setdefault(root(node_key),[]).append(node_key)
    folds={key:int(hashlib.sha256(repr(sorted(value)).encode()).hexdigest(),16)%2 for key,value in components.items()}
    for pair in pairs:
        pair["fold"]=np.asarray([folds[root(a)] for a,b in pair.pop("nodes")],dtype=np.int8)
    return pairs,dict(image_sha256=image_digest(images),opencv_version=cv2.__version__,
        feature_counts=[len(k) for k,d in features],pair_count=len(pairs),
        track_components=len(components),matching="SIFT 2000; mutual ratio .8; F RANSAC 1.5px/.999/5000; >=20 matches/pair",
        split="Connected match components, half-pixel location identity, deterministic SHA256 parity; no shared component between folds",
        reference_inputs_used=False)


def sample_points(pointmap,xy):
    xy=np.asarray(xy,dtype=np.float64)
    height,width=pointmap.shape[:2]
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError("Invalid fixed support")
    if np.any(xy<0) or np.any(xy[:,0]>width-1) or np.any(xy[:,1]>height-1):raise ValueError("Support outside image")
    x,y=xy.T;x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,width-1);y1=np.minimum(y0+1,height-1)
    u=(x-x0)[:,None];v=(y-y0)[:,None]
    return (1-u)*(1-v)*pointmap[y0,x0]+u*(1-v)*pointmap[y0,x1]+(1-u)*v*pointmap[y1,x0]+u*v*pointmap[y1,x1]


def score_pointmaps(points,pairs,min_matches_per_pair=10,min_total=100):
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=4 or points.shape[-1]!=3:raise ValueError("Expected view/height/width/xyz")
    if not np.isfinite(points).all():return dict(eligible=False,reason="nonfinite_prediction")
    views,height,width,_=points.shape
    ys=np.linspace(0,height-1,8,dtype=int);xs=np.linspace(0,width-1,8,dtype=int)
    anchors=points[:,ys[:,None],xs[None,:]].reshape(-1,3)
    distances=np.linalg.norm(anchors[:,None]-anchors[None,:],axis=-1)
    scale=float(np.median(distances[np.triu_indices(len(anchors),1)]))
    if not np.isfinite(scale) or scale<=1e-12:return dict(eligible=False,reason="collapsed_or_invalid_scale")
    folds=[]
    for fold in [0,1]:
        edges=[];graph={i:set() for i in range(views)};total=0
        for pair in pairs:
            chosen=np.asarray(pair["fold"])==fold
            if int(chosen.sum())<min_matches_per_pair:continue
            i,j=pair["i"],pair["j"]
            a=sample_points(points[i],np.asarray(pair["xy_i"])[chosen])
            b=sample_points(points[j],np.asarray(pair["xy_j"])[chosen])
            residual=np.linalg.norm(a-b,axis=-1)/scale
            edges.append(dict(i=i,j=j,count=len(residual),median=float(np.median(residual)),p90=float(np.quantile(residual,.9))))
            graph[i].add(j);graph[j].add(i);total+=len(residual)
        connected={0};pending=[0]
        while pending:
            i=pending.pop()
            for j in graph[i]-connected:connected.add(j);pending.append(j)
        eligible=len(connected)==views and total>=min_total
        folds.append(dict(fold=fold,eligible=eligible,matches=total,connected_views=len(connected),edges=edges,
                          score=float(np.mean([e["median"] for e in edges])) if eligible else None))
    return dict(eligible=all(f["eligible"] for f in folds),scale=scale,folds=folds,
                score=float(np.mean([f["score"] for f in folds])) if all(f["eligible"] for f in folds) else None)
