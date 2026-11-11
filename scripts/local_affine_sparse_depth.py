"""Fit regularized local affine log-depth residuals using sparse source measurements."""
import numpy as np
from scipy.spatial import cKDTree


def intercept(design,weights,residual,center):
    lhs=np.einsum('nki,nk,nkj->nij',design,weights,design)+np.diag([1.,4.,4.])
    rhs=np.einsum('nki,nk,nk->ni',design,weights,residual)
    rhs[:,0]+=center
    return np.linalg.solve(lhs,rhs[...,None])[:,0,0]


def estimate(depth,samples):
    if depth.ndim!=3 or len(samples)!=len(depth) or not np.all(np.isfinite(depth)&(depth>0)):
        raise ValueError('Expected a positive finite depth map for each view')
    ratios=[]
    for predicted,values in zip(depth,samples):
        if len(values)<8 or not np.isfinite(values).all() or np.any(values[:,2]<=0):
            raise ValueError('Insufficient or invalid sparse depth')
        x,y=values[:,:2].T.astype(int)
        ratios.append(np.log(values[:,2]/predicted[y,x]))
    log_gauge=np.median(np.concatenate(ratios))
    height,width=depth.shape[1:]
    y,x=np.indices((height,width))
    query=np.column_stack([x.ravel(),y.ravel()])
    outputs={'affine_spatial':[],'guarded':[]}
    for predicted,values,logs in zip(depth,samples,ratios):
        residual=np.clip(logs-log_gauge,-.15,.15)
        center=np.median(residual)
        xy=values[:,:2]
        sx,sy=xy.T.astype(int)
        fields={a:np.empty(len(query)) for a in outputs}
        tree=cKDTree(xy)
        for start in range(0,len(query),16384):
            stop=min(start+16384,len(query))
            q=query[start:stop]
            distance,nearest=tree.query(q,k=8)
            offsets=(xy[nearest]-q[:,None])/48
            design=np.concatenate([np.ones((*nearest.shape,1)),offsets],axis=2)
            spatial=np.exp(-.5*(distance/48)**2)
            delta=np.log(predicted.ravel()[start:stop,None]/predicted[sy,sx][nearest])
            weights={'affine_spatial':spatial,'guarded':spatial*np.exp(-.5*(delta/.08)**2)}
            for arm,w in weights.items():
                fields[arm][start:stop]=np.clip(intercept(design,w,residual[nearest],center),-.15,.15)
        for arm,field in fields.items():
            outputs[arm].append(predicted*np.exp(field.reshape(height,width)))
    return {a:np.stack(v) for a,v in outputs.items()},float(np.exp(log_gauge))
