"""Estimate robust bounded metric depth scale and shift from neighboring sparse measurements."""
import numpy as np
from scipy.optimize import lsq_linear

from neighbor_sparse_depth import corrections


def fit(predicted,measured):
    x=np.asarray(predicted,dtype=float)
    y=np.asarray(measured,dtype=float)
    if x.ndim!=1 or x.shape!=y.shape or len(x)<8 or not np.all(np.isfinite([x,y])) or min(x.min(),y.min())<=0:
        raise ValueError('Expected at least eight paired positive depths')
    unit=np.median(y)
    design=np.column_stack([x/unit,np.ones(len(x))])
    target=y/unit
    weights=np.ones(len(x))
    regularizer=np.eye(2)*np.sqrt(.01*len(x))
    prior=np.array([1.,0.])
    coefficients=prior.copy()
    for _ in range(5):
        a=np.vstack([design*np.sqrt(weights[:,None]),regularizer])
        b=np.r_[target*np.sqrt(weights),regularizer@prior]
        solution=lsq_linear(a,b,bounds=([.5,-.25],[2.,.25]),tol=1e-12)
        if not solution.success:
            raise ValueError('Constrained calibration failed')
        coefficients=solution.x
        residual=np.abs(design@coefficients-target)
        weights=np.minimum(1.,.02/np.maximum(residual,1e-12))
    return np.array([coefficients[0],coefficients[1]*unit])


def estimate(depth,samples):
    pairs=[]
    for predicted,values in zip(depth,samples):
        x,y=values[:,:2].T.astype(int)
        pairs.append((predicted[y,x],values[:,2]))
    gauge=np.exp(np.median(np.concatenate([np.log(y/x) for x,y in pairs])))
    metric=depth*gauge
    pooled=fit(np.concatenate([x*gauge for x,_ in pairs]),np.concatenate([y for _,y in pairs]))
    individual=np.stack([fit(x*gauge,y) for x,y in pairs])
    outputs={}
    diagnostics={}
    for arm,parameters in [('scene_affine',np.repeat(pooled[None],len(depth),axis=0)),('per_view_affine',individual)]:
        calibrated=parameters[:,0,None,None]*metric+parameters[:,1,None,None]
        floored=calibrated<.25*metric
        calibrated=np.maximum(calibrated,.25*metric)/gauge
        if arm=='per_view_affine':
            updated,final_gauge=corrections(calibrated,samples)
            outputs['guarded']=dict(depth=np.asarray(updated['guarded']),gauge=final_gauge)
        else:
            final_gauge=np.exp(np.median(np.concatenate([
                np.log(v[:,2]/calibrated[i,v[:,1].astype(int),v[:,0].astype(int)]) for i,v in enumerate(samples)])))
        outputs[arm]=dict(depth=calibrated,gauge=float(final_gauge))
        diagnostics[arm]=dict(coefficients=parameters.tolist(),floor_fraction=float(floored.mean()))
    return outputs,dict(initial_gauge=float(gauge),fits=diagnostics)
