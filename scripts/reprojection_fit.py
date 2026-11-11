"""Bounded joint RGB reprojection solver. Inputs are documented in the replay bundle."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

UNIT=np.array([np.deg2rad(1.)]*3+[.01]*3+[.02]*2)


def robust(r):
    # Signed square root of twice the soft-L1 cost, with scale measured in pixels.
    return r*np.sqrt(2/(np.sqrt(1+(r/3)**2)+1))


def project(params,fold,rot,centers,k):
    a,b,rays,observed=fold
    basis=np.clip(1/rays[:,2]-1,-1,1)
    corrected=rays*np.exp(params[a,6]+params[a,7]*basis)[:,None]
    r=rot@Rotation.from_rotvec(params[:,:3]).as_matrix().transpose(0,2,1)
    c=centers+params[:,3:6]
    world=np.einsum("ni,nij->nj",corrected,r[a])+c[a]
    cam=np.einsum("nij,nj->ni",r[b],world-c[b])
    xy=cam[:,:2]/np.maximum(cam[:,2,None],1e-4)
    pixels=xy*np.stack((k[b,0,0],k[b,1,1]),axis=1)+k[b,:2,2]
    return pixels-observed


def fit(arm,folds,rot,c,k,prior_params):
    mask=np.ones((len(c),8),bool)
    mask[0,:7]=False
    if arm=="depth": mask[:,:6]=False
    if arm=="pose": mask[:,6:]=False
    zero=np.zeros_like(mask,dtype=float)
    train,test=folds
    if train is None or test is None:
        return zero,dict(accepted=False,reason="missing_fold")
    a,b,*_=train
    visited={0}
    for _ in range(len(c)-1):
        visited.update(b[np.isin(a,list(visited))].tolist())
    if len(visited)!=len(c): return zero,dict(accepted=False,reason="disconnected")
    def unpack(v):
        out=zero.copy()
        out[mask]=v
        return out*UNIT
    def cost(p,fold):
        return float(np.mean(robust(project(p,fold,rot,c,k))**2))
    def residual(v):
        p=unpack(v)
        image=robust(project(p,train,rot,c,k)).ravel()
        parts=[image/np.sqrt(image.size),v/np.sqrt(v.size)]
        if arm=="joint_prior":
            err=((p[1:,:6]-prior_params[1:,:6])/UNIT[:6]).ravel()
            parts.append(err/np.sqrt(err.size))
        return np.concatenate(parts)
    solution=least_squares(residual,np.zeros(mask.sum()),bounds=(-2.,2.),max_nfev=150)
    if not solution.success: raise RuntimeError(solution.message)
    proposed=unpack(solution.x)
    before,after=cost(zero,test),cost(proposed,test)
    accepted=after<=.99*before and before>0
    return proposed if accepted else zero,dict(accepted=accepted,train_before=cost(zero,train),train_after=cost(proposed,train),
            validation_before=before,validation_after=after,proposed=proposed.tolist(),evals=solution.nfev)
