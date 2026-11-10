"""Final-depth trust bounds with fixed final cameras, no labels or view masks."""
import numpy as np
MODES=("tco","freeze_shape","half_shape","bounded_5pct")
def move_along_rays(points, centers, initial, final, mode):
    if mode not in MODES:raise ValueError(mode)
    X=np.asarray(points);C=np.asarray(centers,dtype=np.float64)
    z0=np.asarray(initial,dtype=np.float64);z1=np.asarray(final,dtype=np.float64)
    if X.shape!=(*z0.shape,3) or z0.shape!=z1.shape or C.shape!=(len(X),3):raise ValueError("Shapes do not match")
    if not all(np.isfinite(a).all() for a in [X,C,z0,z1]) or min(z0.min(),z1.min())<=0:raise ValueError("Invalid geometry")
    # One nuisance scale per scene; retain any genuine change in that scale.
    log_ratio=np.log(z1)-np.log(z0);center=float(np.median(log_ratio));r=log_ratio-center
    radius=np.log(1.05)
    if mode=="tco":return X.copy(),dict(global_log_scale=center,changed_fraction=0.)
    t={"freeze_shape":lambda x:np.zeros_like(x),"half_shape":lambda x:x*.5,"bounded_5pct":lambda x:np.clip(x,-radius,radius)}[mode](r)
    delta=t-r;C=C[:,None,None,:]
    # No-op pixels are copied exactly, and movement stays on the original ray.
    Y=X.copy();changed=delta!=0
    correction=(X-C)*np.expm1(delta)[...,None];Y[changed]=(X[changed]+correction[changed]).astype(X.dtype)
    if not np.isfinite(Y).all():raise ValueError("Invalid result")
    return Y,dict(global_log_scale=center,changed_fraction=float(changed.mean()))
