"""Use RGB correspondences to repair one supplied camera while holding its neighbors fixed."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def diameter(e):
    centers = np.linalg.inv(e)[:,:3,3]
    return float(np.max(np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)))


def perturb_camera(e, k, rotation_vector, center_delta, log_focal):
    out, intr = e.copy(), k.copy()
    center = np.linalg.inv(e)[:3,3] + center_delta
    out[:3,:3] = Rotation.from_rotvec(rotation_vector).as_matrix() @ e[:3,:3]
    out[:3,3] = -out[:3,:3] @ center
    intr[[0,1],[0,1]] *= np.exp(log_focal)
    return out, intr


def prepare_edges(e, k, pairs, view):
    edges = []
    for i,j,a,b in pairs:
        if view not in (i,j) or len(a) < 6:
            continue
        neighbor, xy, target = (j,a,b) if view==i else (i,b,a)
        take = np.linspace(0,len(xy)-1,num=min(64,len(xy)),dtype=int)
        p = np.column_stack((xy[take],np.ones(len(take))))
        q = np.column_stack((target[take],np.ones(len(take))))
        edges.append((e[neighbor],np.linalg.inv(k[neighbor]).T,p,q))
    return edges


def residuals(e, k, edges):
    ki, camera = np.linalg.inv(k), np.linalg.inv(e)
    values = []
    for neighbor,kj,p,q in edges:
        motion = neighbor @ camera
        x,y,z = motion[:3,3]
        cross = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        f = kj @ cross @ motion[:3,:3] @ ki
        a,b = p @ f.T,q @ f
        den = np.sqrt(np.sum(a[:,:2]**2+b[:,:2]**2,axis=1))
        if np.any(den<1e-14):
            raise ValueError("Degenerate camera baseline")
        values.append(np.sum(q*a,axis=1)/den)
    return values


def fit_camera(e, k, pairs, view):
    # Each image pair has one vote, independently of the number of matches.
    edges = prepare_edges(e,k,pairs,view)
    scale = diameter(e)
    if len(edges)<2 or scale<=1e-8:
        return None

    def decode(x):
        return perturb_camera(e[view], k[view], .1*x[:3], .1*scale*x[3:6], .1*x[6])

    def objective(x):
        raw = residuals(*decode(x), edges)
        # Soft L1 on pixel distances with a two-pixel transition, then equal pair weights.
        transformed = [r*np.sqrt(2/(1+np.hypot(1.,r/2)))/np.sqrt(len(r)) for r in raw]
        return np.concatenate([*transformed, x])

    solution = least_squares(objective,np.zeros(7),bounds=(-2.,2.),max_nfev=40)
    if not solution.success or not np.isfinite(solution.x).all():
        return None
    return decode(solution.x)


def validation_score(candidate, edges):
    if len(edges)<2:
        return None
    return float(np.median([np.median(np.abs(r)) for r in residuals(*candidate,edges)]))


def virtual_error(clean_e, clean_k, proposed_e, proposed_k):
    # Offline only. Use the clean camera to create the same rays and depths for every method.
    y,x=np.meshgrid(np.linspace(.1*391,.9*391,5),np.linspace(.1*517,.9*517,7),indexing="ij")
    image=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)))
    rays=image @ np.linalg.inv(clean_k).T
    xyz=(np.array([1.,2.,4.])[:,None,None]*rays).reshape(-1,3)
    points=np.column_stack((xyz,np.ones(len(xyz))))
    transformed=points @ (proposed_e @ np.linalg.inv(clean_e)).T
    projected=transformed[:,:3] @ proposed_k.T
    if np.any(projected[:,2]<=0):
        raise ValueError("Virtual test point behind camera")
    residual=projected[:,:2]/projected[:,2:] - np.tile(image[:,:2],(3,1))
    return float(np.mean(np.linalg.norm(residual,axis=1)))
