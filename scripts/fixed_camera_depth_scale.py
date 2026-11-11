"""Fit depth multipliers from RGB correspondences with frozen camera poses."""
import numpy as np
from scipy.optimize import lsq_linear


DELTA = 0.05
ANCHOR = 0.01
BOUND = 1.15


def displacements(x, y, depth, k, rotation):
    rays = np.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones_like(x)), axis=-1)
    return (rays @ rotation) * depth[..., None]


def sample(xy, depth, k, rotation):
    height, width = depth.shape
    x, y = xy.T
    inside = np.isfinite(xy).all(axis=1) & (x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1)
    x = np.where(inside, x, 0).astype(float)
    y = np.where(inside, y, 0).astype(float)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    neighbors = np.stack((depth[y0,x0], depth[y0,x1], depth[y1,x0], depth[y1,x1]), axis=1)
    dx, dy = x-x0, y-y0
    weights = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), axis=1)
    valid = inside & np.isfinite(neighbors).all(axis=1) & (neighbors > 0).all(axis=1)
    return displacements(x, y, np.sum(np.where(valid[:,None], neighbors, 0.)*weights, axis=1), k, rotation), valid


def centers(poses):
    return -np.einsum('ni,nij->nj', poses[:,:3,3], poses[:,:3,:3])


def fold_system(pairs, depths, k, poses):
    """Each row of A and y is one vector match, with r = A s - y."""
    c = centers(poses)
    count = len(c)
    distances = np.linalg.norm(c[:,None]-c[None,:], axis=-1)
    nonzero = distances[np.triu_indices(count, 1)]
    positive = nonzero[nonzero > 0]
    if not positive.size or not np.isfinite(nonzero).all():
        return None
    baseline = np.median(positive)
    edges = [set() for _ in c]
    blocks, targets, weights, support = [], [], [], []
    for i,j,xy_i,xy_j in pairs:
        a, ok_a = sample(xy_i, depths[i], k[i], poses[i,:3,:3])
        b, ok_b = sample(xy_j, depths[j], k[j], poses[j,:3,:3])
        mask = ok_a & ok_b
        n = int(mask.sum())
        if n < 6:
            continue
        edges[i].add(j)
        edges[j].add(i)
        block = np.zeros((n, 3, count))
        block[:,:,i], block[:,:,j] = a[mask]/baseline, -b[mask]/baseline
        blocks.append(block)
        targets.append(np.tile((c[j]-c[i])/baseline, (n,1)))
        weights.append(np.full(n, 1/n))
        support.append([int(i),int(j),n])
    reached, frontier = set(), {0}
    while frontier:
        reached.update(frontier)
        frontier = set.union(*(edges[i] for i in frontier)) - reached
    if len(reached) != count:
        return None
    return np.concatenate(blocks), np.concatenate(targets), np.concatenate(weights)/len(blocks), support


def robust_loss(system, scales):
    a, y, weight, _ = system
    norm = np.linalg.norm(a @ scales - y, axis=1)
    return float(weight @ np.where(norm <= DELTA, 0.5*norm**2, DELTA*(norm-0.5*DELTA)))


def fit(system, count, shared=False):
    if system is None:
        return np.ones(count), 'insufficient_support'
    a, y, weight, _ = system
    if shared:
        a = a.sum(axis=2, keepdims=True)
    m = a.shape[2]
    singular = np.linalg.svd((np.sqrt(weight)[:,None,None]*a).reshape(-1,m), compute_uv=False)
    if singular[0] == 0 or singular[-1] <= singular[0]*1e-6:
        return np.ones(count), 'rank_deficient'
    s = np.ones(m)
    anchor = np.sqrt(ANCHOR/m)
    for _ in range(8):
        norm = np.linalg.norm(a @ s - y, axis=1)
        w = np.sqrt(weight * DELTA / np.maximum(norm, DELTA))
        design = np.vstack(((w[:,None,None]*a).reshape(-1,m), anchor*np.eye(m)))
        target = np.r_[(w[:,None]*y).ravel(), np.full(m, anchor)]
        solved = lsq_linear(design, target, bounds=(1/BOUND, BOUND), method='bvls')
        if not solved.success or not np.isfinite(solved.x).all():
            raise RuntimeError('Bounded scale fit did not converge')
        change = max(np.abs(solved.x-s))
        s = solved.x
        if change < 1e-7:
            break
    return np.broadcast_to(s, (count,)).copy(), 'fitted'


def guarded_fit(fit_fold, check_fold, count):
    identity = np.ones(count)
    five, five_state = fit(fit_fold, count)
    before = after = None
    state = five_state
    if five_state == 'fitted':
        if check_fold is None:
            state = 'missing_validation'
        else:
            before, after = (robust_loss(check_fold, s) for s in (identity, five))
            state = 'accepted' if before > 1e-12 and after <= .99*before else 'validation_veto'
    return five, five if state=='accepted' else identity, dict(
        per_view_fitting=five_state, decision=state, before=before, after=after,
        support=[[] if system is None else system[3] for system in (fit_fold,check_fold)])


def correct(folds, depths, k, poses):
    systems = [fold_system(p, depths, k, poses) for p in folds]
    five, guarded, decision = guarded_fit(*systems, len(k))
    shared, decision['fitting'] = fit(systems[0], len(k), shared=True)
    return dict(direct=np.ones(len(k)), shared=shared, five=five, guarded=guarded), decision


def point_map(depths, k, poses, scales):
    y,x = np.indices(depths.shape[1:], dtype=float)
    c = centers(poses)
    return np.stack([(c[i] + scales[i]*displacements(x,y,depth,k[i],poses[i,:3,:3])).astype(np.float32)
                     for i,depth in enumerate(depths)])
