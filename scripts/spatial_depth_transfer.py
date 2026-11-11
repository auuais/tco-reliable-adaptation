"""Cross-fit bounded tile scales and score only withheld pixels without a source proposal."""
import argparse
import hashlib
import io
from statistics import mean
import zipfile

import numpy as np

from dense_rgb_depth import FACTORS
from dense_rgb_pixel_check import similarity
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import centers,point_map
from prepare_dense_focal_pilot import REF,ROOT
from run_dense_rgb_depth import PARENT,PLAN,inputs
from run_visibility_model import PUBLIC
from score_dense_focal_pilot import REFS
from score_saved_geometry import read


DESIGN = ROOT/'docs/SPATIAL_DEPTH_TRANSFER.md'
GRIDS = (1,2,4,8)
ARMS = ('all_reference','proposal_reference','random_reference','proposal_value')
ZIP = PUBLIC/'spatial_tile_moments.zip'
TARGET = PUBLIC/'spatial_transfer.json'


def sums(ids,mask,quadratic,linear,n_tiles):
    tile = ids[mask]
    return np.column_stack((np.bincount(tile,minlength=n_tiles),np.bincount(tile,weights=quadratic[mask],minlength=n_tiles),
                            np.bincount(tile,weights=linear[mask],minlength=n_tiles)))


def matched_random(ids,train,selected,n_tiles,seed):
    rng = np.random.default_rng(seed)
    chosen = np.zeros(train.shape,dtype=bool)
    for tile in range(n_tiles):
        avail = np.flatnonzero(train&(ids==tile))
        n = np.count_nonzero(selected&(ids==tile))
        if n:
            chosen.flat[rng.choice(avail,n,replace=False)] = True
    return chosen


def collect_batch(key,rows,plan,z):
    _,depth,k,e,digest = inputs(key,rows,plan)
    p = point_map(depth,k,e,np.ones(5)).astype(float)
    center = centers(e)
    native_ray = p-center[:,None,None,:]
    native_norm = np.sum(native_ray**2,axis=-1)
    spec = plan['references'][key]
    if sha(REF/spec['file'])!=spec['sha256']:
        raise ValueError('Reference changed')
    with np.load(REF/spec['file'],allow_pickle=False) as f:
        targets = dict(f)
    with np.load(io.BytesIO(z.read(key+'.npz')),allow_pickle=False) as f:
        single = f['single_depth']
    valid = targets['valid_mask']
    proposal = single!=0
    _,h,w = depth.shape
    y,x = np.indices((h,w))
    blocks = (x//16+y//16)%2
    metrics = {}
    for ref in REFS:
        q = targets[ref]
        scale,rotation,shift = similarity(p[valid],q[valid])
        residual = q-(scale*p@rotation+shift)
        ray = scale*native_ray@rotation
        metrics[ref] = (np.sum(ray**2,axis=-1),np.sum(ray*residual,axis=-1),np.sum(residual**2,axis=-1))
    packed = {}
    for grid in GRIDS:
        ids = (y*grid//h)*grid + x*grid//w
        tiles = grid**2
        values = np.zeros((len(REFS),5,2,tiles,4+3*len(ARMS)))
        for i in range(5):
            for color in (0,1):
                training = blocks==color
                valid_train = valid[i]&training
                proposed_train = valid_train&proposal[i]
                test = valid[i]&~training&~proposal[i]
                token = f'{key}/{i}/{grid}/{color}'.encode()
                seed = int.from_bytes(hashlib.sha256(token).digest()[:8],'little')
                random = matched_random(ids,valid_train,proposed_train,tiles,seed)
                pseudo = sums(ids,training&proposal[i],native_norm[i],native_norm[i]*(FACTORS[single[i]]-1),tiles)
                for r,ref in enumerate(REFS):
                    quadratic,linear,error = (v[i] for v in metrics[ref])
                    test_sums = sums(ids,test,quadratic,linear,tiles)
                    sse = np.bincount(ids[test],weights=error[test],minlength=tiles)[:,None]
                    fits = [sums(ids,mask,quadratic,linear,tiles) for mask in (valid_train,proposed_train,random)]
                    values[r,i,color] = np.column_stack((test_sums,sse,*fits,pseudo))
        packed[str(grid)] = values
    with io.BytesIO() as buf:
        np.savez_compressed(buf,**packed)
        blob = buf.getvalue()
    return blob,dict(direct_sha256=digest,reference_sha256=spec['sha256'],valid=int(valid.sum()),
                     test_pixels=int(np.count_nonzero(valid&~proposal)),moments_sha256=hashlib.sha256(blob).hexdigest())


def tile_outcome(values,arm):
    n,a,b,baseline = values[...,:4].transpose(3,0,1,2)
    off = 4+3*ARMS.index(arm)
    count,train_a,train_b = values[...,off:off+3].transpose(3,0,1,2)
    active = (count>=16)&(train_a>0)
    delta = np.clip(np.divide(train_b,train_a,out=np.zeros_like(train_a),where=active),min(FACTORS)-1,max(FACTORS)-1)
    score = baseline-2*delta*b+delta**2*a
    if score.min() < -1e-8:
        raise ValueError('Negative tile error beyond floating point tolerance')
    return dict(n=int(n.sum()),changed=int(n[delta!=0].sum()),base_sse=float(baseline.sum()),sse=float(np.maximum(score,0).sum()))


def replay(observations,rows):
    batches = {}
    model = read(PUBLIC/'predictions.json')['batches']
    refs = read(PLAN)['references']
    with zipfile.ZipFile(ZIP) as z:
        for k,o in observations.items():
            data = z.read(k+'.npz')
            if (hashlib.sha256(data).hexdigest()!=o['moments_sha256'] or model[k]['direct_sha256']!=o['direct_sha256']
                    or refs[k]['sha256']!=o['reference_sha256'] or refs[k]['valid_pixels']!=o['valid']):
                raise ValueError('Tile moments refer to different observations')
            result = {ref:{} for ref in REFS}
            with np.load(io.BytesIO(data),allow_pickle=False) as f:
                for grid in GRIDS:
                    a = f[str(grid)]
                    if (a.shape!=(2,5,2,grid**2,16) or not np.isfinite(a).all()
                            or not np.array_equal(a[...,7],a[...,10]) or np.any(a[...,7]>a[...,4]) or np.any(a[...,7]>a[...,13])
                            or not np.array_equal(a[0,...,13:],a[1,...,13:])):
                        raise ValueError('Invalid training balance or reference-independent pseudo-target')
                    for j,ref in enumerate(REFS):
                        methods = {arm:tile_outcome(a[j],arm) for arm in ARMS}
                        if methods['all_reference']['n']!=o['test_pixels']:
                            raise ValueError('Out-of-sample raster coverage changed')
                        result[ref][str(grid)] = methods
            batches[k] = result
    def group(items):
        data = {}
        for ref in REFS:
            data[ref] = {}
            for grid in GRIDS:
                data[ref][str(grid)] = {}
                for arm in ARMS:
                    vals = [batches[r['input_key']][ref][str(grid)][arm] for r in items]
                    data[ref][str(grid)][arm] = dict(sse_reduction=mean(1-v['sse']/v['base_sse'] for v in vals),
                        changed_fraction=mean(v['changed']/v['n'] for v in vals))
        return data
    return dict(primary_valid_fraction=mean(observations[r['input_key']]['test_pixels']/observations[r['input_key']]['valid'] for r in rows),
        overall=group(rows),by_scene={s:group([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})})


def synthetic():
    a = np.zeros((5,2,1,16))
    a[0,0,0,:4] = [40,400,20,2]
    a[0,0,0,4:7] = [16,100,5]
    m = tile_outcome(a,'all_reference')
    np.testing.assert_allclose(m['sse'],1.)
    if m['changed']!=40 or tile_outcome(a,'proposal_value')['changed']!=0:
        raise ValueError('Synthetic support or held-out accounting failed')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('test','measure','verify'))
    action = p.parse_args().action
    synthetic()
    if action!='test':
        rows = read(PARENT/'predictions.json')['rows']
        if action=='measure':
            if TARGET.exists():
                raise FileExistsError(TARGET)
            obs = {}
            part=ZIP.with_suffix('.partial')
            with zipfile.ZipFile(PUBLIC/'choices.zip') as source,zipfile.ZipFile(part,'w') as out:
                plan = read(PLAN)
                for key in sorted({r['input_key'] for r in rows}):
                    blob,obs[key] = collect_batch(key,rows,plan,source)
                    out.writestr(key+'.npz',blob)
                    print(key,'fit and evaluated disjoint spatial blocks',flush=True)
            part.replace(ZIP)
        else:
            saved = read(TARGET)
            obs = saved['batches']
        result = replay(obs,rows)
        if action=='measure':
            dump(TARGET,dict(design_sha256=sha(DESIGN),archive_sha256=sha(ZIP),batches=obs,results=result))
        elif saved['design_sha256']!=sha(DESIGN) or saved['archive_sha256']!=sha(ZIP) or result!=saved['results']:
            raise ValueError('Tile study results differ')
        else:
            print('Replayed all tile fits, both block colors, all four sources and both depth references.')
