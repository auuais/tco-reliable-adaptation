"""Separate camera visibility, selection by nonzero updates and depth-band propagation."""
import argparse
import hashlib
import io
from statistics import mean
import zipfile

import numpy as np

from capture_visibility_support import DESIGN,PUBLIC,ZIP as SUPPORT,unpack,replay as check_support
from dense_rgb_depth import FACTORS
from dense_rgb_pixel_check import similarity
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import centers,point_map
from prepare_dense_focal_pilot import REF
from run_dense_rgb_depth import PARENT,PLAN,inputs
from run_visibility_model import PUBLIC as MODEL
from score_dense_focal_pilot import REFS
from score_saved_geometry import read
from spatial_depth_transfer import matched_random,sums,ZIP as OLD_MOMENTS


TRAIN = ('ref_proposal','ref_identity','ref_eligible','random_within_eligible','random_all_eligible_count',
         'random_all_proposal_count','pseudo_proposal','pseudo_eligible')
ARMS = TRAIN+('pseudo_bands',)
REGIONS = ('combined','eligible_identity','unavailable')
PACK = PUBLIC/'tile_moments.zip'
TARGET = PUBLIC/'summary.json'


def seed(key,view,color,extra=''):
    return int.from_bytes(hashlib.sha256(f'{key}/{view}/4/{color}{extra}'.encode()).digest()[:8],'little')


def collect_batch(key,rows,plan,source,available):
    _,depth,k,e,direct = inputs(key,rows,plan)
    p = point_map(depth,k,e,np.ones(5)).astype(float)
    native = p-centers(e)[:,None,None,:]
    norm = np.sum(native**2,axis=-1)
    ref = plan['references'][key]
    if sha(REF/ref['file'])!=ref['sha256']:
        raise ValueError('Reference changed')
    with np.load(REF/ref['file'],allow_pickle=False) as f:
        targets = dict(f)
    with np.load(io.BytesIO(source.read(key+'.npz')),allow_pickle=False) as f:
        single = f['single_depth']
    eligible = unpack(available.read(key+'.npz'))
    valid = targets['valid_mask']
    proposed = single!=0
    metrics = {}
    for name in REFS:
        q = targets[name]
        scale,rot,shift = similarity(p[valid],q[valid])
        b = scale*native@rot
        residual = q-(scale*p@rot+shift)
        metrics[name] = np.sum(b*b,axis=-1),np.sum(b*residual,axis=-1),np.sum(residual**2,axis=-1)
    h,w = depth.shape[1:]
    y,x = np.indices((h,w))
    blocks = (y//16+x//16)%2
    tile = 4*(4*y//h)+4*x//w
    data = np.zeros((2,5,2,16,3,8+3*len(TRAIN)))
    for i in range(5):
        ids = 3*tile.copy()
        for t in range(16):
            here = tile==t
            thresholds = np.quantile(depth[i,here],[1/3,2/3])
            ids[here] += np.searchsorted(thresholds,depth[i,here],side='right')
        for color in (0,1):
            training = blocks==color
            all_ref = training&valid[i]
            v = all_ref&eligible[i]
            p_train = all_ref&proposed[i]
            masks = [p_train,v&~proposed[i],v,
                matched_random(tile,v,p_train,16,seed(key,i,color,'/eligible')),
                matched_random(tile,all_ref,v,16,seed(key,i,color,'/all-visible-count')),
                matched_random(tile,all_ref,p_train,16,seed(key,i,color))]
            label = norm[i]*(FACTORS[single[i]]-1)
            pseudo = [sums(ids,training&m,norm[i],label,48) for m in (proposed[i],eligible[i])]
            test = ~training&valid[i]&~proposed[i]
            for j,name in enumerate(REFS):
                a,b,err = (m[i] for m in metrics[name])
                cols = []
                for m in (test&eligible[i],test&~eligible[i]):
                    cols.extend([sums(ids,m,a,b,48),np.bincount(ids[m],weights=err[m],minlength=48)[:,None]])
                cols.extend(sums(ids,m,a,b,48) for m in masks)
                data[j,i,color] = np.column_stack((*cols,*pseudo)).reshape(16,3,-1)
    stream = io.BytesIO()
    np.savez_compressed(stream,moments=data)
    blob = stream.getvalue()
    return blob,dict(direct_sha256=direct,reference_sha256=ref['sha256'],moments_sha256=hashlib.sha256(blob).hexdigest(),
                     valid=int(valid.sum()),region_counts=[int(np.sum(valid&eligible&~proposed)),int(np.sum(valid&~eligible))])


def outcome(values,arm,region):
    if region=='combined':
        test = values[...,:4]+values[...,4:8]
    else:
        off = 0 if region=='eligible_identity' else 4
        test = values[...,off:off+4]
    name = 'pseudo_eligible' if arm=='pseudo_bands' else arm
    start = 8+3*TRAIN.index(name)
    train = values[...,start:start+3]
    if arm!='pseudo_bands':
        train = train.sum(axis=-2,keepdims=True)
    count,a,b = (train[...,i] for i in range(3))
    delta = np.clip(np.divide(b,a,out=np.zeros_like(b),where=(count>=16)&(a>0)),min(FACTORS)-1,max(FACTORS)-1)
    n,a,b,sse = (test[...,i] for i in range(4))
    corrected = sse-2*delta*b+delta**2*a
    if np.min(corrected)<-1e-8:
        raise ValueError('Invalid corrected SSE')
    return dict(n=int(n.sum()),changed=int(np.sum(n*(delta!=0))),baseline=float(sse.sum()),sse=float(np.maximum(corrected,0).sum()))


def replay(obs,rows):
    support = check_support()
    if set(obs)!=set(support['batches']):
        raise ValueError('Missing selection batch')
    batches = {}
    with zipfile.ZipFile(PACK) as z,zipfile.ZipFile(OLD_MOMENTS) as prior:
        for key,o in obs.items():
            blob = z.read(key+'.npz')
            if hashlib.sha256(blob).hexdigest()!=o['moments_sha256']:
                raise ValueError('Tile moments changed')
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:
                a = f['moments']
            with np.load(io.BytesIO(prior.read(key+'.npz')),allow_pickle=False) as f:
                older = f['4']
            if a.shape!=(2,5,2,16,3,32) or not np.isfinite(a).all():
                raise ValueError('Invalid tile moment grid')
            coarse = a.sum(axis=-2)
            np.testing.assert_allclose(coarse[...,:4]+coarse[...,4:8],older[...,:4],atol=1e-8,rtol=1e-9)
            for name,old_off in [('ref_proposal',7),('random_all_proposal_count',10),('pseudo_proposal',13)]:
                i = 8+3*TRAIN.index(name)
                np.testing.assert_allclose(coarse[...,i:i+3],older[...,old_off:old_off+3],atol=1e-8,rtol=1e-9)
            if (not np.array_equal(coarse[...,8],coarse[...,17]) or not np.array_equal(coarse[...,14],coarse[...,20])
                    or not np.array_equal(a[0,...,26:],a[1,...,26:])):
                raise ValueError('Count matching or unsupervised fit changed')
            batches[key] = {ref:{region:{arm:outcome(a[j],arm,region) for arm in ARMS} for region in REGIONS} for j,ref in enumerate(REFS)}
            if [batches[key][REFS[0]][r][ARMS[0]]['n'] for r in REGIONS[1:]]!=o['region_counts']:
                raise ValueError('Eligibility-stratified test population differs')
    def group(subset):
        result = {}
        for ref in REFS:
            result[ref] = {}
            for region in REGIONS:
                summary = {}
                for arm in ARMS:
                    cells = [batches[r['input_key']][ref][region][arm] for r in subset]
                    cells = [c for c in cells if c['n'] and c['baseline']>0]
                    summary[arm] = dict(conditions=len(cells),sse_reduction=mean(1-c['sse']/c['baseline'] for c in cells) if cells else None,
                        changed_fraction=mean(c['changed']/c['n'] for c in cells) if cells else None)
                result[ref][region] = summary
        return result
    return dict(overall=group(rows),by_scene={s:group([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('measure','verify'))
    action = parser.parse_args().action
    support = check_support()
    rows = read(PARENT/'predictions.json')['rows']
    if action=='measure':
        if TARGET.exists():
            raise FileExistsError(TARGET)
        part = PACK.with_suffix('.partial')
        obs = {}
        with zipfile.ZipFile(part,'w') as out,zipfile.ZipFile(MODEL/'choices.zip') as source,zipfile.ZipFile(SUPPORT) as available:
            plan = read(PLAN)
            for k in sorted(support['batches']):
                blob,obs[k] = collect_batch(k,rows,plan,source,available)
                out.writestr(k+'.npz',blob)
                print(k,'compared eligible, selected and random supervision',flush=True)
        part.replace(PACK)
    else:
        saved = read(TARGET)
        obs = saved['batches']
    result = replay(obs,rows)
    if action=='measure':
        dump(TARGET,dict(design_sha256=sha(DESIGN),support_sha256=sha(PUBLIC/'support.json'),moments_sha256=sha(PACK),
                         batches=obs,results=result))
    elif (saved['design_sha256']!=sha(DESIGN) or saved['support_sha256']!=sha(PUBLIC/'support.json')
            or saved['moments_sha256']!=sha(PACK) or result!=saved['results']):
        raise ValueError('Observation-selection evidence changed')
    else:
        print('Replayed selection and visibility contrasts and cross-checked shared tile fits against the prior experiment.')
