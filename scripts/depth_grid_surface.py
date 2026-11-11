"""Score reference-assisted depth maps on the same surface metric as the deployed proposals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing
from statistics import mean
import zipfile

import numpy as np

from collect_matched_focal_heads import tensor_digest
from dense_rgb_depth import FACTORS
from dense_rgb_pixel_check import similarity
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import REF,ROOT,write_arrays
from run_dense_rgb_depth import PARENT,PLAN,inputs
from run_visibility_model import PUBLIC,verify as verify_model
from score_dense_focal_pilot import REFS,METRICS
from score_depth_scale_study import evaluate_points
from score_saved_geometry import read


ARMS = ('all_pixels','proposal_sites','accept_or_keep')
DESIGN = ROOT/'docs/DEPTH_GRID_SURFACE_DIAGNOSTIC.md'
RAW = ROOT/'reports/depth_grid_surface'
SEAL = PUBLIC/'grid_surface_seal.json'
ZIP = PUBLIC/'grid_surface_choices.zip'


def generate():
    if SEAL.exists():
        raise FileExistsError(SEAL)
    model = verify_model()['batches']
    rows = read(PARENT/'predictions.json')['rows']
    plan = read(PLAN)
    destinations = RAW/'points'
    destinations.mkdir(parents=True,exist_ok=True)
    observations = {}
    part = ZIP.with_suffix('.partial')
    with zipfile.ZipFile(PUBLIC/'choices.zip') as z, zipfile.ZipFile(part,'w') as out:
        for key,b in model.items():
            _,depth,k,e,_ = inputs(key,rows,plan)
            spec = plan['references'][key]
            if sha(REF/spec['file'])!=spec['sha256']:
                raise ValueError('Reference changed')
            with np.load(REF/spec['file'],allow_pickle=False) as ref:
                valid = ref['valid_mask']
                target = ref['registered_rgb'][valid]
            with np.load(io.BytesIO(z.read(key+'.npz')),allow_pickle=False) as opts:
                single = opts['single_depth']
            maps = np.stack([point_map(depth*f,k,e,np.ones(5)) for f in FACTORS])
            if tensor_digest(maps[0])!=b['direct_sha256']:
                raise ValueError('Depth grid changed')
            s,r,t = similarity(maps[0][valid],target)
            error = np.stack([np.sum((s*p[valid].astype(float)@r+t-target)**2,axis=-1) for p in maps])
            best = error.argmin(axis=0).astype(np.uint8)
            proposed = single[valid]
            proposal_error = np.take_along_axis(error,proposed[None],axis=0)[0]
            choices = dict(all_pixels=best,proposal_sites=np.where(proposed!=0,best,0),
                           accept_or_keep=np.where(proposal_error<error[0],proposed,0))
            packed,records = {},{}
            for arm,indices in choices.items():
                full = np.zeros(valid.shape,dtype=np.uint8)
                full[valid] = indices
                points = np.take_along_axis(maps,full[None,...,None],axis=0)[0]
                h = tensor_digest(points)
                write_arrays(destinations/(h+'.npz'),dict(points=points))
                packed[arm] = full
                records[arm] = dict(tensor_sha256=h,choices_sha256=tensor_digest(full),
                    changed=int(np.count_nonzero(indices)),pixel_sse=float(np.take_along_axis(error,indices[None],axis=0).sum()))
            buf = io.BytesIO()
            np.savez_compressed(buf,**packed)
            out.writestr(key+'.npz',buf.getvalue())
            observations[key] = dict(baseline_sse=float(error[0].sum()),single_sha256=b['choice_hashes']['single_depth'],
                                     reference_sha256=spec['sha256'],arms=records)
            print(key,'sealed three reference-assisted point maps',flush=True)
    part.replace(ZIP)
    dump(SEAL,dict(design_sha256=sha(DESIGN),model_sha256=sha(PUBLIC/'predictions.json'),
                   mask_archive_sha256=sha(ZIP),batches=observations))


def verify():
    saved = read(SEAL)
    model = verify_model()['batches']
    pixels = read(PUBLIC/'pixel_check.json')['batches']
    if (saved['design_sha256']!=sha(DESIGN) or saved['model_sha256']!=sha(PUBLIC/'predictions.json')
            or saved['mask_archive_sha256']!=sha(ZIP) or set(saved['batches'])!=set(model)):
        raise ValueError('Diagnostic provenance differs')
    with zipfile.ZipFile(ZIP) as z, zipfile.ZipFile(PUBLIC/'choices.zip') as launched:
        for key,b in saved['batches'].items():
            with np.load(io.BytesIO(z.read(key+'.npz')),allow_pickle=False) as f:
                masks = dict(f)
            with np.load(io.BytesIO(launched.read(key+'.npz')),allow_pickle=False) as f:
                single = f['single_depth']
            if b['single_sha256']!=tensor_digest(single) or b['reference_sha256']!=pixels[key]['reference_sha256']:
                raise ValueError('Changed candidate sites or reference')
            for arm,m in masks.items():
                obs = b['arms'][arm]
                if m.dtype!=np.uint8 or np.max(m)>=5 or tensor_digest(m)!=obs['choices_sha256'] or np.count_nonzero(m)!=obs['changed']:
                    raise ValueError('Invalid diagnostic mask')
            if (np.any(masks['proposal_sites']!=np.where(single!=0,masks['all_pixels'],0)) or
                    np.any((masks['accept_or_keep']!=0)&(masks['accept_or_keep']!=single))):
                raise ValueError('Reference source selection escaped stated support')
            rs = [b['arms'][arm]['pixel_sse'] for arm in ARMS]+[b['baseline_sse']]
            if not np.isfinite(rs).all() or min(np.diff(rs)) < -1e-8:
                raise ValueError('Global and restricted pixel optimizations violate nesting')
            region = pixels[key]['results']['registered_rgb'].values()
            np.testing.assert_allclose(rs[0],sum(v['arms']['oracle']['sse'] for v in region),rtol=1e-10)
            np.testing.assert_allclose(rs[-1],sum(v['baseline_sse'] for v in region),rtol=1e-10)
    return saved['batches']


def batch(item):
    key,arms = item
    return evaluate_points(key,{o['tensor_sha256'] for o in arms.values()},RAW)


def aggregate(observed,rows):
    def group(ref,arm,subset):
        normalized = [mean(observed[r['input_key']][arm][ref][m]/r['metrics'][ref]['frozen'][m] for r in subset)
                      for m in METRICS[:2]]
        return dict(score=mean(normalized),accuracy_ratio=normalized[0],completeness_ratio=normalized[1],
                    raw={m:mean(observed[r['input_key']][arm][ref][m] for r in subset) for m in METRICS})
    return {ref:{arm:dict(overall=group(ref,arm,rows),scenes={s:group(ref,arm,[r for r in rows if r['scene']==s])
                     for s in sorted({r['scene'] for r in rows})}) for arm in ARMS} for ref in REFS}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('seal','evaluate','verify'))
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    args = p.parse_args()
    if args.action=='seal':
        generate()
    else:
        obs = verify()
        path = PUBLIC/'grid_surface.json'
        if args.action=='evaluate':
            if path.exists():
                raise FileExistsError(path)
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                measured = dict(pool.map(batch,[(k,b['arms']) for k,b in obs.items()]))
            scores = {k:{a:measured[k][v['tensor_sha256']] for a,v in b['arms'].items()} for k,b in obs.items()}
        else:
            saved = read(path)
            scores = saved['scores']
            backup = read(PUBLIC/'grid_surface_archive.json')
            files = backup['files']
            expected = {a['tensor_sha256']+'.npz' for b in obs.values() for a in b['arms'].values()}
            if (len(files)!=len(expected) or {f['file'] for f in files}!=expected or backup['verified_physical_drives']!=2
                    or sum(f['bytes'] for f in files)!=backup['total_bytes']):
                raise ValueError('Reference-assisted raw archive differs')
        for key in obs:
            for arm in ARMS:
                for ref in REFS:
                    m = scores[key][arm][ref]
                    if set(m)!=set(METRICS) or not np.isfinite(list(m.values())).all() or min(m[k] for k in METRICS[:2])<=0 or not 0<=m[METRICS[2]]<=1:
                        raise ValueError('Invalid reference-assisted measurement')
        result = aggregate(scores,read(PARENT/'scores.json')['rows'])
        if args.action=='evaluate':
            dump(path,dict(seal_sha256=sha(SEAL),scores=scores,results=result))
        elif saved['seal_sha256']!=sha(SEAL) or result!=saved['results']:
            raise ValueError('Surface diagnostic results changed')
        else:
            print('Verified the unrestricted, proposal-site and rejection-only reference-assisted surface comparisons.')
