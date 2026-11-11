"""Test signed target-depth faults across visibility, cue and source-count choices."""
import argparse
from itertools import product

import numpy as np

from dense_rgb_depth import LOG_STEPS
from evaluation_stability import dump,sha
from photometric_positive_control import STARTS,reference_camera,summarize as summarize_control
from prepare_dense_focal_pilot import INPUT,REF,ROOT
from run_dense_rgb_depth import PARENT,PLAN,PUBLIC as PREVIOUS
from score_saved_geometry import read
from visibility_depth_evidence import MASKS,CUES,RULES,estimate_views


PUBLIC = ROOT/'results/visibility_depth_evidence'
DESIGN = ROOT/'docs/VISIBILITY_DEPTH_EVIDENCE.md'
KEYS = list(product(MASKS,CUES,RULES))
LABELS = tuple('_'.join(k) for k in KEYS)


def measure(key,plan):
    rgb,ref = plan['inputs'][key],plan['references'][key]
    if sha(INPUT/rgb['file'])!=rgb['sha256'] or sha(REF/ref['file'])!=ref['sha256']:
        raise ValueError('Changed RGB/depth input')
    with np.load(INPUT/rgb['file'],allow_pickle=False) as f:
        images = f['images']
    with np.load(REF/ref['file'],allow_pickle=False) as f:
        valid = f['valid_mask']
        records = [reference_camera(p,m) for p,m in zip(f['registered_rgb'],valid)]
    k,e,depth = [np.stack(items) for items in zip(*(r[:3] for r in records))]
    views = [dict(view=i,camera_check=r[3],states={}) for i,r in enumerate(records)]
    for i,neighbors,state,outputs in estimate_views(images,depth,k,e,STARTS):
        views[i]['neighbors'] = neighbors
        counts = {}
        for keys,label in zip(KEYS,LABELS):
            d,ok = outputs[keys]
            eligible = ok&valid[i]
            counts[label] = dict(eligible=int(eligible.sum()),hist=np.bincount(d[eligible],minlength=len(LOG_STEPS)).tolist())
        views[i]['states'][state] = counts
    print(key,'measured twelve source-view controls at three starting depths',flush=True)
    return dict(rgb_sha256=rgb['sha256'],ref_sha256=ref['sha256'],views=views)


def verify(batches,old):
    if set(batches)!=set(old):
        raise ValueError('Changed control batches')
    for k,b in batches.items():
        other = old[k]
        if any(b[x]!=other[x] for x in ('ref_sha256','rgb_sha256')) or len(b['views'])!=5:
            raise ValueError('Control input identity differs')
        for v,prev in zip(b['views'],other['views']):
            if any(v[x]!=prev[x] for x in ('view','neighbors')) or v['camera_check']['valid_count']!=prev['camera_check']['valid_count']:
                raise ValueError('Camera identity or valid count differs')
            if v['camera_check']['max_pixel_error']>.02:
                raise ValueError('Reference reprojection check failed')
            for state in STARTS:
                cells = v['states'][state]
                if cells['all_candidates_rgb_four_view']!=prev['states'][state]['guarded']:
                    raise ValueError('Original four-view RGB rule failed exact replay')
                for mode,cue,rule in KEYS:
                    cell = cells['_'.join((mode,cue,rule))]
                    n = cell['eligible']
                    if (len(cell['hist'])!=5 or min(cell['hist'])<0 or sum(cell['hist'])!=n
                            or n>v['camera_check']['valid_count'] or n>cells['_'.join((mode,cue,'single'))]['eligible']
                            or n>cells['_'.join(('starting_depth',cue,rule))]['eligible']
                            or n!=cells['_'.join((mode,'rgb',rule))]['eligible']):
                        raise ValueError('Invalid factorial support or change counts')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','verify'))
    action = p.parse_args().action
    plan = read(PLAN)
    rows = read(PARENT/'predictions.json')['rows']
    old = PREVIOUS/'positive_control.json'
    path = PUBLIC/'positive_control.json'
    if action=='evaluate':
        if path.exists():
            raise FileExistsError(path)
        batches = {k:measure(k,plan) for k in sorted({r['input_key'] for r in rows})}
    else:
        saved = read(path)
        batches = saved['batches']
    verify(batches,read(old)['batches'])
    summary = summarize_control(batches,rows,LABELS)
    if action=='evaluate':
        dump(path,dict(design_sha256=sha(DESIGN),original_control_sha256=sha(old),batches=batches,results=summary))
    elif saved['design_sha256']!=sha(DESIGN) or saved['original_control_sha256']!=sha(old) or summary!=saved['results']:
        raise ValueError('Factorial evidence or summary changed')
    else:
        print('Replayed twelve factorial rules, all three starting depths and the original RGB control counts.')
