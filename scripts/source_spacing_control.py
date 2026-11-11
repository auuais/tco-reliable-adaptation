"""Measure source coverage and depth recovery for four source cameras at three spacings."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing
from statistics import mean
import zipfile

import numpy as np

from capture_visibility_support import ZIP as OLD_SUPPORT,unpack
from collect_matched_focal_heads import RAW as HEADS
from collect_relative_focal_windows import FRAMES,SCENES
from dense_rgb_depth import LOG_STEPS
from depth_source_transplant import STATES,select,summarize_pixels
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import centers
from photometric_positive_control import STARTS
from prepare_source_spacing import DESIGN,INDEX,PUBLIC,RAW,REGIMES,sources,verify as verify_inputs
from run_visibility_model import PUBLIC as OLD
from score_saved_geometry import read


CHOICES = PUBLIC/'reference_choices.zip'
SUMMARY = PUBLIC/'reference_control.json'
REGIONS = ('all','previously_available','previously_unavailable')


def measure(scene):
    frames = verify_inputs()['frames'][scene]
    slug = scene.split('/')[0]
    key = slug+'_base'
    observations = {}
    for number,spec in frames.items():
        p = RAW/'reference'/slug/(f'{int(number):06d}.npz')
        if sha(p)!=spec['reference_sha256']:
            raise ValueError('Source geometry file changed')
        with np.load(p,allow_pickle=False) as z:
            observations[int(number)] = dict(z)
    prior = read(OLD/'source_transplant.json')['batches'][key]
    with np.load(HEADS/key/'heads_and_points.npz',allow_pickle=False) as f:
        natural = f['depth'][...,0].astype(float)*prior['scale_to_metres']
    with zipfile.ZipFile(OLD_SUPPORT) as z:
        old_support = unpack(z.read(key+'.npz'))
    records = []
    for i,frame in enumerate(FRAMES):
        target = observations[frame]
        valid,truth = target['valid'],target['depth']
        masks = (valid,valid&old_support[i],valid&~old_support[i])
        packed = dict(valid=np.packbits(valid),previously_available=np.packbits(masks[1]))
        methods = {}
        for regime in REGIMES:
            ids = list(FRAMES) if regime=='dispersed' else [frame,*sources(frame,regime)]
            j = ids.index(frame)
            k,e,depth = [np.stack([observations[x][n] for x in ids]) for n in ('k','e','depth')]
            poses = centers(e)
            distances = np.linalg.norm(poses-poses[j],axis=1)
            data = np.repeat(depth[...,None],2,axis=-1).astype(np.float32)
            states = {}
            for state in STATES:
                start = natural[i] if state=='natural' else truth*np.exp(STARTS[state])
                start = np.where(valid,start,np.nan)
                initial = np.log(start/truth)
                indices,eligible = select(j,start,k,e,data)[0]
                states[state] = {name:summarize_pixels(indices,eligible,m,initial) for name,m in zip(REGIONS,masks)}
                packed[regime+'_'+state] = np.where(eligible,1+indices,0).astype(np.uint8)
                if regime=='dispersed':
                    original = prior['views'][i][state]['reference_reference']['all']
                    for name,value in states[state]['all'].items():
                        np.testing.assert_allclose(value,original[name],rtol=1e-10,atol=1e-9)
            methods[regime] = dict(sources=[x for x in ids if x!=frame],
                                   baseline_m=[float(distances[n]) for n,x in enumerate(ids) if x!=frame],states=states)
        tmp = io.BytesIO()
        np.savez_compressed(tmp,**packed)
        records.append((dict(scene=scene,frame=frame,methods=methods),tmp.getvalue()))
        print(scene,frame,'scored all three four-source settings',flush=True)
    return records


def summarize(records):
    def group(items):
        result = {}
        for regime in REGIMES:
            states = {}
            for state in STATES:
                regions = {}
                for region in REGIONS:
                    cells = [r['methods'][regime]['states'][state][region] for r in items]
                    def ratio(a,b):
                        return mean(c[a]/c[b] for c in cells if c[b])
                    changed = sum(c['changed'] for c in cells)
                    regions[region] = dict(eligible=ratio('eligible','n'),changed=ratio('changed','n'),
                        exact=ratio('exact','n'),improved_among_changed=sum(c['improved'] for c in cells)/changed if changed else None,
                        mean_initial_mse=ratio('initial_sse','n'),mean_final_mse=ratio('final_sse','n'),
                        mean_relative_sse_reduction=1-ratio('final_sse','initial_sse') if state!='clean' else None)
                states[state] = regions
            result[regime] = dict(mean_baseline_m=mean(mean(r['methods'][regime]['baseline_m']) for r in items),states=states)
        return result
    return dict(overall=group(records),by_scene={s:group([r for r in records if r['scene']==s]) for s in SCENES})


def verify(data):
    frames = verify_inputs()['frames']
    old = read(OLD/'source_transplant.json')['batches']
    if (data['design_sha256']!=sha(DESIGN) or data['inputs_sha256']!=sha(INDEX) or data['choices_sha256']!=sha(CHOICES)
            or data['original_sha256']!=sha(OLD/'source_transplant.json') or
            [(r['scene'],r['frame']) for r in data['targets']]!=[(s,f) for s in SCENES for f in FRAMES]):
        raise ValueError('Reference-control identity differs')
    with zipfile.ZipFile(CHOICES) as z,zipfile.ZipFile(OLD_SUPPORT) as prior:
        for row in data['targets']:
            scene,frame = row['scene'],row['frame']
            index = FRAMES.index(frame)
            base = scene.split('/')[0]+'_base'
            available = unpack(prior.read(base+'.npz'))[index]
            with np.load(io.BytesIO(z.read(f'{base}/{frame}.npz')),allow_pickle=False) as f:
                shape = (392,518)
                def mask(name):
                    return np.unpackbits(f[name],count=np.prod(shape)).reshape(shape).astype(bool)
                valid = mask('valid')
                np.testing.assert_array_equal(mask('previously_available'),available&valid)
                if valid.sum()!=frames[scene][str(frame)]['projection_check']['valid_count']:
                    raise ValueError('Invalid target mask')
                for regime,method in row['methods'].items():
                    if method['sources']!=sources(frame,regime) or min(method['baseline_m'])<0 or len(method['baseline_m'])!=4:
                        raise ValueError('Different source cameras')
                    for state in STATES:
                        values=f[regime+'_'+state]
                        if values.shape!=shape or values.dtype!=np.uint8 or values.max()>5:
                            raise ValueError('Invalid choice raster')
                        columns=method['states'][state]
                        for name,m in zip(REGIONS,(valid,valid&available,valid&~available)):
                            eligible = m&(values>0)
                            hist = np.bincount(values[eligible]-1,minlength=5)
                            rec = columns[name]
                            if (int(m.sum())!=rec['n'] or hist.tolist()!=rec['eligible_hist'] or hist.sum()!=rec['eligible']
                                    or hist[1:].sum()!=rec['changed'] or not 0<=rec['improved']<=rec['changed']
                                    or rec['oracle_sse']>min(rec['initial_sse'],rec['final_sse'])+1e-9):
                                raise ValueError('Choice counts or depth sums differ')
                            if state in STARTS:
                                err = np.abs(STARTS[state]+LOG_STEPS)
                                found = int(hist[err<1e-8].sum())
                                if (found!=rec['eligible_exact'] or found+(rec['n']-rec['eligible'] if state=='clean' else 0)!=rec['exact']
                                        or int(hist[err<abs(STARTS[state])-1e-10].sum())!=rec['improved']):
                                    raise ValueError('Known-error outcome failed replay')
                        for key in columns['all']:
                            a,b = [np.asarray(columns[n][key]) for n in REGIONS[1:]]
                            np.testing.assert_allclose(a+b,columns['all'][key],rtol=1e-10,atol=1e-8)
                        if regime=='dispersed':
                            prev = old[base]['views'][index][state]['reference_reference']['all']
                            for key,v in columns['all'].items():
                                np.testing.assert_allclose(v,prev[key],rtol=1e-10,atol=1e-9)
    if summarize(data['targets'])!=data['results']:
        raise ValueError('Reference spacing summary changed')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','verify'))
    args = p.parse_args()
    if args.action=='evaluate':
        if SUMMARY.exists():
            raise FileExistsError(SUMMARY)
        rows=[]
        part = CHOICES.with_suffix('.partial')
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool,zipfile.ZipFile(part,'w') as out:
            for scene_records in pool.map(measure,SCENES):
                for row,blob in scene_records:
                    out.writestr(f"{row['scene'].split('/')[0]}_base/{row['frame']}.npz",blob)
                    rows.append(row)
        part.replace(CHOICES)
        data = dict(design_sha256=sha(DESIGN),inputs_sha256=sha(INDEX),original_sha256=sha(OLD/'source_transplant.json'),
                    choices_sha256=sha(CHOICES),targets=rows,results=summarize(rows),cloud_usd=0)
        verify(data)
        dump(SUMMARY,data)
    else:
        verify(read(SUMMARY))
    print('Verified all 15 targets, signed and natural depth tests, fixed support partitions and the original dispersed control.')
