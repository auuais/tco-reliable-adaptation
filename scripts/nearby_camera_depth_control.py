"""Separate camera and source-depth effects on the original 15 near-context targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing
from statistics import mean
import zipfile

import numpy as np

from collect_spacing_heads import CAPTURE, HEADS, tag
from depth_source_transplant import CELLS, RIGS, STATES, select, summarize_pixels
from dense_rgb_depth import LOG_STEPS
from evaluation_stability import dump, sha
from photometric_positive_control import STARTS
from predict_temporal_depth import SEAL
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read
from source_spacing_control import CHOICES as REFERENCE_CHOICES, SUMMARY as REFERENCE
from visibility_positive_control import PUBLIC as PREVIOUS


DESIGN = ROOT/'docs/NEARBY_CAMERA_DEPTH_CONTROL.md'
ARCHIVE = PUBLIC/'nearby_camera_depth_choices.zip'
RESULT = PUBLIC/'nearby_camera_depth.json'
PRIOR = PREVIOUS/'source_transplant.json'
REGIONS = ('all', 'common')


def measure(scene):
    key = scene.split('/')[0]+'_base'
    frames = read(INDEX)['frames'][scene]
    capture = read(CAPTURE)['records']
    predictions = read(SEAL)['batches'][key+'_short']
    metric_scale = read(PRIOR)['batches'][key]['scale_to_metres']
    _, original, original_k, _, digest = inputs(key, read(PARENT/'predictions.json')['rows'], read(PLAN))
    if digest != predictions['baseline_sha256']:
        raise ValueError('Original target geometry changed')
    rows = []
    with zipfile.ZipFile(REFERENCE_CHOICES) as old:
        for i, frame in enumerate(FRAMES):
            label = tag(scene, 'short', frame)
            record = capture[label]
            if record['ids'][i] != frame or sha(HEADS/(label+'.npz')) != record['archive_sha256']:
                raise ValueError('Wrong local capture')
            with np.load(HEADS/(label+'.npz'), allow_pickle=False) as f:
                depth = f['depth'][..., 0].astype(float)
                k, e = (f[n].astype(float) for n in ('intrinsics', 'extrinsics'))
            ratio = predictions['views'][i]['scale_to_original']
            if ratio != float(np.median(original[i]/depth[i])):
                raise ValueError('Different predicted depth gauge')
            k[:, [0, 1], [0, 1]] = original_k[i, [0, 1], [0, 1]]
            scale = metric_scale*ratio
            e[:, :3, 3] *= scale
            refs = []
            for n in record['ids']:
                path = RAW/'reference'/scene.split('/')[0]/f'{n:06d}.npz'
                if sha(path) != frames[str(n)]['reference_sha256']:
                    raise ValueError('Changed reference geometry')
                with np.load(path, allow_pickle=False) as f:
                    refs.append(dict(f))
            rk, re, truth, valid = (np.stack([v[n] for v in refs]) for n in ('k', 'e', 'depth', 'valid'))
            sources = np.stack([truth, np.where(valid, scale*depth, np.nan)], axis=-1).astype(np.float32)
            masks = dict(valid=np.packbits(valid[i]))
            observations = {}
            with np.load(io.BytesIO(old.read(f'{key}/{frame}.npz')), allow_pickle=False) as expected:
                for state in STATES:
                    start = (original[i]*metric_scale if state=='natural' else truth[i]*np.exp(STARTS[state]))
                    start = np.where(valid[i], start, np.nan)
                    initial = np.log(start/truth[i])
                    corners = {}
                    for rig, (intrinsics, extrinsics) in zip(RIGS, ((rk, re), (k, e))):
                        pair = select(i, start, intrinsics, extrinsics, sources)
                        corners.update({rig+'_'+channel: out for channel, out in zip(RIGS, pair)})
                    common = valid[i] & np.logical_and.reduce([ok for _, ok in corners.values()])
                    observations[state] = {}
                    for cell, (idx, ok) in corners.items():
                        labels = np.where(ok, idx+1, 0).astype(np.uint8)
                        masks[state+'_'+cell] = labels
                        if cell=='reference_reference':
                            np.testing.assert_array_equal(labels, expected['short_'+state])
                        observations[state][cell] = {
                            region:summarize_pixels(idx, ok, mask, initial)
                            for region, mask in zip(REGIONS, (valid[i], common))}
            buf = io.BytesIO()
            np.savez_compressed(buf, **masks)
            rows.append((dict(scene=scene, frame=frame, scale_to_metres=scale, states=observations), buf.getvalue()))
    print(scene, 'replayed reference controls and crossed both measurements', flush=True)
    return rows


def summarize(rows):
    def subset(group):
        result = {}
        for state in STATES:
            result[state] = {}
            for region in REGIONS:
                cells = {}
                for cell in CELLS:
                    items = [r['states'][state][cell][region] for r in group]
                    if min(v['n'] for v in items) <= 0:
                        raise ValueError('Empty target intersection')
                    def fraction(a, b):
                        return mean(v[a]/v[b] for v in items)
                    edits = sum(v['changed'] for v in items)
                    cells[cell] = dict(eligible=fraction('eligible','n'), exact=fraction('exact','n'),
                        changed=fraction('changed','n'), natural_sse_reduction=(
                            1-fraction('final_sse','initial_sse') if state=='natural' else None),
                        improved_among_changed=sum(v['improved'] for v in items)/edits if edits else None)
                result[state][region] = cells
        return result
    return dict(overall=subset(rows), by_scene={s:subset([r for r in rows if r['scene']==s]) for s in SCENES})


def provenance():
    return {p.name:sha(p) for p in (DESIGN, INDEX, CAPTURE, SEAL, PRIOR, REFERENCE, REFERENCE_CHOICES, ARCHIVE)}


def verify(report):
    if report['provenance'] != provenance() or report['cloud_usd'] != 0:
        raise ValueError('Nearby camera/depth provenance changed')
    rows = report['targets']
    if [(r['scene'], r['frame']) for r in rows] != [(s, f) for s in SCENES for f in FRAMES]:
        raise ValueError('Missing physical target')
    earlier = read(REFERENCE)['targets']
    with zipfile.ZipFile(ARCHIVE) as predictions, zipfile.ZipFile(REFERENCE_CHOICES) as reference:
        for row, old in zip(rows, earlier):
            name = f"{row['scene'].split('/')[0]}_base/{row['frame']}.npz"
            with np.load(io.BytesIO(predictions.read(name)), allow_pickle=False) as z, np.load(
                    io.BytesIO(reference.read(name)), allow_pickle=False) as previous:
                np.testing.assert_array_equal(z['valid'], previous['valid'])
                valid = np.unpackbits(z['valid'],count=392*518).reshape(392,518).astype(bool)
                for state in STATES:
                    maps = {c:z[state+'_'+c] for c in CELLS}
                    np.testing.assert_array_equal(maps['reference_reference'], previous['short_'+state])
                    common = valid & np.logical_and.reduce([d>0 for d in maps.values()])
                    for cell, d in maps.items():
                        if d.shape != valid.shape or d.dtype != np.uint8 or d.max()>5 or np.any(d[~valid]):
                            raise ValueError('Invalid nearby update map')
                        for region, mask in zip(REGIONS, (valid, common)):
                            r = row['states'][state][cell][region]
                            present = mask & (d>0)
                            hist = np.bincount(d[present]-1, minlength=5)
                            if (r['n']!=mask.sum() or hist.tolist()!=r['eligible_hist'] or hist.sum()!=r['eligible']
                                    or hist[1:].sum()!=r['changed'] or not 0<=r['improved']<=r['changed']
                                    or r['oracle_sse']>min(r['initial_sse'],r['final_sse'])+1e-9):
                                raise ValueError('Nearby decision accounting differs')
                            if state in STARTS:
                                exact = hist[np.abs(STARTS[state]+LOG_STEPS)<1e-8].sum()
                                if r['exact']!=exact+((mask.sum()-present.sum()) if state=='clean' else 0):
                                    raise ValueError('Injected-depth recovery differs')
                    for k, value in old['methods']['short']['states'][state]['all'].items():
                        np.testing.assert_allclose(row['states'][state]['reference_reference']['all'][k],
                                                   value, rtol=1e-10, atol=1e-9)
    if summarize(rows) != report['results']:
        raise ValueError('Nearby diagnostic summary differs')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('evaluate','verify'))
    if parser.parse_args().action=='evaluate':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = []
        part = ARCHIVE.with_suffix('.partial')
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool, zipfile.ZipFile(part,'w') as dst:
            for group in pool.map(measure, SCENES):
                for row, blob in group:
                    rows.append(row)
                    dst.writestr(f"{row['scene'].split('/')[0]}_base/{row['frame']}.npz", blob)
        part.replace(ARCHIVE)
        report = dict(provenance=provenance(), targets=rows, results=summarize(rows), cloud_usd=0)
        verify(report)
        dump(RESULT, report)
    else:
        verify(read(RESULT))
    print('Verified all four camera/depth combinations and the exact reference decision images for 15 targets.')
