"""Evaluate sealed scale proposals locally with the existing two-reference evaluator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
from statistics import mean
import zipfile

import numpy as np

from collect_matched_focal_heads import PUBLIC as CAPTURES, tensor_digest
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import fit, guarded_fit
from prepare_dense_focal_pilot import INPUT, REF, ROOT
from run_depth_scale_study import ARMS, DESIGN, PLAN, PUBLIC, RAW, decode_systems, grid, name
from score_dense_focal_pilot import METRICS, REFS
from score_saved_geometry import read


LOCAL = ('frozen','direct','shared','five','guarded')
HISTORICAL = ('native_3090','tco_retain','tco_consensus','tco_adaptive')


def evaluate_batch(key):
    import open3d as o3d
    import torch
    from run_repeat_calibration import evaluate
    torch.set_num_threads(2)
    plan = read(PLAN)
    rgb, ref = plan['inputs'][key], plan['references'][key]
    if sha(INPUT/rgb['file'])!=rgb['sha256'] or sha(REF/ref['file'])!=ref['sha256']:
        raise ValueError('Evaluation inputs changed')
    with np.load(INPUT/rgb['file'],allow_pickle=False) as z:
        images = torch.from_numpy(z['images'])
    with np.load(REF/ref['file'],allow_pickle=False) as z:
        refs = dict(z)
    rows = [r for r in read(PUBLIC/'predictions.json')['rows'] if r['input_key']==key]
    digests = {h for r in rows for h in r['predictions'].values()}
    measured = {}
    for digest in sorted(digests):
        folder = RAW/'evaluation'/key/digest
        with np.load(RAW/'points'/(digest+'.npz'),allow_pickle=False) as z:
            points = z['points']
        if tensor_digest(points)!=digest:
            raise ValueError('Sealed point map changed')
        for label in REFS:
            dst = folder/label
            dst.mkdir(parents=True,exist_ok=True)
            path = dst/'checked.json'
            identity = dict(prediction=digest, ref_archive=ref['sha256'], reference=label, valid_pixels=ref['valid_pixels'])
            if path.exists():
                obs = read(path)
            else:
                np.random.seed(42)
                torch.manual_seed(42)
                o3d.utility.random.seed(42)
                metrics = evaluate(points,dict(images=images,pointclouds=refs[label],valid_mask=refs['valid_mask']),dst)
                if metrics['valid_points']!=ref['valid_pixels']:
                    raise ValueError('Evaluation mask changed')
                obs = dict(identity=identity,metrics={m:metrics[m] for m in METRICS})
                temp = path.with_suffix('.tmp')
                dump(temp,obs)
                temp.replace(path)
            if obs['identity']!=identity or not all(np.isfinite(obs['metrics'][m]) for m in METRICS):
                raise ValueError('Dense scores failed identity or finite checks')
            measured.setdefault(digest,{})[label] = obs['metrics']
        print(key,digest[:10],'both references scored',flush=True)
    return key,measured


def join_controls(rows):
    local = {r['input_key']:r for r in read(CAPTURES/'summary.json')['rows']}
    prior = read(ROOT/'results/dense_focal_pilot/summary.json')['rows']
    for row in rows:
        for ref in REFS:
            out = row['metrics'][ref]
            out['frozen'] = local[row['input_key']]['metrics'][ref]['fp32']
            runs = [r for r in prior if name(r['job'])==row['condition']]
            if len(runs)!=6:
                raise ValueError('Missing a historical seed or arm')
            for arm in HISTORICAL:
                if arm=='native_3090':
                    initial = [r['scores'][ref]['initial'] for r in runs]
                    if any(v!=initial[0] for v in initial):
                        raise ValueError('Initial 3090 scores differ within a condition')
                    out[arm] = initial[0]
                else:
                    pair = [r for r in runs if 'tco_'+r['job']['arm']==arm]
                    if sorted(r['job']['seed'] for r in pair)!=[42,123]:
                        raise ValueError('Invalid two-seed comparator')
                    out[arm] = {m:mean(r['scores'][ref]['final'][m] for r in pair) for m in METRICS}


def aggregate(rows):
    def average(group,ref):
        result = {}
        for arm in LOCAL+HISTORICAL:
            cells = [r['metrics'][ref] for r in group]
            relative = [mean(c[arm][m]/c['frozen'][m] for c in cells) for m in METRICS[:2]]
            result[arm] = dict(score=mean(relative),relative_accuracy=relative[0],relative_completeness=relative[1],
                               raw={m:mean(c[arm][m] for c in cells) for m in METRICS})
        return result
    references = {}
    for ref in REFS:
        references[ref] = dict(overall=average(rows,ref),
            scenes={s:average([r for r in rows if r['scene']==s],ref) for s in sorted({r['scene'] for r in rows})},
            cases={s:average([r for r in rows if r['case']==s],ref) for s in sorted({r['case'] for r in rows})})
    primary = references[REFS[0]]['overall']
    released = references[REFS[1]]['overall']
    best = min(('frozen','direct','shared'),key=lambda a:primary[a]['score'])
    historical = min(HISTORICAL,key=lambda a:primary[a]['score'])
    gain = 1-primary['guarded']['score']/primary[best]['score']
    historical_gain = 1-primary['guarded']['score']/primary[historical]['score']
    gain_by_scene = {s:1-d['guarded']['score']/d[best]['score'] for s,d in references[REFS[0]]['scenes'].items()}
    checks = dict(local_gain=gain>=.05,historical_gain=historical_gain>=.05,
        component_safety=all(primary['guarded'][m]<=1.02*primary[best][m] for m in ('relative_accuracy','relative_completeness')),
        all_scenes=min(gain_by_scene.values())>=0, released=released['guarded']['score']<=released[best]['score'])
    return dict(references=references,local_comparator=best,historical_comparator=historical,local_gain=gain,
                historical_gain=historical_gain,gain_by_scene=gain_by_scene,checks=checks,advance=all(checks.values()))


def replay(require_archive=False):
    sealed = read(PUBLIC/'predictions.json')
    if (sealed['design_sha256']!=sha(DESIGN) or sealed['input_plan_sha256']!=sha(PLAN)
            or sealed['systems_sha256']!=sha(PUBLIC/'match_systems.zip')
            or sealed['solver_sha256']!=sha(ROOT/'scripts/fixed_camera_depth_scale.py')):
        raise ValueError('Registration or solver evidence changed')
    if require_archive:
        backup = read(PUBLIC/'archive.json')
        files = backup['files']
        hashes = {h+'.npz' for row in sealed['rows'] for h in row['predictions'].values()}
        if (len(files)!=len(hashes) or {f['file'] for f in files}!=hashes or backup['verified_physical_drives']!=2
                or sum(f['bytes'] for f in files)!=backup['total_bytes']
                or any(f['bytes']<=0 or len(f['sha256'])!=64 for f in files)):
            raise ValueError('Raw prediction archive inventory differs')
    plan = read(PLAN)
    captures = {r['input_key']:r for r in read(CAPTURES/'capture.json')['rows']}
    with zipfile.ZipFile(PUBLIC/'match_systems.zip') as archive:
        if [r['condition'] for r in sealed['rows']]!=[name(j) for j in grid()]:
            raise ValueError('Changed condition order or coverage')
        for j,row in zip(grid(),sealed['rows']):
            if (any(row[k]!=j[k] for k in ('input_key','scene','case','direction'))
                    or row['heads_sha256']!=captures[j['input_key']]['archive_sha256']
                    or row['rgb_sha256']!=plan['inputs'][j['input_key']]['rgb_sha256']
                    or row['prior_sha256']!=plan['priors'][j['prior_key']]['sha256']):
                raise ValueError('Proposal input identity differs')
            train,test = decode_systems(archive.read(row['condition']+'.npz'))
            five,guarded,obs = guarded_fit(train,test,5)
            shared,obs['fitting'] = fit(train,5,shared=True)
            for arm,scale in zip(ARMS,(np.ones(5),shared,five,guarded)):
                np.testing.assert_allclose(scale,row['scales'][arm],atol=1e-9,rtol=0)
            for k in ('before','after'):
                a,b = obs.pop(k),row['decision'][k]
                if a is None or b is None:
                    if a is not b:
                        raise ValueError('Validation score missing')
                else:
                    np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)
            if any(v!=row['decision'][k] for k,v in obs.items()):
                raise ValueError('Held-back acceptance or match support differs')
    return sealed


def main(action,workers):
    sealed = replay(require_archive=action=='verify')
    target = PUBLIC/'scores.json'
    if action=='fits':
        print('Replayed all 12 fits, validation decisions and four scale arms.')
        return
    if action=='evaluate':
        if target.exists():
            raise FileExistsError(target)
        keys = sorted({r['input_key'] for r in sealed['rows']})
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            scores = dict(pool.map(evaluate_batch,keys))
        rows = []
        for r in sealed['rows']:
            obs = {k:r[k] for k in ('condition','input_key','scene','case','direction')}
            obs['metrics'] = {ref:{arm:scores[r['input_key']][h][ref] for arm,h in r['predictions'].items()} for ref in REFS}
            rows.append(obs)
        join_controls(rows)
        dump(target,dict(seal_sha256=sha(PUBLIC/'predictions.json'),rows=rows,results=aggregate(rows),cloud_usd=0))
    else:
        saved = read(target)
        if (saved['seal_sha256']!=sha(PUBLIC/'predictions.json') or
                [r['condition'] for r in saved['rows']]!=[r['condition'] for r in sealed['rows']]):
            raise ValueError('Score provenance or coverage differs')
        for row,s in zip(saved['rows'],sealed['rows']):
            for key in ('condition','input_key','scene','case','direction'):
                if row[key]!=s[key]:
                    raise ValueError('Evaluation identity differs')
            for ref in REFS:
                for m in row['metrics'][ref].values():
                    if (not all(np.isfinite(m[k]) for k in METRICS) or min(m[k] for k in METRICS[:2])<=0
                            or not 0<=m[METRICS[2]]<=1):
                        raise ValueError('Invalid evaluation metric')
        controls = deepcopy(saved['rows'])
        join_controls(controls)
        if aggregate(saved['rows'])!=saved['results'] or controls!=saved['rows']:
            raise ValueError('Comparator or aggregate failed replay')
        print('Replayed both dense reference summaries and all development gates.')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('fits','evaluate','verify'))
    p.add_argument('--workers',type=int,choices=range(1,5),default=3)
    args = p.parse_args()
    main(args.action,args.workers)
