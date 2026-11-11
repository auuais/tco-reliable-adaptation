"""Score native and FP32 point maps obtained from the same frozen local camera/depth heads."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from statistics import mean

import numpy as np

from collect_matched_focal_heads import PUBLIC, RAW, tensor_digest
from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import INPUT, REF, ROOT, jobs
from run_dense_focal_pilot import INPUT_PLAN, RUNS
from score_dense_focal_pilot import METRICS, REFS
from score_saved_geometry import read


def score(row):
    import open3d as o3d
    import torch
    from run_repeat_calibration import evaluate
    torch.set_num_threads(2)
    key = row['input_key']
    folder = RAW/key
    archive = folder/'heads_and_points.npz'
    index = read(INPUT_PLAN)
    rgb = index['inputs'][key]
    ref = index['references'][key]
    if sha(archive)!=row['archive_sha256'] or sha(INPUT/rgb['file'])!=rgb['sha256'] or sha(REF/ref['file'])!=ref['sha256']:
        raise ValueError('Prediction or evaluation assets changed')
    with np.load(archive,allow_pickle=False) as z:
        obs = dict(z)
    for name in ('native','fp32'):
        if tensor_digest(obs[name]) != row[name+'_sha256']:
            raise ValueError('Prediction tensor differs')
    historical = next(r for r in read(ROOT/'results/dense_focal_pilot/seal.json')['rows'] if r['job']['input_key']==key)
    with np.load(RUNS/historical['tag']/historical['initial']['file'],allow_pickle=False) as z:
        original = z['points']
    if tensor_digest(original) != row['sealed_3090_sha256']:
        raise ValueError('Historical tensor differs')
    delta = np.linalg.norm(obs['native'].astype(np.float64)-original,axis=-1)
    bridge = dict(exact=row['bridge_exact'],exact_points=int(np.count_nonzero(delta==0)),point_count=delta.size,
                  median_model_units=float(np.median(delta)),p95_model_units=float(np.quantile(delta,.95)),max_model_units=float(delta.max()))
    if bridge['exact'] != (bridge['exact_points']==bridge['point_count']):
        raise ValueError('Point differences and hash bridge disagree')
    y,x = np.mgrid[:392,:518]
    maximum = 0.
    for i in range(5):
        k = obs['intrinsics'][i].astype(np.float64)
        e = obs['extrinsics'][i].astype(np.float64)
        depth = obs['depth'][i,...,0].astype(np.float64)
        cam = np.stack(((x-k[0,2])*depth/k[0,0],(y-k[1,2])*depth/k[1,1],depth),axis=-1)
        xyz = (cam - e[:3,3]) @ e[:3,:3]
        maximum = max(maximum, float(np.max(np.abs(xyz-obs['fp32'][i]))))
    if maximum > 1e-5*max(1,float(np.abs(obs['fp32']).max())):
        raise ValueError('Independent double-precision unprojection disagrees')
    with np.load(INPUT/rgb['file'],allow_pickle=False) as z:
        images=torch.from_numpy(z['images'])
    with np.load(REF/ref['file'],allow_pickle=False) as z:
        references=dict(z)
    metrics = {}
    for label in REFS:
        metrics[label]={}
        for stage in ('native','fp32'):
            dst=folder/'scores'/label/stage
            dst.mkdir(parents=True,exist_ok=True)
            np.random.seed(42)
            torch.manual_seed(42)
            o3d.utility.random.seed(42)
            measured=evaluate(obs[stage],dict(images=images, pointclouds=references[label],valid_mask=references['valid_mask']),dst)
            if measured['valid_points']!=ref['valid_pixels'] or not all(np.isfinite(measured[k]) for k in METRICS):
                raise ValueError('Unexpected dense metrics')
            metrics[label][stage]={k:measured[k] for k in METRICS}
    out=dict(input_key=key,archive_sha256=row['archive_sha256'],reference_sha256=ref['sha256'],
             historical_native_bridge=bridge,fp32_max_absolute_vs_numpy_float64=maximum,metrics=metrics)
    dump(folder/'scored.json',out)
    print(key, 'scored both local arithmetic modes and depth references',flush=True)
    return out


def summarize(rows):
    by_key={r['input_key']:r for r in rows}
    cases={(j['scene'],j['case'],j['direction']):j['input_key'] for j in jobs()}
    if len(rows)!=len(by_key) or set(by_key)!=set(cases.values()) or len(cases)!=12:
        raise ValueError('Incomplete frozen comparison')
    result={}
    for ref in REFS:
        data=[by_key[key]['metrics'][ref] for key in cases.values()]
        means={m:mean(d['fp32'][m]/d['native'][m] for d in data) for m in METRICS[:2]}
        result[ref]=dict(fp32_over_native_distances=means,normalized_distance_score=mean(means.values()),
                         mean_normal_consistency={s:mean(d[s][METRICS[2]] for d in data) for s in ('native','fp32')})
    return dict(balanced_conditions=12,unique_batches=len(by_key),exact_bridges=sum(r['historical_native_bridge']['exact'] for r in rows),
                max_unprojection_check=max(r['fp32_max_absolute_vs_numpy_float64'] for r in rows),references=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('evaluate','verify'))
    parser.add_argument('--workers',type=int,choices=range(1,5),default=3)
    args=parser.parse_args()
    obs_path=PUBLIC/'capture.json'
    target=PUBLIC/'summary.json'
    if args.action=='evaluate':
        if target.exists():
            raise FileExistsError(target)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(score,read(obs_path)['rows']))
        dump(target,dict(capture_sha256=sha(obs_path),rows=rows,results=summarize(rows),cloud_usd=0))
    else:
        data=read(target)
        if data['capture_sha256']!=sha(obs_path) or data['results']!=summarize(data['rows']):
            raise ValueError('Local frozen sensitivity differs from the measured rows')
        print('Verified both arithmetic modes on nine image batches and both references.')
