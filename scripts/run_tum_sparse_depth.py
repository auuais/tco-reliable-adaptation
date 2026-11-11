"""Seal the fixed TUM sparse-depth corrections, then evaluate the common motion-compensated mask."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import numpy as np

from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import point_map
from neighbor_sparse_depth import corrections
from prepare_dense_focal_pilot import write_arrays
from prepare_tum_sparse_packet import GRIDS,RAW
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from score_saved_geometry import read
from supplied_pose_sparse_depth import METHODS,place


def predict():
    inputs=read(PUBLIC/'inputs.json')
    captures={}
    batches={}
    maps=RAW/'points'
    maps.mkdir(parents=True,exist_ok=True)
    with np.load(PUBLIC/'measurements.npz') as sensors:
        if sha(PUBLIC/'measurements.npz')!=inputs['packet_sha256']:
            raise ValueError('Changed input measurements')
        for sequence in SEQUENCES:
            path=RAW/(sequence+'.npz')
            capture=PUBLIC/'captures'/(sequence+'.json')
            metadata=read(capture)
            assert sha(path)==metadata['archive_sha256']
            assert metadata['rgb_tensor_sha256']==inputs['sequences'][sequence]['rgb_tensor_sha256']
            captures[sequence]=sha(capture)
            with np.load(path) as z:
                depth=z['depth'][...,0].astype(float)
                model=z['extrinsics'].astype(float)
                frozen=z['native']
            k=sensors[sequence+'_intrinsics']
            supplied=sensors[sequence+'_poses']
            arms={'frozen':frozen,'direct':point_map(depth,k,model,np.ones(5))}
            details={}
            for grid in GRIDS:
                samples=[sensors[f'{sequence}_{grid}_{i}'] for i in range(5)]
                updates,gauge=corrections(depth,samples)
                updates['direct']=depth
                details[grid]=dict(gauge=gauge,counts=list(map(len,samples)))
                for arm in METHODS:
                    original=point_map(np.asarray(updates[arm]),k,model,np.ones(5))
                    arms[grid+'_'+arm]=place(original,model,supplied,gauge)
            hashes={}
            for arm,points in arms.items():
                h=tensor_digest(points)
                write_arrays(maps/(h+'.npz'),dict(points=points))
                hashes[arm]=h
            batches[sequence]=dict(hashes=hashes,grids=details)
            print(sequence,'ten predictions sealed',flush=True)
    record=dict(inputs_sha256=sha(PUBLIC/'inputs.json'),captures=captures,batches=batches)
    seal=PUBLIC/'predictions.json'
    if seal.exists() and read(seal)!=record:
        raise ValueError('Sealed TUM predictions changed')
    dump(seal,record)


def evaluate_sequence(sequence):
    import torch
    import open3d as o3d
    from run_repeat_calibration import evaluate
    torch.set_num_threads(2)
    seal=read(PUBLIC/'predictions.json')
    assert seal['inputs_sha256']==sha(PUBLIC/'inputs.json')
    capture=PUBLIC/'captures'/(sequence+'.json')
    assert seal['captures'][sequence]==sha(capture)
    assert read(capture)['archive_sha256']==sha(RAW/(sequence+'.npz'))
    assert read(PUBLIC/'inputs.json')['packet_sha256']==sha(PUBLIC/'measurements.npz')
    reference=read(PUBLIC/'references.json')
    assert reference['evaluation_selection_sha256']==sha(PUBLIC/'evaluation_selection.json')
    assert reference['prediction_seal_sha256']==sha(PUBLIC/'predictions.json')
    ref=reference['sequences'][sequence]
    ref_path=RAW/'references'/ref['file']
    assert sha(ref_path)==ref['sha256']
    with np.load(ref_path) as z:
        truth,mask,depth=z['points'],z['valid_mask'],z['depth']
    meta=read(PUBLIC/'inputs.json')['sequences'][sequence]
    with np.load(RAW/(sequence+'.npz')) as heads:
        model_camera=heads['extrinsics'].astype(float)
    with np.load(PUBLIC/'measurements.npz') as packet:
        supplied_camera=packet[sequence+'_poses']
    rgb_path=RAW/'inference'/meta['rgb_file']
    assert sha(rgb_path)==meta['rgb_sha256']
    with np.load(rgb_path) as z:
        images=torch.from_numpy(z['images'])
    output={}
    for digest in sorted(set(seal['batches'][sequence]['hashes'].values())):
        with np.load(RAW/'points'/(digest+'.npz')) as z:
            points=z['points']
        assert tensor_digest(points)==digest
        folder=RAW/'evaluation'/sequence/digest
        path=folder/'checked.json'
        identity=dict(prediction=digest,reference_sha256=ref['sha256'],valid_pixels=int(mask.sum()))
        if path.exists():
            row=read(path)
            assert row['identity']==identity
        else:
            folder.mkdir(parents=True,exist_ok=True)
            np.random.seed(42)
            torch.manual_seed(42)
            o3d.utility.random.seed(42)
            metrics=evaluate(points,dict(images=images,pointclouds=truth,valid_mask=mask),folder)
            row=dict(identity=identity,metrics=metrics)
            dump(path,row)
        metric_depth={}
        aliases=[a for a,h in seal['batches'][sequence]['hashes'].items() if h==digest]
        for grid in GRIDS:
            for arm in aliases:
                if arm in ('frozen','direct'):
                    cameras=model_camera
                    gauge=seal['batches'][sequence]['grids'][grid]['gauge']
                elif arm.startswith(grid+'_'):
                    cameras=supplied_camera
                    gauge=1.
                else:
                    continue
                axial=(np.einsum('nhwj,nj->nhw',points,cameras[:,2,:3])+cameras[:,2,3,None,None])*gauge
                if not np.all(np.isfinite(axial[mask]) & (axial[mask]>0)):
                    raise ValueError('Invalid metric depth on the common evaluation mask')
                metrics=[]
                for i in range(len(depth)):
                    predicted=axial[i][mask[i]]
                    truth_depth=depth[i][mask[i]]
                    metrics.append(dict(mean_absolute_log_depth=float(np.mean(np.abs(np.log(predicted/truth_depth)))),
                                        rmse_m=float(np.sqrt(np.mean((predicted-truth_depth)**2))),
                                        abs_relative=float(np.mean(np.abs(predicted-truth_depth)/truth_depth))))
                metric_depth.setdefault(grid,{})[arm]={m:mean(v[m] for v in metrics) for m in metrics[0]}
        row=dict(row,metric_depth=metric_depth)
        output[digest]=row
        print(sequence,digest[:8],'surface scored',flush=True)
    return sequence,output


def summarize(seal,observations):
    result={}
    for grid in GRIDS:
        sequences={}
        candidate=grid+'_guarded'
        for scene,b in seal['batches'].items():
            frozen=observations[scene][b['hashes']['frozen']]['metrics']
            cell={}
            for arm,h in b['hashes'].items():
                if arm in ('frozen','direct') or arm.startswith(grid+'_'):
                    raw=observations[scene][h]['metrics']
                    ratios=[raw[m]/frozen[m] for m in ('accuracy_mean_m','completeness_mean_m')]
                    cell[arm]=dict(score=mean(ratios),accuracy=ratios[0],completeness=ratios[1],raw=raw,
                                   metric_depth=observations[scene][h]['metric_depth'][grid][arm])
            sequences[scene]=cell
        overall={a:{m:mean(v[a][m] for v in sequences.values()) for m in ('score','accuracy','completeness')} for a in cell}
        for arm in overall:
            keys=sequences[SEQUENCES[0]][arm]['metric_depth']
            overall[arm]['metric_depth']={m:mean(v[arm]['metric_depth'][m] for v in sequences.values()) for m in keys}
        best=min((a for a in overall if a!=candidate),key=lambda a:overall[a]['score'])
        gain=1-overall[candidate]['score']/overall[best]['score']
        per_sequence={s:1-v[candidate]['score']/v[best]['score'] for s,v in sequences.items()}
        checks=dict(gain=gain>=.05,components=all(overall[candidate][m]<=1.02*overall[best][m] for m in ('accuracy','completeness')),
                    all_sequences=min(per_sequence.values())>=0)
        result[grid]=dict(overall=overall,sequences=sequences,best_control=best,gain=gain,per_sequence_gain=per_sequence,
                          checks=checks,advance=all(checks.values()))
    return result


def score():
    seal=read(PUBLIC/'predictions.json')
    with ProcessPoolExecutor(3) as pool:
        observations=dict(pool.map(evaluate_sequence,SEQUENCES))
    dump(PUBLIC/'observations.json',observations)
    result=summarize(seal,observations)
    dump(PUBLIC/'scores.json',dict(seal_sha256=sha(PUBLIC/'predictions.json'),grids=result))
    print({g:(v['gain'],v['checks']) for g,v in result.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','score'))
    args=p.parse_args()
    (predict if args.action=='predict' else score)()
