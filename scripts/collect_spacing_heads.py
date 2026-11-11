"""Capture local frozen head outputs for temporal five-view bundles without reference data."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from collect_matched_focal_heads import RAW as ORIGINAL,tensor_digest
from collect_relative_focal_windows import FRAMES,SCENES,UPSTREAM
from evaluation_stability import dump,sha
from focal_zoom_observations import MODEL_SHA,digest_large,image_digest
from prepare_dense_focal_pilot import ROOT,write_arrays
from prepare_source_spacing import INDEX,PUBLIC,RAW,REGIMES,sources
from run_dense_focal_pilot import check_source
from score_saved_geometry import read


DESIGN = ROOT/'docs/TEMPORAL_DEPTH_REDESIGN.md'
HEADS = RAW/'heads'
CAPTURE = PUBLIC/'frozen_heads.json'
FIELDS = ('depth','extrinsics','intrinsics')


def jobs():
    for scene in SCENES:
        yield scene,'dispersed',None,list(FRAMES)
        for regime in REGIMES[1:]:
            for i,frame in enumerate(FRAMES):
                ids = sources(frame,regime)
                ids.insert(i,frame)
                yield scene,regime,frame,ids


def tag(scene,regime,target):
    return scene.split('/')[0]+'_'+regime+('' if target is None else f'_{target:06d}')


def capture(checkpoint):
    import torch
    from selected_readouts import enable
    from stage_initial_forward import stage_initial_forward
    sys.path[:0] = [str(UPSTREAM),str(UPSTREAM/'base_models/vggt')]
    from tco_vggt_lora import TCO_VGGT_LoRA
    check_source()
    if digest_large(checkpoint/'model.safetensors')!=MODEL_SHA:
        raise ValueError('Unexpected model weights')
    source = read(INDEX)['frames']
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    model = TCO_VGGT_LoRA.from_pretrained(str(checkpoint),enable_track=False).eval()
    model.point_head = None
    retained = enable(model)
    stage_initial_forward(model,lambda name,value:None)
    HEADS.mkdir(parents=True,exist_ok=True)
    code = sha(Path(__file__))
    records = {}
    for scene,regime,target,ids in jobs():
        key = tag(scene,regime,target)
        path = HEADS/(key+'.npz')
        record = HEADS/(key+'.json')
        images = []
        for number in ids:
            p = RAW/'rgb'/scene.split('/')[0]/f'{number:06d}.npz'
            if sha(p)!=source[scene][str(number)]['rgb_sha256']:
                raise ValueError('RGB input file differs')
            with np.load(p,allow_pickle=False) as f:
                images.append(f['rgb'])
        rgb = np.stack(images)
        identity = dict(scene=scene,regime=regime,target=target,ids=ids,rgb_sha256=image_digest(rgb),collector_sha256=code)
        if record.exists():
            old = read(record)
            if any(old[k]!=v for k,v in identity.items()) or sha(path)!=old['archive_sha256']:
                raise ValueError('Incomplete or conflicting previous capture')
            records[key] = old
            continue
        model.to('cuda')
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            image = torch.from_numpy(rgb).unsqueeze(0).to('cuda')
            tokens,outputs = model.forward_original_vggt(image)
            arrays = {n:outputs[n][0].float().cpu().numpy() for n in FIELDS}
            del image,tokens,outputs
        elapsed=time.perf_counter()-start
        if (arrays['depth'].shape!=(5,392,518,1) or arrays['extrinsics'].shape!=(5,4,4) or arrays['intrinsics'].shape!=(5,3,3)
                or not all(np.isfinite(v).all() for v in arrays.values()) or arrays['depth'].min()<=0):
            raise ValueError('Invalid temporal frozen outputs')
        digests = {n:tensor_digest(v) for n,v in arrays.items()}
        bridge = None
        if regime=='dispersed':
            with np.load(ORIGINAL/(scene.split('/')[0]+'_base')/'heads_and_points.npz',allow_pickle=False) as previous:
                bridge = {n:dict(identical=digests[n]==tensor_digest(previous[n]),
                    max_absolute_difference=float(np.max(np.abs(arrays[n]-previous[n])))) for n in FIELDS}
        write_arrays(path,arrays)
        result = dict(**identity,archive_sha256=sha(path),head_sha256=digests,seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),archived_dispersed_bridge=bridge)
        dump(record,result)
        records[key]=result
        print(key,'captured',round(elapsed,2),'seconds',flush=True)
    value = dict(design_sha256=sha(DESIGN),index_sha256=sha(INDEX),model_sha256=MODEL_SHA,retained_layers=retained,
                 device=torch.cuda.get_device_name(),records=records,cloud_usd=0)
    if CAPTURE.exists() and value!=read(CAPTURE):
        raise ValueError('Frozen capture record differs')
    dump(CAPTURE,value)


def verify():
    d = read(CAPTURE)
    expected = list(jobs())
    if (d['model_sha256']!=MODEL_SHA or d['design_sha256']!=sha(DESIGN) or d['index_sha256']!=sha(INDEX)
            or len(d['records'])!=len(expected) or d['cloud_usd']!=0):
        raise ValueError('Frozen source-spacing inventory differs')
    for scene,regime,target,ids in expected:
        r=d['records'][tag(scene,regime,target)]
        if (r['ids']!=ids or r['target']!=target or r['scene']!=scene or r['regime']!=regime or r['seconds']<=0
                or r['peak_allocated_bytes']<=0 or set(r['head_sha256'])!=set(FIELDS)):
            raise ValueError('Invalid frozen bundle')
        if regime=='dispersed' and any(v['identical']!=(v['max_absolute_difference']==0) for v in r['archived_dispersed_bridge'].values()):
            raise ValueError('Invalid bridge to original dispersed heads')
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('capture','verify'))
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args()
    if args.action=='capture':
        if args.checkpoint is None:
            p.error('--checkpoint is required for capture')
        capture(args.checkpoint)
    verify()
    print('Verified 3 dispersed controls and 30 temporal five-view forward records.')
