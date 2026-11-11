"""Capture frozen depth and cameras once, then evaluate two unprojections of the same heads."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from evaluation_stability import dump, sha
from focal_zoom_observations import MODEL_SHA, UPSTREAM, digest_large, image_digest
from prepare_dense_focal_pilot import INPUT, ROOT, write_arrays
from run_dense_focal_pilot import INPUT_PLAN, check_source
from score_saved_geometry import read


RAW = ROOT/'reports/matched_focal_heads'
PUBLIC = ROOT/'results/matched_focal_heads'


def tensor_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def collect(checkpoint):
    import torch
    from selected_readouts import enable
    from stage_initial_forward import stage_initial_forward
    sys.path[:0] = [str(UPSTREAM), str(UPSTREAM/'base_models/vggt')]
    from tco_vggt_lora import TCO_VGGT_LoRA
    from utils.geometry import unproject_depth_to_world_points_torch as unproject
    check_source()
    if digest_large(checkpoint/'model.safetensors') != MODEL_SHA:
        raise ValueError('Unexpected VGGT checkpoint')
    index = read(INPUT_PLAN)['inputs']
    seal = ROOT/'results/dense_focal_pilot/seal.json'
    native_hashes = {key:{r['initial']['tensor_sha256'] for r in read(seal)['rows'] if r['job']['input_key']==key}
                     for key in index}
    if len(index) != 9 or any(len(v)!=1 for v in native_hashes.values()):
        raise ValueError('Expected nine unique sealed original baseline tensors')
    RAW.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    model = TCO_VGGT_LoRA.from_pretrained(str(checkpoint),enable_track=False).eval()
    model.point_head = None  # The original frozen pass calls only camera_head and depth_head.
    retained = enable(model)
    phases = {}
    stage_initial_forward(model, lambda name,value: phases.update({name:value}))
    observations = []
    for key, source in sorted(index.items()):
        directory = RAW/key
        directory.mkdir(exist_ok=True)
        path, record = directory/'heads_and_points.npz', directory/'observation.json'
        expected_hash, = native_hashes[key]
        if record.exists():
            obs = read(record)
            if (obs['archive_sha256'] != sha(path) or obs['rgb_sha256'] != source['rgb_sha256']
                    or obs['sealed_3090_sha256'] != expected_hash or obs['collector_sha256'] != sha(Path(__file__))):
                raise ValueError('Previous collection differs: '+key)
            observations.append(obs)
            continue
        image = INPUT/source['file']
        if sha(image) != source['sha256']:
            raise ValueError('Changed input archive: '+key)
        with np.load(image,allow_pickle=False) as f:
            rgb = f['images']
        if image_digest(rgb) != source['rgb_sha256']:
            raise ValueError('Changed RGB tensor')
        model.to('cuda')
        phases.clear()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            images = torch.from_numpy(rgb).unsqueeze(0).to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens, heads = model.forward_original_vggt(images)
                operands = [heads[n] for n in ('depth','extrinsics','intrinsics')]
                native = unproject(*operands)[0].float().cpu().numpy()
            with torch.autocast('cuda',enabled=False):
                fp32 = unproject(*(v.float() for v in operands))[0].float().cpu().numpy()
            arrays = {n:heads[n][0].float().cpu().numpy() for n in ('depth','extrinsics','intrinsics')}
            del tokens, heads, operands, images
        if (native.shape != (5,392,518,3) or fp32.shape != native.shape or
                not all(np.isfinite(v).all() for v in (*arrays.values(),native,fp32))):
            raise ValueError('Invalid model outputs')
        elapsed = time.perf_counter()-start
        write_arrays(path,dict(**arrays,native=native,fp32=fp32))
        native_sha = tensor_digest(native)
        obs = dict(input_key=key,rgb_sha256=source['rgb_sha256'],archive_sha256=sha(path),
                   collector_sha256=sha(Path(__file__)),sealed_3090_sha256=expected_hash,
                   native_sha256=native_sha,fp32_sha256=tensor_digest(fp32),bridge_exact=(native_sha==expected_hash),
                   max_change_model_units=float(np.max(np.abs(fp32-native))),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   instrumented_seconds=elapsed,device=torch.cuda.get_device_name(),phases=phases.copy(),
                   head_sha256={n:tensor_digest(v) for n,v in arrays.items()})
        dump(record,obs)
        observations.append(obs)
        print(key, 'sealed RTX 3090 map matches:', obs['bridge_exact'],flush=True)
    result = dict(seal_sha256=sha(seal),model_sha256=MODEL_SHA,retained_layers=retained,
                  rows=observations,exact_native_bridges=sum(r['bridge_exact'] for r in observations),cloud_usd=0)
    out = PUBLIC/'capture.json'
    if out.exists() and read(out) != result:
        raise ValueError('Existing public capture differs')
    dump(out,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    collect(parser.parse_args().checkpoint)
