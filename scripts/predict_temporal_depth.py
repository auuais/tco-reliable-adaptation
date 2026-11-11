"""Correct original target depths using temporally selected frozen source observations."""
import argparse
import io
import time
import zipfile

import numpy as np

from collect_matched_focal_heads import tensor_digest
from collect_relative_focal_windows import FRAMES,SCENES
from collect_spacing_heads import CAPTURE,DESIGN,HEADS,tag,verify as verify_capture
from dense_rgb_depth import FACTORS
from depth_source_transplant import costs_for_sources
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import centers,point_map
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import PUBLIC,RAW,REGIMES
from run_dense_rgb_depth import PARENT,PLAN,inputs
from score_saved_geometry import read
from visibility_depth_evidence import screen


ARMS = ('context_native','context_calibrated','half_blend','single','context_checked','checked')
CHOICES = PUBLIC/'temporal_choices.zip'
SEAL = PUBLIC/'temporal_predictions.json'


def decisions(i,target,k,e,source):
    c = centers(e)
    neighbors = sorted((j for j in range(5) if j!=i),key=lambda j:(np.linalg.norm(c[j]-c[i]),j))
    y,x = np.indices(target.shape,dtype=float)
    ray = np.stack(((x-k[i,0,2])/k[i,0,0],(y-k[i,1,2])/k[i,1,1],np.ones_like(x)),axis=-1)@e[i,:3,:3]
    pair = []
    for j in neighbors:
        rot = e[j,:3,:3].T
        pair.append(costs_for_sources(target,ray@rot,c[i]@rot+e[j,:3,3],k[j],source[j])[0])
    rules = screen([p[0] for p in pair],[p[1] for p in pair])
    return rules['single'],rules['two_witness']


def infer():
    if SEAL.exists():
        raise FileExistsError(SEAL)
    capture = verify_capture()['records']
    old = read(PARENT/'predictions.json')['rows']
    plan = read(PLAN)
    dst = RAW/'points'
    dst.mkdir(parents=True,exist_ok=True)
    batches = {}
    part = CHOICES.with_suffix('.partial')
    with zipfile.ZipFile(part,'w') as packed:
        for scene in SCENES:
            key = scene.split('/')[0]+'_base'
            _,original,rig_k,rig_e,direct_hash = inputs(key,old,plan)
            focal = rig_k[:,[0,1],[0,1]]
            np.testing.assert_array_equal(focal,np.tile(focal[0],(5,1)))
            for regime in REGIMES[1:]:
                depths = {a:[] for a in ARMS}
                raw_k,info = [],[]
                decisions_by_arm = {a:[] for a in ('single','checked','context_checked')}
                for i,frame in enumerate(FRAMES):
                    name = tag(scene,regime,frame)
                    record = capture[name]
                    path = HEADS/(name+'.npz')
                    if record['ids'][i]!=frame or sha(path)!=record['archive_sha256']:
                        raise ValueError('Wrong temporal head bundle')
                    with np.load(path,allow_pickle=False) as f:
                        local = f['depth'][...,0].astype(float)
                        k,e = (f[n].astype(float) for n in ('intrinsics','extrinsics'))
                    raw_k.append(k[i].copy())
                    k[:,[0,1],[0,1]] = focal[0]
                    start = time.perf_counter()
                    ratio = float(np.median(original[i]/local[i]))
                    if not np.isfinite(ratio) or ratio<=0:
                        raise ValueError('Invalid reference-free scale alignment')
                    aligned = ratio*local[i]
                    source = np.repeat(local[...,None],2,axis=-1).astype(np.float32)
                    one,two = decisions(i,original[i]/ratio,k,e,source)
                    primary_seconds = time.perf_counter()-start
                    _,context = decisions(i,local[i],k,e,source)
                    rules = dict(single=one,checked=two,context_checked=context)
                    for a,(d,m) in rules.items():
                        decisions_by_arm[a].append(d)
                        depths[a].append(FACTORS[d]*(aligned if a=='context_checked' else original[i]))
                    depths['context_native'].append(aligned)
                    depths['context_calibrated'].append(aligned)
                    depths['half_blend'].append(np.sqrt(aligned*original[i]))
                    info.append(dict(view=i,frame=frame,bundle=name,scale_to_original=ratio,primary_seconds=primary_seconds,
                        eligible={a:int(m.sum()) for a,(_,m) in rules.items()},
                        histogram={a:np.bincount(d.ravel(),minlength=5).tolist() for a,(d,_) in rules.items()}))
                maps,times = {},{}
                for a in ARMS:
                    start=time.perf_counter()
                    points=point_map(np.stack(depths[a]),np.stack(raw_k) if a=='context_native' else rig_k,rig_e,np.ones(5))
                    times[a]=time.perf_counter()-start
                    h=tensor_digest(points)
                    write_arrays(dst/(h+'.npz'),dict(points=points))
                    maps[a]=h
                names = {a:np.stack(d) for a,d in decisions_by_arm.items()}
                buf = io.BytesIO()
                np.savez_compressed(buf,**names)
                label=key+'_'+regime
                packed.writestr(label+'.npz',buf.getvalue())
                batches[label] = dict(input_key=key,scene=scene,regime=regime,baseline_sha256=direct_hash,
                    views=info,point_hashes=maps,construction_seconds=times,choice_hashes={a:tensor_digest(d) for a,d in names.items()})
                print(label,'sealed six temporal geometry outputs',flush=True)
    part.replace(CHOICES)
    dump(SEAL,dict(design_sha256=sha(DESIGN),capture_sha256=sha(CAPTURE),choices_sha256=sha(CHOICES),batches=batches))


def verify():
    data = read(SEAL)
    baseline = read(PARENT/'predictions.json')['rows']
    capture = verify_capture()['records']
    if data['design_sha256']!=sha(DESIGN) or data['capture_sha256']!=sha(CAPTURE) or data['choices_sha256']!=sha(CHOICES):
        raise ValueError('Temporal inference provenance differs')
    expected={s.split('/')[0]+'_base_'+g for s in SCENES for g in REGIMES[1:]}
    if set(data['batches'])!=expected:
        raise ValueError('Incomplete temporal grid')
    with zipfile.ZipFile(CHOICES) as z:
        for label,b in data['batches'].items():
            if set(b['point_hashes'])!=set(ARMS) or any(b['baseline_sha256']!=r['predictions']['direct'] for r in baseline if r['input_key']==b['input_key']):
                raise ValueError('Invalid temporal comparator')
            with np.load(io.BytesIO(z.read(label+'.npz')),allow_pickle=False) as f:
                rules=dict(f)
            for a,d in rules.items():
                if tensor_digest(d)!=b['choice_hashes'][a] or d.shape!=(5,392,518) or d.dtype!=np.uint8 or d.max()>=5:
                    raise ValueError('Choice raster differs')
                for i,v in enumerate(b['views']):
                    h=np.bincount(d[i].ravel(),minlength=5).tolist()
                    if (h!=v['histogram'][a] or sum(h[1:])>v['eligible'][a] or capture[v['bundle']]['ids'][i]!=FRAMES[i]
                            or v['scale_to_original']<=0 or v['primary_seconds']<=0):
                        raise ValueError('Temporal inference support differs')
            active=rules['checked']!=0
            if np.any(rules['checked'][active]!=rules['single'][active]):
                raise ValueError('Confirmation changed the proposed direction')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('infer','verify'))
    if p.parse_args().action=='infer':
        infer()
    verify()
    print('Verified temporal depth candidates, gauges and unchanged-context controls for all 15 targets.')
