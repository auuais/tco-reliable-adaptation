"""Recompute first-source eligibility while replaying all five deployed depth rules."""
import argparse
import hashlib
import io
import zipfile

import numpy as np

from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump,sha
from prepare_dense_focal_pilot import ROOT
from run_dense_rgb_depth import PARENT,PLAN,inputs
from run_visibility_model import PUBLIC as MODEL,verify
from score_saved_geometry import read
from visibility_depth_evidence import MODEL_ARMS,estimate_views


PUBLIC = ROOT/'results/observation_selection'
DESIGN = ROOT/'docs/ELIGIBILITY_AND_SELECTION.md'
ZIP = PUBLIC/'source_support.zip'
SEAL = PUBLIC/'support.json'


def capture():
    if SEAL.exists():
        raise FileExistsError(SEAL)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    part = ZIP.with_suffix('.partial')
    model = verify()['batches']
    rows = read(PARENT/'predictions.json')['rows']
    plan = read(PLAN)
    obs = {}
    with zipfile.ZipFile(MODEL/'choices.zip') as prior,zipfile.ZipFile(part,'w') as out:
        for key,b in model.items():
            images,d,k,e,_ = inputs(key,rows,plan)
            with np.load(io.BytesIO(prior.read(key+'.npz')),allow_pickle=False) as f:
                decisions = dict(f)
            masks = []
            for i,neighbors,_,results in estimate_views(images,d,k,e,{'frozen':0.}):
                if neighbors!=b['views'][i]['source_order']:
                    raise ValueError('Recomputed source selection differs')
                for arm,spec in MODEL_ARMS.items():
                    index,eligible = results[spec]
                    if not np.array_equal(index,decisions[arm][i]) or int(eligible.sum())!=b['views'][i]['eligible'][arm]:
                        raise ValueError('Original deployed arm failed exact image replay')
                masks.append(results[MODEL_ARMS['single_depth']][1])
            mask = np.stack(masks)
            buf = io.BytesIO()
            np.savez_compressed(buf,packed=np.packbits(mask.ravel()))
            blob = buf.getvalue()
            out.writestr(key+'.npz',blob)
            obs[key] = dict(mask_sha256=tensor_digest(mask),packed_sha256=hashlib.sha256(blob).hexdigest(),
                            eligible_counts=mask.sum(axis=(1,2)).tolist(),exact_arm_view_replays=5*len(MODEL_ARMS))
            print(key,'reproduced five arms and saved initial source support',flush=True)
    part.replace(ZIP)
    dump(SEAL,dict(design_sha256=sha(DESIGN),model_sha256=sha(MODEL/'predictions.json'),zip_sha256=sha(ZIP),batches=obs))


def unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as f:
        return np.unpackbits(f['packed'],count=5*392*518).reshape(5,392,518).astype(bool)


def replay():
    saved = read(SEAL)
    model = verify()['batches']
    if (saved['design_sha256']!=sha(DESIGN) or saved['model_sha256']!=sha(MODEL/'predictions.json')
            or saved['zip_sha256']!=sha(ZIP) or set(saved['batches'])!=set(model)):
        raise ValueError('Eligibility study or source changed')
    with zipfile.ZipFile(ZIP) as z,zipfile.ZipFile(MODEL/'choices.zip') as source:
        for key,o in saved['batches'].items():
            blob = z.read(key+'.npz')
            m = unpack(blob)
            with np.load(io.BytesIO(source.read(key+'.npz')),allow_pickle=False) as f:
                single = f['single_depth']
            if (np.any((single!=0)&~m) or hashlib.sha256(blob).hexdigest()!=o['packed_sha256'] or tensor_digest(m)!=o['mask_sha256']
                    or o['exact_arm_view_replays']!=5*len(MODEL_ARMS) or m.sum(axis=(1,2)).tolist()!=o['eligible_counts']
                    or o['eligible_counts']!=[v['eligible']['single_depth'] for v in model[key]['views']]):
                raise ValueError('Replayed source support differs')
    return saved


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('capture','verify'))
    if p.parse_args().action=='capture':
        capture()
    else:
        replay()
        print('Verified each first-source support mask against the five-rule image replay and the original counts.')
