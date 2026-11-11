"""Record coordinate quantization and duplicate geometry for the independent normal-score audit."""
import numpy as np

from evaluation_stability import dump,sha
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from prepare_tum_sparse_packet import RAW
from score_saved_geometry import read
from collect_matched_focal_heads import tensor_digest


def audit():
    seal=read(PUBLIC/'predictions.json')
    references=read(PUBLIC/'references.json')
    result={}
    for scene in SEQUENCES:
        ref=references['sequences'][scene]
        with np.load(RAW/'references'/ref['file']) as z:
            mask=z['valid_mask']
        result[scene]={}
        for arm in ('frozen','centered_guarded','shifted_guarded'):
            h=seal['batches'][scene]['hashes'][arm]
            with np.load(RAW/'points'/(h+'.npz')) as z:
                source=z['points']
            assert tensor_digest(source)==h
            xyz=np.ascontiguousarray(source[mask],dtype=np.float32)
            unique=len(np.unique(xyz,axis=0))
            lower_zero=float(np.mean((xyz.view(np.uint32)&0xffff)==0))
            result[scene][arm]=dict(prediction=h,points=len(xyz),unique_points=unique,
                                    duplicate_fraction=1-unique/len(xyz),bfloat16_representable_coordinate_fraction=lower_zero)
    dump(PUBLIC/'coordinate_precision.json',dict(prediction_sha256=sha(PUBLIC/'predictions.json'),cases=result))
    print({s:{a:(round(v['duplicate_fraction'],4),round(v['bfloat16_representable_coordinate_fraction'],4)) for a,v in arms.items()} for s,arms in result.items()})


if __name__=='__main__':
    audit()
