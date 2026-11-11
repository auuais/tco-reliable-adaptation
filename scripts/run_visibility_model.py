"""Seal five visibility/depth-evidence arms using frozen model cameras and depths."""
import argparse

import numpy as np

from dense_rgb_depth import FACTORS
from evaluation_stability import sha
from prepare_dense_focal_pilot import ROOT
from run_dense_rgb_depth import generate,verify_decisions
from score_saved_geometry import read
from visibility_depth_evidence import MODEL_ARMS,estimate_views
from visibility_positive_control import DESIGN,PUBLIC


RAW = ROOT/'reports/visibility_depth_evidence'
SOURCE = ROOT/'scripts/visibility_depth_evidence.py'


def predict(images,depths,k,poses):
    maps = {a:[] for a in MODEL_ARMS}
    views = []
    for i,neighbors,_,outputs in estimate_views(images,depths,k,poses,{'frozen':0.}):
        counts,hist = {},{}
        for arm,spec in MODEL_ARMS.items():
            indices,eligible = outputs[spec]
            maps[arm].append(indices)
            counts[arm] = int(eligible.sum())
            hist[arm] = np.bincount(indices.ravel(),minlength=len(FACTORS)).tolist()
        views.append(dict(view=i,source_order=neighbors,histograms=hist,eligible=counts))
    return {a:np.stack(v) for a,v in maps.items()},views


def verify():
    def support(maps,views):
        for i,v in enumerate(views):
            counts = v['eligible']
            if not 0<=counts['four_view_depth']<=counts['guarded']<=counts['single_depth']:
                raise ValueError('Source witnesses do not nest')
            if counts['checked_rgb']!=counts['guarded'] or counts['all_candidate_depth']>counts['guarded']:
                raise ValueError('Support selection depends on the cue')
            for arm,d in maps.items():
                if not np.count_nonzero(d[i])<=counts[arm]<=d[i].size:
                    raise ValueError('Unsupported model prediction')
            take = maps['guarded'][i]!=0
            if np.any(maps['guarded'][i][take]!=maps['single_depth'][i][take]):
                raise ValueError('Second witness altered the first-view proposal')
    return verify_decisions(PUBLIC,DESIGN,SOURCE,tuple(MODEL_ARMS),support)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('infer','verify'))
    if p.parse_args().action=='infer':
        if read(PUBLIC/'positive_control.json')['design_sha256']!=sha(DESIGN):
            raise ValueError('The reference-camera control must use the registered design')
        generate(predict,SOURCE,DESIGN,PUBLIC,RAW)
    else:
        verify()
        print('Verified all five sealed frozen-model choices and their independent witness support.')
