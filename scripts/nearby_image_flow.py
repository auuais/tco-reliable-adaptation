"""Capture forward and reverse image motion before evaluating metric depth."""
import argparse
import time

import numpy as np

from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, ROOT, SCENES, sources
from score_saved_geometry import read


DESIGN = ROOT/'docs/NEARBY_RGB_CORRESPONDENCE.md'
CAPTURE = PUBLIC/'nearby_rgb_flows.json'
FLOW = RAW/'flow'
SHAPE = (392, 518, 2)


def pairs():
    for sequence in SCENES:
        for frame in FRAMES:
            for neighbor in sources(frame, 'short'):
                yield sequence, frame, neighbor


def tag(sequence, frame, neighbor):
    return f'{sequence.split("/")[0]}_{frame:06d}_{neighbor:06d}'


def grayscale(sequence, frame, source):
    import cv2
    path = RAW/'rgb'/sequence.split('/')[0]/f'{frame:06d}.npz'
    if sha(path) != source[str(frame)]['rgb_sha256']:
        raise ValueError('Changed input image')
    with np.load(path, allow_pickle=False) as f:
        rgb = f['rgb']
    image = np.rint(255*rgb.transpose(1,2,0)).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def capture():
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    source = read(INDEX)['frames']
    FLOW.mkdir(parents=True, exist_ok=True)
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    getters = ('FinestScale','GradientDescentIterations','PatchSize','PatchStride',
               'UseMeanNormalization','UseSpatialPropagation','VariationalRefinementIterations',
               'VariationalRefinementAlpha','VariationalRefinementDelta','VariationalRefinementGamma')
    settings = {name:getattr(engine, 'get'+name)() for name in getters}
    records = {}
    for seq, frame, neighbor in pairs():
        label = tag(seq, frame, neighbor)
        path = FLOW/(label+'.npz')
        checkpoint = path.with_suffix('.json')
        identity = dict(sequence=seq, frame=frame, neighbor=neighbor,
                        target_rgb_sha256=source[seq][str(frame)]['rgb_sha256'],
                        source_rgb_sha256=source[seq][str(neighbor)]['rgb_sha256'])
        if checkpoint.exists():
            record = read(checkpoint)
            if record['identity']!=identity or record['archive_sha256']!=sha(path) or record['version']!=cv2.__version__ or record['settings']!=settings:
                raise ValueError('Cached flow was produced using different inputs')
        else:
            first = grayscale(seq, frame, source[seq])
            second = grayscale(seq, neighbor, source[seq])
            start = time.perf_counter()
            forward = engine.calc(first, second, None)
            reverse = engine.calc(second, first, None)
            seconds = time.perf_counter()-start
            arrays = dict(forward=forward, reverse=reverse)
            if any(v.dtype!=np.float32 or v.shape!=SHAPE or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError('Invalid image flow')
            write_arrays(path, arrays)
            record = dict(identity=identity, seconds=seconds, version=cv2.__version__, settings=settings,
                          archive_sha256=sha(path), arrays={k:tensor_digest(v) for k,v in arrays.items()})
            dump(checkpoint, record)
        records[label] = record
    report = dict(design_sha256=sha(DESIGN), input_sha256=sha(INDEX), settings=settings,
                  version=cv2.__version__, preset='MEDIUM', threads=1, opencl=False, initialized=False, records=records)
    if CAPTURE.exists() and report!=read(CAPTURE):
        raise ValueError('Changed the recorded optical-flow experiment')
    dump(CAPTURE, report)


def verify():
    report = read(CAPTURE)
    frames = read(INDEX)['frames']
    if (report['design_sha256']!=sha(DESIGN) or report['input_sha256']!=sha(INDEX)
            or len(report['records'])!=60 or report['preset']!='MEDIUM' or report['opencl']
            or report['initialized'] or report['threads']!=1):
        raise ValueError('Incomplete image-motion capture')
    for seq, frame, neighbor in pairs():
        item = report['records'][tag(seq, frame, neighbor)]
        expected = dict(sequence=seq, frame=frame, neighbor=neighbor,
                        target_rgb_sha256=frames[seq][str(frame)]['rgb_sha256'],
                        source_rgb_sha256=frames[seq][str(neighbor)]['rgb_sha256'])
        if (item['identity']!=expected or item['seconds']<=0 or item['settings']!=report['settings']
                or item['version']!=report['version'] or set(item['arrays'])!={'forward','reverse'}
                or any(len(s)!=64 for s in [item['archive_sha256'],*item['arrays'].values()])):
            raise ValueError('Flow capture identity differs')
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture','verify'))
    if parser.parse_args().action=='capture':
        capture()
    verify()
    print('Verified the signed RGB-only motion capture manifest for 60 pairs.')
