"""Audit the sealed sparse-depth predictions against metric target depth after inference."""
import numpy as np

from evaluation_stability import dump, sha
from neighbor_sparse_depth import PUBLIC, RAW, PACKET, SEAL, FRAMES, SCENES
from prepare_source_spacing import RAW as SPACING, INDEX
from run_dense_rgb_depth import inputs, PLAN, PARENT
from collect_matched_focal_heads import tensor_digest
from score_saved_geometry import read


def audit():
    seal = read(SEAL)
    source = read(INDEX)['frames']
    if sha(PACKET) != seal['packet_sha256']:
        raise ValueError('Changed packet')
    rows = []
    with np.load(PACKET) as packet:
        for scene in SCENES:
            slug = scene.split('/')[0]
            key = slug + '_base'
            proof = seal['batches'][key]
            _, depth, _, cameras, direct = inputs(key, read(PARENT/'predictions.json')['rows'], read(PLAN))
            assert direct == proof['direct']
            depths = {'direct': depth}
            for arm, digest in proof['predictions'].items():
                with np.load(RAW/'points'/(digest+'.npz')) as archive:
                    points = archive['points']
                assert tensor_digest(points) == digest
                depths[arm] = np.einsum('nhwj,nj->nhw', points, cameras[:, 2, :3]) + cameras[:, 2, 3, None, None]
            for i, frame in enumerate(FRAMES):
                path = SPACING/'reference'/slug/f'{frame:06d}.npz'
                assert sha(path) == source[scene][str(frame)]['reference_sha256']
                samples = packet[f'{slug}_{frame}']
                x, y = samples[:, :2].astype(int).T
                with np.load(path) as z:
                    truth, valid = z['depth'], z['valid'].astype(bool)
                good = valid[y, x]
                measurement_error = np.abs(samples[good, 2] - truth[y[good], x[good]])
                errors = {}
                for arm, predicted in depths.items():
                    estimate = predicted[i][valid] * proof['gauge']
                    if not np.all(np.isfinite(estimate) & (estimate > 0)):
                        raise ValueError('Invalid metric prediction on target mask')
                    errors[arm] = float(np.mean(np.abs(np.log(estimate/truth[valid]))))
                rows.append(dict(scene=scene, frame=frame, samples=len(samples), labeled_samples=int(good.sum()),
                    median_anchor_error_m=float(np.median(measurement_error)), mean_absolute_log_depth=errors))
    dump(PUBLIC/'metric_depth.json', dict(seal_sha256=sha(SEAL), rows=rows,
        mean_absolute_log_depth={a:float(np.mean([r['mean_absolute_log_depth'][a] for r in rows])) for a in errors}))


if __name__ == '__main__':
    audit()
