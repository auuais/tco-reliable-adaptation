"""Sparse registered RGB-D samples from neighboring 7-Scenes frames, with fixed supplied poses."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from calibration_audit import resized_k
from collect_relative_focal_windows import FRAMES, SCENES
from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import point_map
from focal_zoom_observations import SCALE
from prepare_dense_focal_pilot import ROOT, write_arrays
from run_dense_rgb_depth import inputs, PLAN, PARENT
from score_depth_scale_study import aggregate, evaluate_points, HISTORICAL, REFS
from score_saved_geometry import read

PUBLIC = ROOT / 'results/neighbor_sparse_depth'
RAW = ROOT / 'reports/neighbor_sparse_depth'
PACKET = PUBLIC / 'measurements.npz'
SEAL = PUBLIC / 'predictions.json'
PANEL = ROOT / 'results/source_frame_spacing/profile_camera_scores.json'
ARMS = ('range_median', 'range_spatial', 'guarded')


def project_samples(mm, uv, rays, relative, target, source):
    xyz = (rays * (mm / 1000.)[:, None]) @ relative[:3, :3].T + relative[:3, 3]
    valid = (mm > 0) & (mm <= 10000) & (xyz[:, 2] > 0)
    xy = (xyz[valid] @ target.T)[:, :2] / xyz[valid, 2:3]
    pixel = np.rint(xy).astype(int)
    keep = ((pixel >= 0) & (pixel < [518, 392])).all(axis=1)
    return np.column_stack([pixel, xyz[valid, 2], np.full(len(pixel), source),
                                      uv[valid], mm[valid]])[keep]


def packet(assets, destination=PACKET, origin=(40, 30)):
    index = read(ROOT / 'results/source_frame_spacing/inputs.json')['frames']
    calibration = np.asarray(read(ROOT / 'manifests/sensor_frame_v14.json')['calibration'])
    native = resized_k(525, 640, 480)
    target = resized_k(525)
    target[[0, 1], [0, 1]] *= SCALE
    u, v = np.meshgrid(np.arange(origin[0], 640, 80), np.arange(origin[1], 480, 60))
    uv = np.column_stack([u.ravel(), v.ravel()])
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(native).T
    records = {}
    for scene in SCENES:
        def checked(frame, suffix):
            if suffix == 'depth.proj.png' and frame in FRAMES:
                raise ValueError('Target depth is evaluation-only')
            path = assets / scene / f'frame-{frame:06d}.{suffix}'
            if sha(path) != index[scene][str(frame)]['original'][suffix]:
                raise ValueError(str(path))
            return path

        def pose(frame):
            return calibration @ np.linalg.inv(np.loadtxt(checked(frame, 'pose.txt')))

        for frame in FRAMES:
            world_to_target = pose(frame)
            observed = []
            for source in (frame - 4, frame + 4):
                relative = world_to_target @ np.linalg.inv(pose(source))
                with Image.open(checked(source, 'depth.proj.png')) as image:
                    assert image.size == (640, 480)
                    mm = np.asarray(image)[uv[:, 1], uv[:, 0]]
                observed.extend(project_samples(mm, uv, rays, relative, target, source))
            nearest = {}
            for row in sorted(observed, key=lambda r: r[2]):
                nearest.setdefault(tuple(row[:2]), row)
            records[f'{scene.split("/")[0]}_{frame}'] = np.array(list(nearest.values()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(destination, records)


def corrections(depth, samples):
    height, width = depth.shape[1:]
    y_grid, x_grid = np.indices((height, width))
    query = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    ratios = []
    for predicted, data in zip(depth, samples):
        x, y = data[:, :2].T.astype(int)
        ratios.append(np.log(data[:, 2] / predicted[y, x]))
    gauge = np.median(np.concatenate(ratios))
    outputs = {arm: [] for arm in ARMS}
    for predicted, data, logs in zip(depth, samples, ratios):
        residual = np.clip(logs - gauge, -.15, .15)
        center = np.median(residual)
        xy = data[:, :2]
        x, y = xy.T.astype(int)
        distance, indices = cKDTree(xy).query(query, k=8)
        spatial = np.exp(-.5 * (distance / 48.) ** 2)
        edge = np.log(predicted.ravel()[:, None] / predicted[y, x][indices])
        outputs['range_median'].append(predicted * np.exp(center))
        for arm, weight in [('range_spatial', spatial), ('guarded', spatial * np.exp(-.5 * (edge / .08) ** 2))]:
            field = (center + np.sum(weight * residual[indices], axis=1)) / (1 + weight.sum(axis=1))
            outputs[arm].append(predicted * np.exp(field.reshape(height, width)))
    return outputs, float(np.exp(gauge))


def predict():
    rows = read(PARENT / 'predictions.json')['rows']
    batches = {}
    maps = RAW / 'points'
    maps.mkdir(parents=True, exist_ok=True)
    with np.load(PACKET, allow_pickle=False) as measurements:
        for scene in SCENES:
            slug = scene.split('/')[0]
            key = slug + '_base'
            _, depth, k, e, direct = inputs(key, rows, read(PLAN))
            samples = [measurements[f'{slug}_{frame}'] for frame in FRAMES]
            if min(map(len, samples)) < 8 or not np.all(np.isfinite(depth) & (depth > 0)):
                raise ValueError('Insufficient valid samples or invalid model depth')
            outputs, gauge = corrections(depth, samples)
            hashes = {}
            for arm, corrected in outputs.items():
                points = point_map(np.asarray(corrected), k, e, np.ones(len(depth)))
                hashes[arm] = tensor_digest(points)
                write_arrays(maps / (hashes[arm] + '.npz'), dict(points=points))
            batches[key] = dict(predictions=hashes, direct=direct, counts=list(map(len, samples)), gauge=gauge)
    record = dict(packet_sha256=sha(PACKET), design_sha256=sha(ROOT / 'docs/NEIGHBOR_SPARSE_DEPTH.md'), batches=batches)
    if SEAL.exists() and read(SEAL) != record:
        raise ValueError('Sealed predictions changed')
    dump(SEAL, record)


def score():
    sealed = read(SEAL)
    if sha(PACKET) != sealed['packet_sha256']:
        raise ValueError('Changed measurements')
    old = read(PANEL)['results']['references']
    rows = []
    with ProcessPoolExecutor(3) as pool:
        jobs = {key: pool.submit(evaluate_points, key, set(b['predictions'].values()), RAW)
                for key, b in sealed['batches'].items()}
        for scene in SCENES:
            key = scene.split('/')[0] + '_base'
            _, scores = jobs[key].result()
            metrics = {}
            for ref in REFS:
                control = {('previous_profile' if a == 'guarded' else a): b['raw']
                           for a, b in old[ref]['scenes'][scene].items()}
                metrics[ref] = dict(control)
                metrics[ref].update({arm: scores[h][ref] for arm, h in sealed['batches'][key]['predictions'].items()})
            rows.append(dict(scene=scene, case='base', metrics=metrics))
    controls = tuple(a for a in rows[0]['metrics'][REFS[0]] if a not in HISTORICAL + ('guarded',))
    result = aggregate(rows, controls + ('guarded',), controls)
    dump(PUBLIC / 'scores.json', dict(seal_sha256=sha(SEAL), panel_sha256=sha(PANEL), results=result))
    observations = {}
    for key, batch in sealed['batches'].items():
        observations[key] = {arm: {ref: read(RAW/'evaluation'/key/digest/ref/'checked.json') for ref in REFS}
                             for arm, digest in batch['predictions'].items()}
    dump(PUBLIC/'surface_observations.json', observations)
    print(result['checks'], result['local_gain'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'predict', 'score'])
    parser.add_argument('--assets', type=Path, help='Original public 7-Scenes dataset directory')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.assets is None:
            parser.error('--assets required')
        packet(args.assets)
    elif args.action == 'predict':
        predict()
    else:
        score()
