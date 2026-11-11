"""Measure direct and weighted surface distance at each rigid-alignment stage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from statistics import mean

import numpy as np
from scipy.spatial import cKDTree

import motion_reliability_surface as surface
import score_temporal_depth as temporal
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, ROOT, SCENES
from score_saved_geometry import read

DESIGN = ROOT/'docs/MOTION_ICP_EFFECT.md'
RESULT = PUBLIC/'motion_icp_effect.json'


def measure(scene):
    import open3d as o3d
    from run_repeat_calibration import umeyama
    key = scene.split('/')[0]+'_base'
    spec = read(temporal.PLAN)['references'][key]
    path = temporal.REF/spec['file']
    if sha(path)!=spec['sha256']: raise ValueError('Changed reference data')
    with np.load(path) as f:
        refs = dict(f)
    seal = read(surface.SEAL)['hashes'][key]
    base = read(temporal.SEAL)['batches'][key+'_short']['baseline_sha256']
    clouds = {'direct':temporal.load_points(temporal.OLD_POINTS,base),
              'weighted':temporal.load_points(surface.DATA,seal['weighted'])}
    mask = refs['valid_mask']
    result = {}
    for ref in temporal.REFS:
        target = refs[ref][mask]
        tree = cKDTree(target)
        dst = o3d.geometry.PointCloud()
        dst.points = o3d.utility.Vector3dVector(target)
        modes = {}
        for arm,points in clouds.items():
            scale,r,t = umeyama(points[mask].T,target.T)
            aligned = (scale*np.einsum('nhwj,ij->nhwi',points,r)+t.T)[mask]
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(aligned)
            icp = o3d.pipelines.registration.registration_icp(src,dst,.1,np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
            transform = icp.transformation
            if arm=='direct':
                direct_transform = transform.copy()
            src_tree = cKDTree(aligned)
            distances = {}
            for mode,mat in [('initial',np.eye(4)),('shared',direct_transform),('own',transform)]:
                if arm=='direct' and mode=='own':
                    distances[mode] = distances['shared']
                    continue
                rot,shift = mat[:3,:3],mat[:3,3]
                pred = aligned@rot.T+shift
                inv = (target-shift)@rot
                distances[mode] = dict(accuracy_mean_m=float(tree.query(pred,workers=2)[0].mean()),
                        completeness_mean_m=float(src_tree.query(inv,workers=2)[0].mean()))
            modes[arm] = dict(distances=distances,transform=transform.tolist(),fitness=icp.fitness,inlier_rmse=icp.inlier_rmse)
        result[ref] = modes
    print(scene,'evaluated both references',flush=True)
    return scene,result


def replay(obs):
    historic = temporal.controls()
    old = read(surface.RESULT)['observed']
    comparison = {}
    for ref in temporal.REFS:
        scenes = {}
        for scene in SCENES:
            key = scene.split('/')[0]+'_base'
            by_arm = obs[scene][ref]
            for arm,data in by_arm.items():
                expected = historic[scene][ref]['direct'] if arm=='direct' else old[key]['weighted'][ref]
                for m in temporal.METRICS[:2]:
                    np.testing.assert_allclose(data['distances']['own'][m],expected[m],rtol=0,atol=.00002)
                matrix = np.array(data['transform'])
                np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-8)
            def normalized(arm,mode):
                d = by_arm[arm]['distances'][mode]
                return mean(d[m]/historic[scene][ref]['frozen'][m] for m in temporal.METRICS[:2])
            scenes[scene] = {mode:{a:normalized(a,mode) for a in ('direct','weighted')} for mode in ('initial','shared','own')}
        overall = {mode:1-mean(s[mode]['weighted'] for s in scenes.values())/mean(s[mode]['direct'] for s in scenes.values())
                   for mode in ('initial','shared','own')}
        comparison[ref] = dict(scenes=scenes,overall_gain=overall)
    return comparison


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('measure','verify'))
    deps = {v.name:sha(v) for v in (surface.SEAL,surface.RESULT,temporal.SEAL,temporal.PLAN,temporal.PARENT/'scores.json')}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if p.parse_args().action=='measure':
        if RESULT.exists(): raise FileExistsError(RESULT)
        with ProcessPoolExecutor(max_workers=3) as pool:
            obs = dict(pool.map(measure,SCENES))
        dump(RESULT,dict(inputs=deps,observed=obs,results=replay(obs)))
    data = read(RESULT)
    if data['inputs']!=deps or replay(data['observed'])!=data['results']:
        raise ValueError('ICP comparison replay failed')
    print({r:v['overall_gain'] for r,v in data['results'].items()})
