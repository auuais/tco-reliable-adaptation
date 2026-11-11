"""Protect scene weighting when sequence and view counts differ in the held-out study."""
import unittest

from run_heldout_sparse_depth import summarize,GRIDS,REFS,METHODS


class HeldoutSceneWeighting(unittest.TestCase):
    def test_each_scene_has_equal_weight(self):
        seal={'batches':{}}
        observations={}
        counts={'office':4,'pumpkin':2,'redkitchen':5,'stairs':2}
        ratios={'office':.8,'pumpkin':.9,'redkitchen':1.,'stairs':1.1}
        for scene,n in counts.items():
            for i in range(n):
                key=f'{scene}/seq-{i:02d}'
                hashes={a:a for a in ('frozen','direct')}
                hashes.update({g+'_'+a:g+'_'+a for g in GRIDS for a in METHODS})
                seal['batches'][key]={'hashes':hashes}
                observations[key]={}
                for arm in hashes:
                    value=ratios[scene] if arm.endswith('_guarded') else 1.
                    observations[key][arm]={r:{'metrics':{'accuracy_mean_m':value,'completeness_mean_m':value}} for r in REFS}
        result=summarize(seal,observations)
        for grid in GRIDS:
            metric=result[grid]['references']['registered_rgb']['overall'][grid+'_guarded']
            self.assertAlmostEqual(metric['score'],.95)
            self.assertFalse(result[grid]['checks']['all_scenes'])
            self.assertFalse(result[grid]['advance'])
