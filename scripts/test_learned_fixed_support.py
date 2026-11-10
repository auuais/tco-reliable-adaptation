import unittest
import numpy as np
from learned_fixed_support import coverage,split_components


class SupportTests(unittest.TestCase):
    def test_transitive_tracks_cannot_cross_folds(self):
        pairs=[dict(i=0,j=1,xy_i=np.array([[1.,2.],[20.,30.]]),xy_j=np.array([[4.,5.],[40.,50.]])),
               dict(i=1,j=2,xy_i=np.array([[4.,5.]]),xy_j=np.array([[7.,8.]]))]
        self.assertEqual(split_components(pairs),2)
        self.assertEqual(pairs[0]["fold"][0],pairs[1]["fold"][0])

    def test_disconnected_dense_component_must_abstain(self):
        pair=dict(i=1,j=2,fold=np.tile([0,1],200))
        for fold in coverage([pair]):
            self.assertFalse(fold["eligible"])
            self.assertEqual(fold["connected_views"],1)
            self.assertEqual(fold["matches"],200)

    def test_coverage_matches_original_scorer(self):
        from fixed_match_validation import score_pointmaps
        rng=np.random.default_rng(42);points=rng.normal(size=(5,4,4,3))
        pairs=[dict(i=0,j=j,xy_i=np.tile([[1.,1.]],(50,1)),xy_j=np.tile([[2.,2.]],(50,1)),fold=np.tile([0,1],25)) for j in range(1,5)]
        expected=score_pointmaps(points,pairs)["folds"]
        for a,b in zip(coverage(pairs),expected):
            for key in ["eligible","matches","connected_views"]:self.assertEqual(a[key],b[key])


if __name__=="__main__":unittest.main()
