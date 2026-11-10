"""Known geometry fixtures: gauge, local inconsistency and abstention."""
import unittest
import numpy as np
from fixed_match_validation import score_pointmaps, sample_points


class FixedMatchValidationTests(unittest.TestCase):
    def setUp(self):
        y,x=np.mgrid[:12,:16]
        plane=np.stack([x/5,y/7,2+.03*x],axis=-1)
        self.points=np.stack([plane,plane,plane])
        xy=np.stack([x.ravel(),y.ravel()],axis=-1).astype(float)
        self.pairs=[dict(i=i,j=i+1,xy_i=xy,xy_j=xy,fold=np.arange(len(xy))%2) for i in [0,1]]

    def test_consistent_geometry_and_single_view_error(self):
        self.assertEqual(score_pointmaps(self.points,self.pairs)["score"],0)
        moved=self.points.copy();moved[1]+=[.1,0,0]
        self.assertGreater(score_pointmaps(moved,self.pairs)["score"],.01)

    def test_global_similarity_does_not_change_error(self):
        moved=self.points.copy();moved[1]+=[.1,.2,-.1]
        angle=.73;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        transformed=7.3*moved@rotation.T+np.array([13,-5,8])
        self.assertAlmostEqual(score_pointmaps(moved,self.pairs)["score"],score_pointmaps(transformed,self.pairs)["score"],places=12)

    def test_missing_support_and_invalid_geometry_abstain(self):
        self.assertFalse(score_pointmaps(self.points,self.pairs[:1])["eligible"])
        self.assertFalse(score_pointmaps(np.zeros_like(self.points),self.pairs)["eligible"])
        invalid=self.points.copy();invalid[1,0,0,0]=np.nan
        self.assertFalse(score_pointmaps(invalid,self.pairs)["eligible"])
        one_fold=[dict(pair,fold=np.zeros_like(pair["fold"])) for pair in self.pairs]
        self.assertFalse(score_pointmaps(self.points,one_fold)["eligible"])

    def test_bilinear_sampling_recovers_affine_plane(self):
        xy=np.array([[1.25,3.5],[15,11]])
        expected=np.stack([xy[:,0]/5,xy[:,1]/7,2+.03*xy[:,0]],axis=-1)
        np.testing.assert_allclose(sample_points(self.points[0],xy),expected,rtol=1e-12)
        with self.assertRaises(ValueError):sample_points(self.points[0],np.array([[16,0]]))


if __name__=="__main__":unittest.main()
