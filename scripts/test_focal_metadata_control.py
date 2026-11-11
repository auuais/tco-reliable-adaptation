import unittest

import numpy as np

from focal_metadata_control import decisions, focal_proposal, fold_gain, matched_tracks


class FocalMetadataControlTest(unittest.TestCase):
    def test_consistent_zoom_and_wrong_metadata_have_opposite_evidence(self):
        cameras = np.repeat(np.eye(4)[None],5,axis=0)
        cameras[:,:3,3] = [[0,0,0],[.24,.05,0],[.03,-.32,.04],[-.30,.14,.01],[.25,-.20,0]]
        k = np.repeat(np.array([[[500.,0,259],[0,500.,196],[0,0,1]]]),5,axis=0)
        self.assertEqual(focal_proposal(k),(None,None))
        observed = k.copy()
        observed[1,[0,1],[0,1]] *= 1.12
        view, proposed = focal_proposal(observed)
        self.assertEqual(view,1)
        np.testing.assert_array_equal(proposed,k[view])
        xyz = np.random.default_rng(2401).uniform([-1.3,-1.2,2],[1.3,1.2,5],size=(150,3))
        points = []
        for camera,matrix in zip(cameras,k):
            projected = (xyz + camera[:3,3]) @ matrix.T
            points.append(projected[:,:2]/projected[:,2:])
        folds = [[(i,j,points[i][side::2],points[j][side::2]) for i in range(5) for j in range(i+1,5)]
                 for side in (0,1)]
        cases, counts, _ = matched_tracks(folds,view,k[view,:2,2],1.12)
        self.assertGreater(min(counts),20)
        for case, should_edit in (("metadata_error", True), ("valid_zoom", False)):
            gains = [fold_gain(cameras,observed,proposed,view,pairs) for pairs in cases[case]]
            self.assertEqual(decisions(gains)["both_folds"],should_edit)
        for before,after in zip(cases["metadata_error"],cases["valid_zoom"]):
            for (i,j,a,b),(u,v,c,d) in zip(before,after):
                self.assertEqual((i,j),(u,v))
                np.testing.assert_array_equal(b if i==view else a,d if i==view else c)
                xy, zoom = (a,c) if i==view else (b,d)
                np.testing.assert_allclose(zoom,k[view,:2,2]+1.12*(xy-k[view,:2,2]))

    def test_two_fold_rule_abstains_when_either_fold_lacks_support(self):
        for gains in ([4.,None],[None,4.],[3.,-.4],[.24,3.]):
            self.assertFalse(decisions(gains)["both_folds"])


if __name__ == "__main__":
    unittest.main()
