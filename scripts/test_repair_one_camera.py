import unittest

import numpy as np

from repair_one_camera import fit_camera, perturb_camera, virtual_error


class CameraRepairTest(unittest.TestCase):
    def test_known_camera_corruption_on_fresh_synthetic_correspondences(self):
        rng=np.random.default_rng(48001)
        e=np.repeat(np.eye(4)[None],5,axis=0)
        e[:,:3,3]=[[0,0,0],[.24,.05,0],[.03,-.32,.04],[-.30,.14,.01],[.25,-.20,0]]
        k=np.repeat(np.array([[[500.,0.,259.],[0.,500.,196.],[0.,0.,1.]]]),5,axis=0)
        world=rng.uniform([-1,-1,2],[1,1,5],(80,3))
        uv=[]
        for cam,intr in zip(e,k):
            p=(world @ cam[:3,:3].T + cam[:3,3]) @ intr.T
            uv.append(p[:,:2]/p[:,2:])
        pairs=[(i,j,uv[i],uv[j]) for i in range(5) for j in range(i+1,5)]
        observed_e,observed_k=e.copy(),k.copy()
        observed_e[0],observed_k[0]=perturb_camera(e[0],k[0],[.06,-.02,.01],[.015,-.018,.012],np.log(1.12))
        initial=virtual_error(e[0],k[0],observed_e[0],observed_k[0])
        corrected=fit_camera(observed_e,observed_k,pairs,0)
        self.assertIsNotNone(corrected)
        self.assertLess(virtual_error(e[0],k[0],*corrected),.25*initial)
        self.assertLess(virtual_error(e[1],k[1],e[1],k[1]),1e-10)
        self.assertIsNone(fit_camera(observed_e,observed_k,pairs[:1],0))


if __name__=="__main__":
    unittest.main()
