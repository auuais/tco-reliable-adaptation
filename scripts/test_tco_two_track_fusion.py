"""Signed gauge, z-buffer, intrinsics, fallback and log-median controls on synthetic geometry."""
import unittest
import numpy as np

from tco_two_track_fusion import fit_rig,fuse,splat,unproject


def fixture():
    ext=np.tile(np.eye(4),(5,1,1))
    ext[:,0,3]=-np.arange(5)*.1
    k=np.tile(np.array([[8.,0.,4.],[0.,9.,3.],[0.,0.,1.]]),(5,1,1))
    d=np.full((5,7,9),2.)
    return dict(extrinsics=ext,intrinsics=k,depth=d)


class FusionTest(unittest.TestCase):
    def test_known_scale_rotation_and_translation(self):
        p=fixture()
        e=p["extrinsics"]
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        shift=np.array([.6,-.3,.7])
        c=np.c_[np.arange(5)*.1,np.zeros((5,2))]
        prior=e.copy()
        prior[:,:3,:3]=rotation.T
        world=1.7*c @ rotation.T+shift
        prior[:,:3,3]=np.einsum("vij,vj->vi",-prior[:,:3,:3],world)
        r=fit_rig(e,prior)
        np.testing.assert_allclose(r["rotation"],rotation,atol=1e-12)
        np.testing.assert_allclose(r["translation"],shift,atol=1e-12)
        self.assertAlmostEqual(r["scale"],1.7)
        alternative=dict(extrinsics=prior,depth=p["depth"]*1.7,intrinsics=p["intrinsics"])
        points,receipt=fuse([p,alternative,alternative],prior)
        expected=unproject(p["depth"][0]*1.7,p["intrinsics"][0],rotation.T,shift)
        np.testing.assert_allclose(points["fusion"][0],expected,atol=2e-7)
        self.assertFalse(receipt["failed_gauges"])

    def test_plane_changed_intrinsics_and_holes(self):
        p=fixture()
        k=p["intrinsics"].copy()
        k[:,0,2]+=2.
        alt=dict(p,intrinsics=k)
        output,meta=fuse([p,alt,alt],p["extrinsics"])
        np.testing.assert_array_equal(output["fusion"],output["project0"])
        row=meta["views"][0]
        self.assertEqual(row["fusion_anchor_fallback"],14)
        self.assertEqual(row["anchor_fill"],[0,14,14])

    def test_nearest_depth_and_occlusion_sign(self):
        k=fixture()["intrinsics"][0]
        world=np.array([[0.,0.,3.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,1.]])
        raster,n=splat(world,k,np.eye(3),np.zeros(3),(7,9))
        self.assertEqual(n,2)
        self.assertEqual(raster[3,4],1.)
        self.assertEqual(np.isfinite(raster).sum(),1)

    def test_two_source_log_median_and_degenerate_fallback(self):
        p=fixture()
        far=dict(p,depth=p["depth"]*4)
        absent=dict(p,depth=np.full_like(p["depth"],np.nan))
        points,meta=fuse([p,far,absent],p["extrinsics"])
        np.testing.assert_allclose(points["fusion"][0,...,2],4.)
        self.assertEqual(meta["views"][0]["sources_per_pixel"][2],63)
        degenerate=dict(p,extrinsics=np.tile(np.eye(4),(5,1,1)))
        points,meta=fuse([p,far,degenerate],p["extrinsics"])
        np.testing.assert_array_equal(points["fusion"],points["project0"])
        self.assertEqual(meta["failed_gauges"],[2])

    def test_duplicate_source_identity(self):
        p=fixture()
        p["depth"][:,1:4,2:5]+=.7
        output,meta=fuse([p,p,p],p["extrinsics"])
        for v in output.values():
            np.testing.assert_array_equal(v,output["project0"])
        self.assertEqual(sum(r["fusion_anchor_fallback"] for r in meta["views"]),0)


if __name__=="__main__":
    unittest.main()
