"""Generated fixtures only. No released images, depth, geometry, networks or GPU involved."""
import hashlib
import io
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import prepare_tco_hypersim_replication as intake
from tco_hypersim_geometry import GL_TO_CV, camera_from_metadata, convert_frame, world_to_camera
from tco_hypersim_range import SceneArchive


def camera(tilt):
    rays=np.array([[.56,0,.01],[0,.42,-.01],[tilt,tilt/2,-1.]])
    inv=np.linalg.inv(rays)
    projection=np.eye(4)
    projection[[0,1,3],:3]=inv
    projection[[0,1,3],3]=0
    row={"settings_output_img_width":128,"settings_output_img_height":96}
    for prefix,M in (("M_cam_from_uv",rays),("M_proj",projection)):
        row.update({f"{prefix}_{i}{j}":x for (i,j),x in np.ndenumerate(M)})
    return row


class Bridge(unittest.TestCase):
    def test_tilted_rays_radial_depth_gl_pose_units_and_positions(self):
        cm=camera_from_metadata(camera(.035))
        w,h=cm["size"]
        y,x=np.indices((h,w))
        source=np.stack((x,y,np.ones_like(x)),axis=-1)
        rays=source@cm["native_to_ray"].T
        views=rays*(1.8/rays[...,2:3])
        # Independent OpenGL-to-world construction from the declared source pose.
        rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        centre=np.array([10.,-18.,39.])
        unit=.0254
        pos=(views@GL_TO_CV@rot.T / unit+centre).astype(np.float16)
        radial=np.linalg.norm(views,axis=-1).astype(np.float16)
        extr=world_to_camera(rot,centre,unit)
        rgb=np.empty((h,w,3),np.uint8)
        rgb[:]=[189,22,43]
        result,points,mask,checked=convert_frame(rgb,radial,cm,extr,pos,unit)
        self.assertTrue(checked["bridge_passed"])
        self.assertTrue(mask.all())
        self.assertTrue(np.allclose(result[:,0,0],np.array([189,22,43])/255))
        self.assertEqual(points.shape[:2],mask.shape)
        # Treating Euclidean distance as optical-axis depth gives the wrong world points.
        wrong_depth=(radial.astype(float)*np.linalg.norm(rays/rays[...,2:3],axis=-1)).astype(np.float16)
        self.assertFalse(convert_frame(rgb,wrong_depth,cm,extr,pos,unit)[-1]["bridge_passed"])
        self.assertFalse(convert_frame(rgb,radial,cm,extr,pos,unit*1.2)[-1]["bridge_passed"])

    def test_scene_and_frame_order_does_not_depend_on_geometry(self):
        poses={i:world_to_camera(np.eye(3),np.array([i*.02,0,0]),1) for i in (0,2,4,6,8,9,10,11,12)}
        ids,base=intake.window([12,10,8,6,0,11,4,2,9],poses)
        self.assertEqual(ids,[0,2,4,6,8])
        self.assertAlmostEqual(base,.16)
        self.assertEqual(intake.window([0,2],poses),(None,None))
        self.assertNotIn("position",intake.member_names("ai_001_010","cam_00",0,False))

    def test_selective_zip_and_refusal_of_unranged_body(self):
        scene="ai_123_456"
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(scene+"/_detail/metadata_scene.csv",b"unit")
            archive.writestr(scene+"/unused.bin",bytes(range(255))*4)
        content=stream.getvalue()
        receives=[]
        class Reply(io.BytesIO):
            pass
        def open_url(req,timeout):
            if req.method=="HEAD":
                r=Reply(b"")
                r.headers={"Content-Length":str(len(content)),"ETag":'"test"'}
                return r
            first,last=map(int,req.get_header("Range")[6:].split("-"))
            receives.append((first,last))
            r=Reply(content[first:last+1])
            r.status=206
            r.headers={"Content-Range":f"bytes {first}-{last}/{len(content)}","ETag":'"test"'}
            return r
        def mock_url(req,timeout):
            if req.get_method()=="HEAD":
                return open_url(req,timeout)
            req.method="GET"
            return open_url(req,timeout)
        url="https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/"+scene+".zip"
        with TemporaryDirectory() as folder,patch("tco_hypersim_range.urlopen",mock_url):
            with SceneArchive(url,Path(folder)) as ar:
                data,digest=ar.member(scene+"/_detail/metadata_scene.csv")
            self.assertEqual(data,b"unit")
            self.assertEqual(digest,hashlib.sha256(b"unit").hexdigest())
            self.assertTrue(receives)
            self.assertFalse((Path(folder)/scene/"unused.bin").exists())
        def ignore_range(req,timeout):
            if req.get_method()=="HEAD":
                return open_url(req,timeout)
            r=Reply(b"")
            r.status=200
            r.headers={}
            return r
        with TemporaryDirectory() as folder,patch("tco_hypersim_range.urlopen",ignore_range):
            with self.assertRaisesRegex(ValueError,"requested archive range"):
                SceneArchive(url,Path(folder))

    def test_missing_supported_primary_prevents_archive_access(self):
        with TemporaryDirectory() as folder,patch.object(intake,"PUBLIC",Path(folder)),patch.object(intake,"SceneArchive") as network:
            with self.assertRaisesRegex(ValueError,"Awaiting supported"):
                intake.lock("train")
            network.assert_not_called()

    @unittest.skipUnless(find_spec("h5py"),"staged Hypersim Python environment supplies h5py")
    def test_generated_five_frame_hdf5_lock_then_bridge(self):
        import h5py
        from PIL import Image
        scene,track="ai_123_001","cam_00"
        row=camera(.035)
        unit=.0254
        row["settings_units_info_meters_scale"]=unit
        ids=list(range(5))
        rotations=np.tile(np.eye(3),(5,1,1))
        centres=np.array([[i*2.,1.,-1.] for i in ids])
        def pack(data):
            stream=io.BytesIO()
            with h5py.File(stream,"w") as archive:
                archive.create_dataset("dataset",data=data)
            return stream.getvalue()
        meta=scene+"/_detail/"
        source={meta+"metadata_scene.csv":f"parameter_name,parameter_value\nmeters_per_asset_unit,{unit}\n".encode()}
        for key,a in (("frame_indices",np.array(ids)),("orientations",rotations),("positions",centres)):
            source[meta+track+"/camera_keyframe_"+key+".hdf5"]=pack(a)
        cm=camera_from_metadata(row)
        w,h=cm["size"]
        y,x=np.indices((h,w))
        rays=np.stack((x,y,np.ones_like(x)),axis=-1)@cm["native_to_ray"].T
        views=1.5*rays/rays[...,2:]
        for i in ids:
            names=intake.member_names(scene,track,i,True)
            stream=io.BytesIO()
            Image.fromarray(np.full((h,w,3),20*i+15,np.uint8)).save(stream,format="JPEG")
            source[names["rgb"]]=stream.getvalue()
            source[names["range"]]=pack(np.linalg.norm(views,axis=-1).astype(np.float16))
            source[names["position"]]=pack((views@GL_TO_CV/unit+centres[i]).astype(np.float16))
        accesses=[]
        class Archive:
            def __init__(self,*args):
                self.names=set(source)
                self.remote=self
                self.etag='"fixture"'
                self.length=10000
                self.received=0
            def __enter__(self):
                return self
            def __exit__(self,*args):
                pass
            def member(self,name):
                accesses.append(name)
                data=source[name]
                self.received+=len(data)
                return data,hashlib.sha256(data).hexdigest()
        metadata=({scene:row},{scene:"train"},{scene:{track:ids}},{scene:"fixture"})
        from contextlib import ExitStack, redirect_stdout
        with TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            base=Path(tmp)
            for k,v in (("PUBLIC",base/"public"),("MANIFEST",base/"registered.json"),("SceneArchive",Archive)):
                stack.enter_context(patch.object(intake,k,v))
            for k,value in (("contrast_gate",{"candidate":"Refresh1","primary_gate_sha256":"fixture"}),
                            ("exposures",set()),("check_committed",None),("metadata",metadata)):
                stack.enter_context(patch.object(intake,k,return_value=value))
            intake.lock("train")
            self.assertFalse(any("/images/" in a for a in accesses))
            intake.materialize("train")
            result=intake.read(base/"public"/"hypersim_bridge.json")
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["checks"]),5)


if __name__=="__main__":
    unittest.main()
