"""Synthetic CPU checks for dense conversion; never open study NPZ/reference data."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

import profiled_m3_points as geometry


ROOT = Path(__file__).resolve().parents[1]


def pinned_functions(path,names):
    tree = ast.parse((ROOT/path).read_text())
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(functions)!=len(names):
        raise AssertionError("Pinned function inventory differs")
    namespace = {"np":np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def fixtures(shape=geometry.NATIVE_SHAPE):
    views,height,width=shape
    z = np.broadcast_to(np.linspace(1.5,2.5,width,dtype=np.float32),(views,height,width)).copy()
    k = np.tile(np.array([[470.4,0,255.9],[0,470.4,191.9],[0,0,1]],dtype=np.float32),(views,1,1))
    camera = np.tile(np.eye(4,dtype=np.float32),(views,1,1))
    for view in range(views):
        angle=.1*view
        camera[view,:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        camera[view,:3,3]=[1.+.2*view,-.1*view,2.+.05*view]
    native=dict(depth=z,intrinsics=k,c2w=camera)
    g0=np.array(1.7,dtype=np.float64)
    scales=np.linspace(.9,1.1,views,dtype=np.float64)
    metric=camera.astype(np.float64)
    metric[:,:3,3]=float(g0)*(metric[:,:3,3]-metric[0,:3,3])
    final=dict(depth=(scales*float(g0))[:,None,None]*z.astype(np.float64),intrinsics=k.copy(),
               c2w=metric,g0=g0,scales=scales)
    return native,final


def array_hash(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


class M3PointFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native,cls.final=fixtures()
        cls.converted=geometry.convert_endpoints(cls.native,cls.final)

    def test_exact_pinned_production_resize(self):
        pinned=pinned_functions("scripts/score_mast3r_transfer.py",{"resample"})["resample"]
        expected,k=pinned(self.final["depth"],self.final["intrinsics"].astype(float))
        np.testing.assert_array_equal(self.converted["final"]["depth_m"],expected)
        np.testing.assert_array_equal(self.converted["final"]["intrinsics"],k)

    def test_exact_pinned_point_map_arithmetic(self):
        pinned=pinned_functions("scripts/fixed_camera_depth_scale.py",{"point_map","displacements","centers"})["point_map"]
        for name in ("native","final"):
            value=self.converted[name]
            expected=pinned(value["depth_m"],value["intrinsics"],np.linalg.inv(value["c2w"])[:,:3,:],np.ones(5))
            np.testing.assert_array_equal(value["points"],expected)
            self.assertEqual(value["points"].dtype,np.float32)

    def test_independent_2x2_bilinear_and_half_pixel_intrinsics(self):
        depth=np.array([[[1.,2.],[3.,5.]]])
        k=np.array([[[2.,0.,.5],[0.,3.,.5],[0.,0.,1.]]])
        resized,adjusted=geometry.resize_depth_intrinsics(depth,k,(4,4))
        manual=np.empty((4,4))
        for y in range(4):
            for x in range(4):
                sx=np.clip((x+.5)/2-.5,0,1);sy=np.clip((y+.5)/2-.5,0,1)
                manual[y,x]=(1-sy)*((1-sx)*1+sx*2)+sy*((1-sx)*3+sx*5)
        np.testing.assert_array_equal(resized[0],manual)
        np.testing.assert_array_equal(adjusted[0],[[4.,0.,1.5],[0.,6.,1.5],[0.,0.,1.]])

    def test_independent_C2W_inverse_K_convention(self):
        depth=np.array([[[1.,2.],[3.,4.]]])
        k=np.array([[[2.,0.,.25],[0.,3.,.75],[0.,0.,1.]]])
        camera=np.eye(4)[None]
        camera[0,:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        camera[0,:3,3]=[1.,2.,3.]
        actual=geometry.world_points(depth,k,camera)
        y,x=np.indices((2,2));pixel=np.stack([x,y,np.ones_like(x)],axis=-1)
        expected=(pixel@np.linalg.inv(k[0]).T*depth[0,...,None])@camera[0,:3,:3].T+camera[0,:3,3]
        np.testing.assert_allclose(actual[0],expected,rtol=1e-7,atol=3e-7)
        wrong=geometry.world_points(depth,k,np.linalg.inv(camera))
        self.assertFalse(np.allclose(wrong[0],expected))

    def test_production_sampled_inverse_K_C2W_matches_with_float32_bound(self):
        xy=np.array([[0,0],[517,391],[260,197],[51,337]])
        pixels=np.column_stack([xy,np.ones(len(xy))])
        for endpoint in ("native","final"):
            values=self.converted[endpoint]
            for view in range(5):
                camera=values["c2w"][view]
                z=values["depth_m"][view,xy[:,1],xy[:,0]]
                rays=pixels@np.linalg.inv(values["intrinsics"][view]).T
                direct=(rays*z[:,None])@camera[:3,:3].T+camera[:3,3]
                actual=values["points"][view,xy[:,1],xy[:,0]]
                # Native float32 rotations are approximately orthogonal; the
                # pinned inverse/transpose path adds that error to output f32
                # rounding. This fixture bounds their combined effect.
                bound=4*np.finfo(np.float32).eps*max(1.,np.max(np.abs(direct)))
                np.testing.assert_allclose(actual,direct,rtol=0,atol=bound)

    def test_final_already_metric_native_has_no_relative_scales(self):
        main,native=self.converted["final"],self.converted["native"]
        np.testing.assert_array_equal(main["c2w"],self.final["c2w"])
        np.testing.assert_array_equal(native["c2w"],self.final["c2w"])
        np.testing.assert_allclose(main["depth_m"],self.final["scales"][:,None,None]*native["depth_m"],rtol=2e-15,atol=2e-15)
        self.assertFalse(np.array_equal(main["depth_m"],native["depth_m"]))
        np.testing.assert_array_equal(main["c2w"][0,:3,3],np.zeros(3))

    def test_global_native_origin_shift_cancels_without_realigning(self):
        native,final=deepcopy((self.native,self.final))
        # Exact float64 translation avoids inventing additional float32 input
        # rounding when testing the coordinate identity.
        native["c2w"]=native["c2w"].astype(float)
        native["c2w"][:,:3,:3]=np.eye(3)
        final["c2w"][:,:3,:3]=np.eye(3)
        native["c2w"][:,:3,3]+=[10.,-20.,30.]
        checked=geometry.validate_endpoints(native,final)
        np.testing.assert_allclose(checked["native_metric_c2w"],final["c2w"],rtol=0,atol=6e-15)
        np.testing.assert_array_equal(checked["final_metric_depth"],final["depth"])

    def test_global_native_units_compensated_once_by_g0(self):
        native,final=deepcopy((self.native,self.final))
        native["depth"]*=2.
        native["c2w"][:,:3,3]*=2.
        final["g0"]/=2.
        converted=geometry.convert_endpoints(native,final)
        for name in ("native","final"):
            for field in geometry.POINT_FIELDS:
                np.testing.assert_array_equal(converted[name][field],self.converted[name][field])

    def test_double_g0_depth_and_translation_rejected(self):
        native,final=deepcopy((self.native,self.final))
        final["depth"]*=final["g0"]
        with self.assertRaisesRegex(ValueError,"Final depth"):
            geometry.validate_endpoints(native,final)
        native,final=deepcopy((self.native,self.final))
        final["c2w"][:,:3,3]*=final["g0"]
        with self.assertRaisesRegex(ValueError,"Final centers"):
            geometry.validate_endpoints(native,final)

    def test_relative_scale_must_not_scale_centers(self):
        native,final=deepcopy((self.native,self.final))
        final["c2w"][:,:3,3]*=final["scales"][:,None]
        with self.assertRaisesRegex(ValueError,"Final centers"):
            geometry.validate_endpoints(native,final)

    def test_native_final_field_bijections_and_production_shape(self):
        for which,key in (("native","depth"),("final","g0"),("final","scales")):
            native,final=deepcopy((self.native,self.final))
            value=native if which=="native" else final
            value.pop(key)
            with self.assertRaisesRegex(ValueError,"inventory"):
                geometry.validate_endpoints(native,final)
        for which in ("native","final"):
            native,final=deepcopy((self.native,self.final))
            (native if which=="native" else final)["reference_depth"]=np.array(1.)
            with self.assertRaisesRegex(ValueError,"inventory"):
                geometry.validate_endpoints(native,final)
        native,final=fixtures((5,2,2))
        with self.assertRaisesRegex(ValueError,"shape"):
            geometry.convert_endpoints(native,final)
        self.assertEqual(list(inspect.signature(geometry.convert_endpoints).parameters),["native","final"])

    def test_nonpositive_or_nonfinite_gauge_scale_and_wrong_dtypes_rejected(self):
        for field,bad in (("g0",np.array(0.)),("g0",np.array(float("nan"))),
                          ("g0",np.array([1.])),("scales",np.zeros(5)),("scales",np.ones(4)),
                          ("scales",np.ones(5,dtype=np.int64))):
            native,final=deepcopy((self.native,self.final));final[field]=bad
            with self.assertRaises(ValueError):
                geometry.validate_endpoints(native,final)
        native,final=deepcopy((self.native,self.final));native["depth"]=native["depth"].astype(np.float16)
        with self.assertRaises(ValueError):
            geometry.validate_endpoints(native,final)

    def test_changed_K_R_skew_projective_and_nonrigid_output_rejected(self):
        native,final=deepcopy((self.native,self.final));final["intrinsics"][0,0,2]+=1.
        with self.assertRaisesRegex(ValueError,"intrinsics must equal"):
            geometry.validate_endpoints(native,final)
        native,final=deepcopy((self.native,self.final));final["c2w"][0,0,0]=.99
        with self.assertRaisesRegex(ValueError,"rotations must equal"):
            geometry.validate_endpoints(native,final)
        for i,j,value in ((0,1,.01),(1,0,.01),(2,0,.001)):
            native,final=deepcopy((self.native,self.final))
            native["intrinsics"][0,i,j]=value;final["intrinsics"][0,i,j]=value
            with self.assertRaises(ValueError):
                geometry.validate_endpoints(native,final)
        native,final=deepcopy((self.native,self.final))
        native["c2w"][0,:3,:3]*=.999;final["c2w"][0,:3,:3]=native["c2w"][0,:3,:3]
        with self.assertRaisesRegex(ValueError,"optimized rotations"):
            geometry.validate_endpoints(native,final)

    def test_dtype_aware_depth_relationship(self):
        native,final=deepcopy((self.native,self.final))
        final["depth"][0,0,0]+=1e-7
        with self.assertRaisesRegex(ValueError,"Final depth"):
            geometry.validate_endpoints(native,final)
        native,final=deepcopy((self.native,self.final))
        final["depth"]=final["depth"].astype(np.float32)
        final["depth"][0,0,0]=np.nextafter(final["depth"][0,0,0],np.float32(np.inf))
        result=geometry.validate_endpoints(native,final)
        self.assertEqual(result["audit"]["depth_consistency"]["rtol"],64*np.finfo(np.float32).eps)

    def test_nonfinite_artifacts_and_float32_point_overflow_rejected(self):
        native,final=deepcopy((self.native,self.final));native["depth"][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            geometry.validate_endpoints(native,final)
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            geometry.world_points(np.full((1,2,2),1e100),np.eye(3)[None],np.eye(4)[None])

    def test_inputs_are_not_mutated_or_aliased(self):
        native,final=deepcopy((self.native,self.final))
        hashes={side:{k:array_hash(v) for k,v in value.items()} for side,value in (("native",native),("final",final))}
        converted=geometry.convert_endpoints(native,final)
        converted["final"]["depth_m"][0,0,0]=99.
        converted["native"]["c2w"][0,0,0]=99.
        for side,value in (("native",native),("final",final)):
            self.assertEqual(hashes[side],{k:array_hash(v) for k,v in value.items()})

    def test_fixed_mask_counting_preserves_all_points_and_cannot_filter(self):
        converted=deepcopy(self.converted)
        mask=np.zeros(geometry.DENSE_SHAPE,dtype=bool);mask[:,0,0]=True
        converted["final"]["depth_m"][0,0,0]=0.
        converted["native"]["depth_m"][1,0,0]=-1.
        converted["final"]["depth_m"][0,1,1]=-2.
        before=array_hash(converted["final"]["points"]);mask_before=array_hash(mask)
        result=geometry.fixed_mask_validity(converted,mask)
        self.assertEqual(result["mask_count"],5)
        self.assertEqual(result["evaluated_mask_count"],5)
        self.assertEqual(result["main_nonpositive_depth"],1)
        self.assertEqual(result["native_nonpositive_depth"],1)
        self.assertEqual(result["detail"]["final"]["nonpositive_outside_mask"],1)
        self.assertFalse(result["candidate_main_depth_valid"])
        self.assertEqual(result["candidate_dependent_exclusions"],0)
        self.assertEqual(before,array_hash(converted["final"]["points"]))
        self.assertEqual(mask_before,array_hash(mask))
        other=mask.copy();other[0,0,0]=False
        self.assertEqual(geometry.fixed_mask_validity(converted,other)["main_nonpositive_depth"],0)
        self.assertEqual(before,array_hash(converted["final"]["points"]))

    def test_nonfinite_fields_outside_mask_remain_visible(self):
        converted=deepcopy(self.converted)
        mask=np.zeros(geometry.DENSE_SHAPE,dtype=bool);mask[:,0,0]=True
        converted["native"]["points"][0,1,1,0]=np.nan
        converted["final"]["depth_m"][0,0,0]=np.nan
        result=geometry.fixed_mask_validity(converted,mask)
        self.assertEqual(result["native_nonfinite_fields"],1)
        self.assertEqual(result["main_nonfinite_fields"],1)
        self.assertEqual(result["mask_count"],5)
        self.assertEqual(result["detail"]["final"]["nonfinite_depth_on_mask"],1)
        self.assertFalse(result["candidate_main_depth_valid"])

    def test_reference_mask_shape_boolean_and_nonempty_contract(self):
        for mask in (np.ones((5,2,2),dtype=bool),np.zeros(geometry.DENSE_SHAPE,dtype=bool),
                     np.ones(geometry.DENSE_SHAPE,dtype=np.uint8)):
            with self.assertRaises(ValueError):
                geometry.fixed_mask_validity(self.converted,mask)


def run_fixtures():
    result=unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(M3PointFixtures).run(result)
    if not result.wasSuccessful():
        raise AssertionError("\n".join(str(item) for item in result.failures+result.errors))
    return dict(pass_all=True,synthetic_only=True,tests_passed=result.testsRun,real_prediction_arrays_loaded=False,
                real_reference_arrays_loaded=False,target_depth_files_opened=0,gpu_used=False)


if __name__=="__main__":
    print(json.dumps(run_fixtures(),indent=2))
