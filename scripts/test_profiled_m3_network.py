"""Synthetic CPU fixtures for the new M3 case/endpoint input adapter.

No study arrays, held scores, model initialization or GPU execution. Endpoint
plumbing is exercised with synthetic two-by-two tensors and mocked native KDP;
its arithmetic body is separately checked against the frozen M2 source AST.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import profiled_m3_network as network


SOURCE_ROOT=Path(__file__).resolve().parents[1]


def payload(root,side="earlier",grid="centered",pose="clean"):
    identities={}
    for name in ("synthetic/selected_packet.npz","synthetic/sealed_folds.npz"):
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        # Deliberately not decodable arrays: the adapter only verifies bytes.
        path.write_bytes((name+" opaque input identity\n").encode())
        identities[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    xy=np.stack((20+16*np.arange(16),24+16*(np.arange(16)%4)),axis=1).astype(np.int64)
    original_uv=(xy+.5)*np.array([518/512,392/384])-.5
    depths=1+np.arange(16)/16
    samples,original,folds=[],[],[]
    for view,frame in enumerate(network.FRAMES):
        values=np.column_stack((original_uv,depths+view/8,
            np.full(16,frame+(-4 if side=="earlier" else 4)),
            20+np.arange(16)*32,15+60*(np.arange(16)%4),1000+np.arange(16)*16))
        original.append(values)
        samples.append(dict(xy=xy.copy(),depth=values[:,2].copy()))
        folds.append(np.array([0]*8+[1]*8,dtype=np.int64))
    c2w=np.repeat(np.eye(4)[None],5,axis=0)
    c2w[:,0,3]=np.arange(5)*.1
    if pose=="noisy":
        # Same arbitrary, finite, invertible noisy packet is sent to all paths.
        c2w[1:,1,3]=.02
    return dict(scene="heads",side=side,grid=grid,pose_condition=pose,
        packet_sha256=identities["synthetic/selected_packet.npz"],
        splits_sha256=identities["synthetic/sealed_folds.npz"],input_sha256=identities,
        input_semantic_sha256={"pose_matrices":hashlib.sha256(c2w.tobytes()).hexdigest()},
        K=np.array([[470.4,0,255.9],[0,470.4,191.9],[0,0,1.]]),
        w2c=np.linalg.inv(c2w),c2w=c2w,samples=samples,folds=folds,original_samples=original)


class M3NetworkFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="profiled-m3-network-")
        self.root=Path(self.temp.name).resolve()
        self.root_patch=patch.object(network,"ROOT",self.root)
        self.root_patch.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.root_patch.stop)

    def prepare(self,values=None):
        return network.from_prepared_case(payload(self.root) if values is None else values,device="cpu")

    def test_both_signed_sources_and_grids_preserve_all_original_rows(self):
        for side in ("earlier","later"):
            for grid in ("centered","shifted"):
                with self.subTest(side=side,grid=grid):
                    values=payload(self.root,side,grid)
                    with patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("GPU forbidden")):
                        case=self.prepare(values)
                        native=network.native_prior_samples(case)
                    self.assertEqual(case["samples"][0]["xy"].device.type,"cpu")
                    self.assertEqual(case["samples"][0]["xy"].dtype,torch.int64)
                    self.assertEqual(case["samples"][0]["depth"].dtype,torch.float64)
                    for view in range(5):
                        np.testing.assert_array_equal(native[view],values["original_samples"][view])
                        self.assertEqual(native[view].shape,(16,7))
                        self.assertEqual(int((case["folds"][view]==0).sum()),8)
                        self.assertEqual(int((case["folds"][view]==1).sum()),8)

    def test_bad_file_hash_missing_file_and_path_escape_are_rejected(self):
        values=payload(self.root)
        values["input_sha256"]["synthetic/selected_packet.npz"]="0"*64
        with self.assertRaisesRegex(ValueError,"Prepared input changed"):
            self.prepare(values)
        values=payload(self.root)
        values["input_sha256"]={"synthetic/missing.npz":"0"*64}
        with self.assertRaises((ValueError,FileNotFoundError)):
            self.prepare(values)
        values["input_sha256"]={"../outside":"0"*64}
        with self.assertRaises(ValueError):
            self.prepare(values)

    def test_file_mutation_after_preparation_is_caught_before_callback(self):
        case=self.prepare()
        (self.root/"synthetic/selected_packet.npz").write_bytes(b"changed after preparation")
        with self.assertRaisesRegex(ValueError,"Prepared input changed"):
            network.native_prior_samples(case)

    def test_tensor_camera_fold_and_raw_row_mutations_are_detected(self):
        mutations=(lambda c:c["samples"][0]["depth"].__setitem__(0,2.),
                   lambda c:c["samples"][0]["xy"].__setitem__((0,0),21),
                   lambda c:c["folds"][0].__setitem__(0,1),
                   lambda c:c["K"].__setitem__((0,0),500),
                   lambda c:c["w2c"].__setitem__((0,0,3),.1),
                   lambda c:c["c2w"].__setitem__((0,0,3),.1),
                   lambda c:c["original_samples"][0].__setitem__((0,6),1001),
                   lambda c:c.__setitem__("pose_condition","noisy"))
        for mutation in mutations:
            case=self.prepare()
            mutation(case)
            with self.assertRaisesRegex(ValueError,"Fitting tensors changed"):
                network.native_prior_samples(case)

    def test_payload_and_callback_arrays_do_not_alias_case_storage(self):
        values=payload(self.root)
        case=self.prepare(values)
        expected=deepcopy(values)
        values["K"][0,0]=999
        values["samples"][0]["depth"][0]=999
        values["folds"][0][0]=1
        values["original_samples"][0][0,6]=999
        native=network.native_prior_samples(case)
        np.testing.assert_array_equal(case["K"].numpy(),expected["K"])
        np.testing.assert_array_equal(native[0],expected["original_samples"][0])
        native[0][0,6]=999
        np.testing.assert_array_equal(network.native_prior_samples(case)[0],expected["original_samples"][0])

    def test_target_or_opposite_temporal_source_rows_are_rejected(self):
        for side in ("earlier","later"):
            for forbidden in (25,29 if side=="earlier" else 21):
                values=payload(self.root,side)
                values["original_samples"][0][0,3]=forbidden
                with self.assertRaisesRegex(ValueError,"wrong temporal direction"):
                    self.prepare(values)

    def test_missing_view_and_fold_count_or_dtype_cannot_be_coerced(self):
        for field in ("samples","folds","original_samples"):
            values=payload(self.root)
            values[field].pop()
            with self.assertRaisesRegex(ValueError,"five registered views"):
                self.prepare(values)
        for mutation in (lambda p:p["folds"][0].__setitem__(0,1),
                         lambda p:p["folds"][0].__setitem__(0,2),
                         lambda p:p["folds"].__setitem__(0,p["folds"][0].astype(float)),
                         lambda p:p["samples"][0].__setitem__("xy",p["samples"][0]["xy"].astype(float)),
                         lambda p:p["folds"].__setitem__(0,p["folds"][0][:-1])):
            values=payload(self.root)
            mutation(values)
            with self.assertRaises(ValueError):
                self.prepare(values)

    def test_finite_correct_shape_and_inverse_camera_contract(self):
        for field in ("K","w2c","c2w"):
            values=payload(self.root)
            values[field].flat[0]=float("nan")
            with self.assertRaisesRegex(ValueError,"Invalid "+field):
                self.prepare(values)
            values=payload(self.root)
            values[field]=values[field][:-1]
            with self.assertRaisesRegex(ValueError,"Invalid "+field):
                self.prepare(values)
        values=payload(self.root)
        values["w2c"][0,0,3]=1
        with self.assertRaisesRegex(ValueError,"transforms disagree"):
            self.prepare(values)

    def test_dataset_scale_SO3_deviation_is_preserved_not_orthogonalized(self):
        values=payload(self.root)
        values["c2w"][1,0,0]+=6.7e-4
        values["w2c"]=np.linalg.inv(values["c2w"])
        case=self.prepare(values)
        np.testing.assert_array_equal(case["c2w"].numpy(),values["c2w"])
        np.testing.assert_array_equal(case["w2c"].numpy(),values["w2c"])

    def test_invalid_focal_singular_and_non_normalized_intrinsics_rejected(self):
        for name,mutation in (
                ("zero_K",lambda k:k.fill(0)),
                ("negative_fx",lambda k:k.__setitem__((0,0),-470.4)),
                ("negative_fy",lambda k:k.__setitem__((1,1),-470.4)),
                ("scaled_bottom",lambda k:k.__setitem__((2,2),2)),
                ("projective_bottom",lambda k:k.__setitem__((2,0),1e-5)),
                ("singular_positive_focals",lambda k:k.__setitem__((slice(0,2),slice(0,2)),[[1.,1.],[1.,1.]]))):
            with self.subTest(invalid=name):
                values=payload(self.root)
                mutation(values["K"])
                with self.assertRaisesRegex(ValueError,"normalized camera intrinsic"):
                    self.prepare(values)

    def test_reflection_rejected_even_when_supplied_cameras_are_exact_inverses(self):
        values=payload(self.root)
        values["c2w"][1,0,0]=-1
        values["w2c"]=np.linalg.inv(values["c2w"])
        np.testing.assert_allclose(values["w2c"]@values["c2w"],np.repeat(np.eye(4)[None],5,axis=0),rtol=0,atol=1e-12)
        with self.assertRaisesRegex(ValueError,"right handed"):
            self.prepare(values)

    def test_non_affine_camera_rows_rejected_despite_exact_inverse_pair(self):
        for column,value in ((0,.01),(3,2.)):
            with self.subTest(column=column):
                values=payload(self.root)
                values["c2w"][1,3,column]=value
                values["w2c"]=np.linalg.inv(values["c2w"])
                np.testing.assert_allclose(values["w2c"]@values["c2w"],np.repeat(np.eye(4)[None],5,axis=0),rtol=0,atol=1e-12)
                with self.assertRaisesRegex(ValueError,"affine"):
                    self.prepare(values)

    def test_native_pixels_and_depth_values_require_exact_equivalence(self):
        for mutation in (lambda p:p["samples"][0]["xy"].__setitem__((0,0),21),
                         lambda p:p["samples"][0]["depth"].__setitem__(0,np.nextafter(1.,2.))):
            values=payload(self.root)
            mutation(values)
            with self.assertRaises(AssertionError):
                self.prepare(values)
        # Ordering is immaterial to native equivalence, but original rows stay
        # in their selected source order, including all source metadata columns.
        values=payload(self.root)
        order=np.arange(16)[::-1]
        values["samples"][0]={key:value[order].copy() for key,value in values["samples"][0].items()}
        values["folds"][0]=values["folds"][0][order].copy()
        case=self.prepare(values)
        np.testing.assert_array_equal(network.native_prior_samples(case)[0],values["original_samples"][0])

    def test_duplicate_outside_nonfinite_and_nonpositive_observations_rejected(self):
        mutations=(lambda p:p["original_samples"][0].__setitem__((0,slice(0,2)),p["original_samples"][0][1,:2]),
                   lambda p:p["original_samples"][0].__setitem__((0,0),-100),
                   lambda p:p["original_samples"][0].__setitem__((0,6),float("inf")),
                   lambda p:p["samples"][0]["depth"].__setitem__(0,0),
                   lambda p:p["samples"][0]["depth"].__setitem__(0,float("nan")))
        for mutation in mutations:
            values=payload(self.root)
            mutation(values)
            with self.assertRaises(ValueError):
                self.prepare(values)

    def test_only_declared_bytes_read_and_no_array_or_held_source_access(self):
        values=payload(self.root)
        allowed={str((self.root/name).resolve()) for name in values["input_sha256"]}
        observed=[]
        original=Path.read_bytes
        def guarded(path):
            self.assertIn(str(path.resolve()),allowed)
            observed.append(str(path.resolve()))
            return original(path)
        with patch.object(Path,"read_bytes",guarded),patch.object(np,"load",side_effect=AssertionError("No arrays should be decoded")):
            case=self.prepare(values)
            network.native_prior_samples(case)
        self.assertEqual(set(observed),allowed)
        self.assertEqual(len(observed),4)

    def test_lazy_case_loader_forwards_exact_registered_condition(self):
        values=payload(self.root,"later","shifted","noisy")
        helper=SimpleNamespace(load_prepared_case=lambda *args: values)
        with patch.dict(sys.modules,{"prepare_profiled_m3_inputs":helper}),patch.object(helper,"load_prepared_case",return_value=values) as loader:
            case=network.load_case("sealed-plan.json","attempt01","heads","later","shifted","noisy",device="cpu")
        loader.assert_called_once_with("sealed-plan.json","attempt01","heads","later","shifted","noisy")
        self.assertEqual((case["side"],case["grid"],case["pose_condition"]),("later","shifted","noisy"))
        np.testing.assert_array_equal(case["w2c"].numpy(),values["w2c"])

    def test_capture_endpoint_AST_is_exact_frozen_M2_body(self):
        def function(path):
            tree=ast.parse(path.read_text(encoding="utf-8"))
            return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="capture_endpoint")
        old=function(SOURCE_ROOT/"scripts/check_profiled_endpoints.py")
        new=function(SOURCE_ROOT/"scripts/profiled_m3_network.py")
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_mock_endpoint_uses_selected_noisy_priors_then_optimized_native_cameras(self):
        values=payload(self.root,"later","shifted","noisy")
        case=self.prepare(values)
        seen={}
        alignment=ModuleType("mast3r.cloud_opt.sparse_ga")
        cloud=ModuleType("mast3r.cloud_opt")
        parent=ModuleType("mast3r")
        parent.cloud_opt,cloud.sparse_ga=cloud,alignment
        callbacks=[]
        def prior(module,arguments,original,poses,intrinsic,weight):
            seen.update(original=original,poses=poses.copy(),K=intrinsic.copy(),weight=weight)
            return SimpleNamespace(counts=[16]*5,trace=[dict(finite_gradient=True,received_gradient=1.) for _ in range(300)])
        @contextmanager
        def hook(module,factory):
            callback=factory({})
            callbacks.append(callback)
            yield [callback]
        native_cameras=values["c2w"].copy()
        native_cameras[:,2,3]+=5
        def capture(model,scene,cache,raw,public,kind):
            self.assertEqual((scene,kind),("heads_base","native"))
            for path in (cache,raw,public):
                path.mkdir(parents=True)
            np.savez(raw/"native.npz",depth=np.ones((5,2,2)),intrinsics=np.repeat(values["K"][None],5,axis=0),c2w=native_cameras)
            return dict(prior_cache_files=0,coarse_iterations=300,fine_iterations=300)
        def refine(depth,intrinsics,cameras,samples,matches,confidence):
            np.testing.assert_array_equal(cameras.numpy(),native_cameras)
            self.assertFalse(np.array_equal(cameras.numpy(),values["c2w"]))
            self.assertTrue(all(len(item["xy"])==16 for item in samples))
            seen["final_receives_native_cameras"]=True
            return depth*2,cameras.clone(),torch.ones(5),dict(g0=2.)
        def write_arrays(path,arrays):
            np.savez(path,**arrays)
        modules={"mast3r":parent,"mast3r.cloud_opt":cloud,"mast3r.cloud_opt.sparse_ga":alignment,
            "mast3r_alignment_hook":SimpleNamespace(optimizer_hook=hook),
            "mast3r_calibrated_alignment":SimpleNamespace(CalibratedPriors=prior),
            "mast3r_network_capture":SimpleNamespace(capture=capture,views_for=lambda *_:(None,list("abcde"),None)),
            "prepare_dense_focal_pilot":SimpleNamespace(write_arrays=write_arrays)}
        with patch.dict(sys.modules,modules),patch.object(network,"load_fresh_graph",return_value=([],[],dict(synthetic=True))),patch.object(network,"refine_native",side_effect=refine):
            result=network.capture_endpoint(object(),case,self.root/"cache",self.root/"raw",self.root/"public")
        np.testing.assert_array_equal(seen["poses"],values["w2c"])
        np.testing.assert_array_equal(seen["K"],values["K"])
        for view in range(5):
            np.testing.assert_array_equal(seen["original"][view],values["original_samples"][view])
        self.assertEqual(seen["weight"],.01)
        self.assertTrue(seen["final_receives_native_cameras"])
        self.assertEqual(result[-1]["sample_counts"],[16]*5)
        self.assertTrue(result[-1]["original_prior_rows_preserved"])
        self.assertEqual(len(callbacks),1)


def run_fixtures():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(M3NetworkFixtures)
    result=unittest.TestResult()
    suite.run(result)
    if not result.wasSuccessful():
        raise AssertionError("\n".join(str(value) for value in result.errors+result.failures))
    return dict(pass_all=True,synthetic_only=True,tests_passed=result.testsRun,
        real_prediction_arrays_opened=False,held_scores_read=False,gpu_used=False,
        scientific_M3_execution_authorized=False)


if __name__=="__main__":
    print(json.dumps(run_fixtures(),indent=2))
