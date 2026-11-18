"""Synthetic-only boundary/axial checks; never load study arrays or scores."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

import profiled_m3_diagnostics as diagnostics
import prepare_profiled_m3_references as reference_preparation


ROOT = Path(__file__).resolve().parents[1]


def fixture(views=2,height=9,width=11):
    y,x = np.mgrid[:height,:width]
    points = np.stack((x*.01,y*.01,1.+x*.001),axis=-1)
    prediction = np.repeat(points[None],views,axis=0)
    mask = np.ones((views,height,width),dtype=bool)
    camera = np.repeat(np.eye(4)[None],views,axis=0)
    transform = dict(similarity_scale=1.,similarity_rotation=np.eye(3).tolist(),
                     similarity_translation=np.zeros((3,1)).tolist(),
                     icp_transformation=np.eye(4).tolist())
    return dict(pred_depth_m=prediction[...,2].copy(),prediction=prediction,
                reference=prediction.copy(),mask=mask,aligned_valid_points=prediction[mask].copy(),
                reference_label="registered_rgb",alignment=transform,reference_w2c=camera)


def kernel_transform(prediction,mask,alignment):
    # Independent homogeneous composition for synthetic fixtures, not a fit.
    similarity = np.eye(4)
    similarity[:3,:3] = alignment["similarity_scale"]*np.asarray(alignment["similarity_rotation"])
    similarity[:3,3] = np.asarray(alignment["similarity_translation"])[:,0]
    composed = np.asarray(alignment["icp_transformation"])@similarity
    homogeneous = np.column_stack((prediction[mask],np.ones(mask.sum())))
    return (homogeneous@composed.T)[:,:3]


class DiagnosticsTests(unittest.TestCase):
    def test_partition_bodies_match_pinned_source(self):
        source = ast.parse((ROOT/'scripts/depth_spatial_support.py').read_text(encoding='utf-8'))
        for name in ('near','partition'):
            original = next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name==name)
            actual = ast.parse(inspect.getsource(getattr(diagnostics,name))).body[0]
            self.assertEqual(ast.dump(original),ast.dump(actual))

    def test_partition_independently_replays_pinned_functions(self):
        source = ast.parse((ROOT/'scripts/depth_spatial_support.py').read_text(encoding='utf-8'))
        functions = [node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in ('near','partition')]
        namespace = dict(np=np,distance_transform_edt=distance_transform_edt)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'pinned_partition','exec'),namespace)
        case = fixture()
        case['reference'][:, :,6:,2] += .2
        case['mask'][:,4,4] = False
        case['reference'][~case['mask']] = np.nan
        for view in range(2):
            expected = namespace['partition'](case['reference'][view],case['mask'][view])
            actual = diagnostics.partition(case['reference'][view],case['mask'][view])
            for left,right in zip(expected,actual):
                np.testing.assert_array_equal(left,right)

    def test_identity_has_zero_error_and_no_image_border_edges(self):
        case = fixture()
        result = diagnostics.diagnose(**case)
        surface = result['paired_pixel_distance']['pooled_pixels']
        self.assertEqual(surface['all']['mean_absolute_m'],0)
        self.assertEqual(surface['interior']['n'],case['mask'].sum())
        self.assertEqual(surface['missing_neighbor']['status'],'empty')
        self.assertEqual(surface['jump_neighbor']['status'],'empty')
        self.assertIsNone(surface['jump_neighbor']['rmse_m'])
        self.assertEqual(result['metric_axial_depth']['pooled_pixels']['all']['mean_absolute_m'],0)
        json.dumps(result,allow_nan=False)

    def test_missing_priority_and_euclidean_radius(self):
        case = fixture(views=1,height=15,width=15)
        reference,valid = case['reference'][0],case['mask'][0]
        valid[7,7] = False
        reference[:,8:,2] += .2
        missing,jump,interior = diagnostics.partition(reference,valid)
        self.assertTrue(missing[7,8])
        self.assertFalse(jump[7,8])
        self.assertTrue(jump[1,8])
        self.assertTrue(missing[7,9])
        self.assertFalse(missing[9,9])  # sqrt(8)>2
        np.testing.assert_array_equal(missing.astype(int)+jump+interior,valid)

    def test_jump_requires_strictly_more_than_five_centimeters(self):
        points = np.zeros((7,7,3),dtype=float)
        points[:,4:,2] = .05
        mask = np.ones((7,7),dtype=bool)
        self.assertFalse(diagnostics.partition(points,mask)[1].any())
        points[:,4:,2] = .05001
        self.assertTrue(diagnostics.partition(points,mask)[1].any())

    def test_scene_transform_is_checked_without_refitting_or_extra_gauge(self):
        case = fixture()
        transform = case['alignment']
        transform['similarity_scale'] = 2.
        transform['similarity_rotation'] = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        transform['similarity_translation'] = [[.4],[-.3],[.5]]
        icp = np.eye(4)
        angle = .13
        icp[:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        icp[:3,3] = [.02,-.01,.03]
        transform['icp_transformation'] = icp.tolist()
        aligned = kernel_transform(case['prediction'],case['mask'],transform)
        case['aligned_valid_points'] = aligned
        case['reference'] = aligned.reshape(case['reference'].shape)
        result = diagnostics.diagnose(**case)
        self.assertLess(result['paired_pixel_distance']['pooled_pixels']['all']['rmse_m'],1e-14)
        # An evaluation alignment that makes the surface exact cannot repair
        # the original metric depth, which remains in its pre-alignment units.
        self.assertGreater(result['metric_axial_depth']['pooled_pixels']['all']['mean_absolute_m'],1.)
        self.assertFalse(result['metric_axial_depth']['evaluation_alignment_applied'])
        self.assertFalse(result['metric_axial_depth']['extra_metric_gauge_applied'])
        self.assertFalse(result['alignment_check']['transform_refitted'])

    def test_wrong_aligned_points_or_order_rejected(self):
        for kind in ('offset','order'):
            with self.subTest(kind=kind):
                case = fixture()
                if kind=='offset':
                    case['aligned_valid_points'] += 1e-5
                else:
                    case['aligned_valid_points'] = case['aligned_valid_points'][::-1].copy()
                with self.assertRaisesRegex(ValueError,'one scene transform'):
                    diagnostics.diagnose(**case)

    def test_changed_mask_same_count_cannot_reorder_kernel_points(self):
        case = fixture()
        case['mask'][0,0,0] = False
        case['aligned_valid_points'] = case['prediction'][case['mask']].copy()
        case['mask'][0,0,0] = True
        case['mask'][0,0,1] = False
        with self.assertRaisesRegex(ValueError,'fixed mask order'):
            diagnostics.diagnose(**case)

    def test_primary_axial_uses_supplied_reference_camera_z(self):
        case = fixture()
        case['reference_w2c'][:,2,3] = .25
        case['pred_depth_m'] = case['reference'][...,2]+.25
        result = diagnostics.diagnose(**case)
        self.assertEqual(result['metric_axial_depth']['pooled_pixels']['all']['mean_absolute_m'],0.)
        # Substituting the opposite camera convention changes the diagnostic.
        case['reference_w2c'][:,2,3] = -.25
        other = diagnostics.diagnose(**case)
        self.assertAlmostEqual(other['metric_axial_depth']['pooled_pixels']['all']['bias_m'],.5)

    def test_approximately_rigid_reference_camera_is_not_repaired(self):
        case = fixture()
        case['reference_w2c'][:,2,2] = 1.0001
        case['pred_depth_m'] = case['reference'][...,2]*1.0001
        before = case['reference_w2c'].copy()
        result = diagnostics.diagnose(**case)
        self.assertEqual(result['metric_axial_depth']['pooled_pixels']['all']['mean_absolute_m'],0.)
        self.assertGreater(result['metric_axial_depth']['camera']['max_rotation_orthogonality_error'],0.)
        self.assertFalse(result['metric_axial_depth']['camera']['matrices_modified'])
        np.testing.assert_array_equal(before,case['reference_w2c'])

    def test_camera_preparation_digest_matches_prepared_approximately_rigid_packet(self):
        case = fixture()
        approximate = case['reference_w2c'].copy()
        approximate[0,0,1] = .0002
        approximate[1,2,2] = 1.0001
        approximate[1,2,3] = .03
        prepared = reference_preparation.matrix_packet(approximate,2,'synthetic approximately rigid cameras')
        case['reference_w2c'] = prepared
        case['pred_depth_m'] = (np.einsum('nhwj,nj->nhw',case['reference'],prepared[:,2,:3])
                                +prepared[:,2,3,None,None])
        result = diagnostics.diagnose(**case)
        audit = result['metric_axial_depth']['camera']
        self.assertEqual(audit['camera_preparation_sha256'],reference_preparation.tensor_sha(prepared))
        self.assertEqual(audit['camera_tensor_sha256'],diagnostics.tensor_sha256(prepared))
        self.assertNotEqual(audit['camera_preparation_sha256'],audit['camera_tensor_sha256'])
        self.assertGreater(audit['max_rotation_orthogonality_error'],0.)
        self.assertEqual(result['metric_axial_depth']['pooled_pixels']['all']['mean_absolute_m'],0.)
        for field in ('camera_tensor_sha256','camera_preparation_sha256'):
            self.assertEqual(audit['camera_hash_algorithms'][field]['algorithm'],'SHA256')
            self.assertTrue(audit['camera_hash_algorithms'][field]['schema'].endswith('_v1'))
            self.assertIn('json.dumps',audit['camera_hash_algorithms'][field]['encoding'])
        np.testing.assert_array_equal(prepared,approximate)

    def test_released_axial_explicitly_unsupported(self):
        case = fixture()
        case.update(reference_label='released',reference_w2c=None)
        result = diagnostics.diagnose(**case)
        self.assertFalse(result['metric_axial_depth']['supported'])
        self.assertNotIn('pooled_pixels',result['metric_axial_depth'])
        case['reference_w2c'] = np.repeat(np.eye(4)[None],2,axis=0)
        with self.assertRaisesRegex(ValueError,'unsupported'):
            diagnostics.diagnose(**case)

    def test_invalid_reference_holes_are_allowed_but_valid_nonfinite_is_rejected(self):
        case = fixture()
        case['mask'][0,4,5] = False
        case['reference'][0,4,5] = np.nan
        case['aligned_valid_points'] = case['prediction'][case['mask']]
        result = diagnostics.diagnose(**case)
        self.assertEqual(result['mask_count'],int(case['mask'].sum()))
        case['reference'][0,4,6] = np.nan
        with self.assertRaisesRegex(ValueError,'fixed-mask reference'):
            diagnostics.diagnose(**case)

    def test_nonpositive_predictions_are_measured_not_filtered(self):
        case = fixture()
        case['pred_depth_m'][0,0,0] = -1.
        result = diagnostics.diagnose(**case)
        self.assertEqual(result['predicted_nonpositive_on_mask'],1)
        self.assertEqual(result['candidate_dependent_exclusions'],0)
        self.assertEqual(result['metric_axial_depth']['pooled_pixels']['all']['n'],int(case['mask'].sum()))
        self.assertGreater(result['metric_axial_depth']['pooled_pixels']['all']['mean_absolute_m'],0.)

    def test_pooled_pixels_and_equal_views_are_explicitly_different(self):
        case = fixture(height=3,width=3)
        case['mask'][0] = False
        case['mask'][0,1,1] = True
        case['prediction'][0,:, :,0] += 10.
        case['aligned_valid_points'] = case['prediction'][case['mask']]
        case['pred_depth_m'][0] += 10.
        result = diagnostics.diagnose(**case)
        for name in ('paired_pixel_distance','metric_axial_depth'):
            section = result[name]
            self.assertAlmostEqual(section['pooled_pixels']['all']['mean_absolute_m'],1.)
            self.assertAlmostEqual(section['equal_nonempty_views']['all']['mean_view_absolute_m'],5.)
            self.assertEqual(section['pooled_pixels']['all']['n'],10)
            self.assertAlmostEqual(section['pooled_pixels']['all']['rmse_m'],np.sqrt(10.))

    def test_empty_view_and_bins_remain_null_with_counts(self):
        case = fixture()
        case['mask'][0] = False
        case['aligned_valid_points'] = case['prediction'][case['mask']]
        result = diagnostics.diagnose(**case)
        section = result['paired_pixel_distance']
        self.assertEqual(section['by_view'][0]['all']['status'],'empty')
        self.assertIsNone(section['by_view'][0]['all']['mean_absolute_m'])
        self.assertEqual(section['equal_nonempty_views']['all']['empty_views'],1)
        self.assertEqual(section['equal_nonempty_views']['all']['nonempty_views'],1)

    def test_regions_depend_on_reference_not_prediction(self):
        case = fixture()
        case['mask'][0,4,5] = False
        case['reference'][:, :,6:,2] += .2
        case['aligned_valid_points'] = case['prediction'][case['mask']]
        original = diagnostics.diagnose(**case)
        case['prediction'][...,0] += .03
        case['aligned_valid_points'] = case['prediction'][case['mask']]
        changed = diagnostics.diagnose(**case)
        self.assertEqual(original['partition']['region_mask_sha256'],changed['partition']['region_mask_sha256'])
        self.assertEqual(original['mask_tensor_sha256'],changed['mask_tensor_sha256'])

    def test_input_arrays_unchanged_and_no_external_verification_claim(self):
        case = fixture()
        before = deepcopy(case)
        result = diagnostics.diagnose(**case)
        for name,value in case.items():
            if isinstance(value,np.ndarray):
                np.testing.assert_array_equal(value,before[name])
            else:
                self.assertEqual(value,before[name])
        self.assertFalse(result['external_provenance_verified'])
        self.assertFalse(result['files_opened'])
        self.assertFalse(result['release_authorized'])
        self.assertFalse(result['metric_axial_depth']['camera']['provenance_verified'])

    def test_mask_dtype_shape_empty_and_prediction_nonfinite_rejected(self):
        mutations = [lambda c:c.update(mask=c['mask'].astype(int)),
                     lambda c:c.update(mask=c['mask'][0]),
                     lambda c:c['mask'].fill(False),
                     lambda c:c['prediction'].__setitem__((0,0,0,0),np.nan),
                     lambda c:c['pred_depth_m'].__setitem__((0,0,0),np.inf)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                case = fixture()
                mutate(case)
                with self.assertRaises(ValueError):
                    diagnostics.diagnose(**case)

    def test_missing_or_invalid_reference_cameras_rejected(self):
        mutations = [lambda c:c.update(reference_w2c=None),
                     lambda c:c['reference_w2c'].__setitem__((0,3,0),1.),
                     lambda c:c['reference_w2c'].__setitem__((0,2,2),0.),
                     lambda c:c['reference_w2c'].__setitem__((0,0,0),-1.),
                     lambda c:c['reference_w2c'].__setitem__((0,0,3),np.nan)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                case = fixture()
                mutate(case)
                with self.assertRaises(ValueError):
                    diagnostics.diagnose(**case)

    def test_invalid_or_incomplete_transform_rejected(self):
        mutations = [lambda a:a.pop('similarity_scale'),
                     lambda a:a.update(similarity_scale=0.),
                     lambda a:a.update(similarity_scale=float('inf')),
                     lambda a:a.update(similarity_translation=[0.,0.,0.]),
                     lambda a:a.update(similarity_rotation=(2*np.eye(3)).tolist()),
                     lambda a:a['icp_transformation'][3].__setitem__(0,1.)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                case = fixture()
                mutate(case['alignment'])
                with self.assertRaises(ValueError):
                    diagnostics.diagnose(**case)

    def test_mask_digest_includes_shape_dtype_and_order(self):
        mask = fixture()['mask']
        self.assertNotEqual(diagnostics.tensor_sha256(mask),diagnostics.tensor_sha256(mask.reshape(1,-1)))
        self.assertNotEqual(diagnostics.tensor_sha256(mask),diagnostics.tensor_sha256(mask.astype(np.uint8)))


if __name__=='__main__':
    unittest.main()
