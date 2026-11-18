"""Generated reference geometry only; never decode a study reference archive."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

import profiled_m3_reference_arrays as refs
from prepare_profiled_m3_references import tensor_sha


def fixture():
    shape=refs.SHAPE
    k=np.array([[450.,0.,255.5],[0.,450.,191.5],[0.,0.,1.]],dtype=np.float64)
    camera=np.tile(np.eye(4),(5,1,1));camera[:,0,0]=1.0001
    camera[:,0,3]=np.arange(5)*.1
    y,x=np.indices(shape[1:])
    u=(x+.5)*512/518-.5;v=(y+.5)*384/392-.5
    z=2.+x/10000.+y/10000.
    local=np.stack(((u-k[0,2])/k[0,0]*z,(v-k[1,2])/k[1,1]*z,z),axis=-1)
    points=np.stack([((local-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T).astype(np.float32) for pose in camera])
    mask=np.zeros(shape,dtype=bool);mask[:,::8,::8]=True
    arrays=dict(registered_rgb=points,released=points.copy()+np.float32(.2),valid_mask=mask)
    spec=dict(reference_fields=list(refs.FIELDS),fixed_mask=dict(shape=list(shape),dtype='bool',
        declared_count=int(mask.sum()),shared_between_references=True,tensor_sha256=None),
        primary_axial_camera=dict(reference='registered_rgb',dtype='<f8',shape=[5,4,4],
            world_to_camera=camera.tolist(),tensor_sha256=tensor_sha(camera),matrices_orthogonalized=False))
    return arrays,spec,k


class ReferenceArrayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays,cls.spec,cls.k=fixture()

    def test_projection_helper_is_exact_original_ast(self):
        root=Path(__file__).resolve().parents[1]
        def function(path):
            return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                        if isinstance(node,ast.FunctionDef) and node.name=='project')
        self.assertEqual(ast.dump(function(root/'scripts/profiled_m3_reference_arrays.py')),
                         ast.dump(function(root/'scripts/audit_mast3r_crossover_matches.py')))

    def test_fixed_mask_projection_and_nonorthogonal_camera_are_preserved(self):
        before=deepcopy(self.spec)
        report=refs.validate(self.arrays,self.spec,self.k)
        self.assertTrue(report['pass_all'])
        self.assertGreater(report['primary_camera']['max_rotation_orthogonality_error'],1e-5)
        self.assertLess(max(report['max_native_pixel_error_per_view']),.01)
        self.assertEqual(report['parameters_fitted'],0)
        self.assertFalse(report['physical_calibration_accuracy_established'])
        self.assertFalse(report['score_access_authorized'])
        self.assertEqual(self.spec,before)

    def test_undefined_reference_holes_are_not_new_exclusions(self):
        arrays={key:value.copy() for key,value in self.arrays.items()}
        for name in refs.FIELDS[:2]: arrays[name][~arrays['valid_mask']]=np.nan
        report=refs.validate(arrays,self.spec,self.k)
        self.assertEqual(report['mask_count'],self.spec['fixed_mask']['declared_count'])
        self.assertEqual(report['candidate_dependent_exclusions'],0)

    def test_wrong_shape_dtype_or_extra_field_rejected(self):
        for fault in ('point_dtype','mask_dtype','shape','extra'):
            with self.subTest(fault=fault):
                arrays=self.arrays.copy()
                if fault=='point_dtype': arrays['released']=arrays['released'].astype(np.float64)
                elif fault=='mask_dtype': arrays['valid_mask']=arrays['valid_mask'].astype(np.uint8)
                elif fault=='shape': arrays['registered_rgb']=arrays['registered_rgb'][:4]
                else: arrays['candidate_mask']=arrays['valid_mask']
                with self.assertRaises(ValueError):refs.validate(arrays,self.spec,self.k)

    def test_empty_view_wrong_count_and_wrong_mask_hash_rejected(self):
        for fault in ('empty','count','hash'):
            with self.subTest(fault=fault):
                arrays=self.arrays.copy();spec=deepcopy(self.spec)
                if fault=='empty':
                    arrays['valid_mask']=arrays['valid_mask'].copy();arrays['valid_mask'][0]=False
                    spec['fixed_mask']['declared_count']=int(arrays['valid_mask'].sum())
                elif fault=='count':spec['fixed_mask']['declared_count']+=1
                else:spec['fixed_mask']['tensor_sha256']='0'*64
                with self.assertRaises(ValueError):refs.validate(arrays,spec,self.k)

    def test_nonfinite_valid_reference_rejected_for_both_labels(self):
        for label in refs.FIELDS[:2]:
            with self.subTest(label=label):
                arrays=self.arrays.copy();arrays[label]=arrays[label].copy();arrays[label][0,0,0,0]=np.nan
                with self.assertRaisesRegex(ValueError,'Nonfinite reference'):refs.validate(arrays,self.spec,self.k)

    def test_wrong_camera_hash_and_camera_repair_are_rejected(self):
        for fault in ('hash','repair'):
            with self.subTest(fault=fault):
                spec=deepcopy(self.spec)
                if fault=='hash':spec['primary_axial_camera']['world_to_camera'][0][0][3]+=.1
                else:spec['primary_axial_camera']['matrices_orthogonalized']=True
                with self.assertRaises(ValueError):refs.validate(self.arrays,spec,self.k)

    def test_native_projection_mismatch_cannot_be_fitted_away(self):
        k=self.k.copy();k[0,2]+=.1
        with self.assertRaisesRegex(ValueError,'projection convention'):refs.validate(self.arrays,self.spec,k)

    def test_nonpositive_primary_depth_rejected(self):
        arrays=self.arrays.copy();arrays['registered_rgb']=arrays['registered_rgb'].copy()
        arrays['registered_rgb'][0,0,0,2]=-1
        with self.assertRaisesRegex(ValueError,'Nonpositive valid primary'):refs.validate(arrays,self.spec,self.k)


if __name__=='__main__':
    unittest.main()
