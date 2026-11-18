"""IO integrity tests; one full-raster synthetic conversion, no study labels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_profiled_m3_conversions as io
from profiled_m3_points import DENSE_SHAPE, NATIVE_SHAPE, POINT_FIELDS


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class Fixture:
    def __init__(self, root, full=False):
        self.root = root
        for name in io.SOURCES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((SOURCE_ROOT / name).read_bytes())
        self.config = io.configuration(root)
        self.job = "heads_earlier_centered_clean_r0_P0"
        self.attempt = "synthetic01"
        native = dict(dummy=np.ones(2))
        final = dict(dummy=np.zeros(2))
        if full:
            k = np.tile(np.array([[450., 0., 255.5], [0., 450., 191.5], [0., 0., 1.]], dtype=np.float32), (5,1,1))
            camera = np.tile(np.eye(4, dtype=np.float32), (5,1,1))
            camera[:,0,3] = np.arange(5, dtype=np.float32) * .1
            depth = np.full(NATIVE_SHAPE, 2., dtype=np.float32)
            native = dict(depth=depth, intrinsics=k, c2w=camera)
            metric_camera = camera.astype(np.float64)
            metric_camera[:,:3,3] *= 1.25
            final = dict(depth=depth.astype(np.float64)*1.25, intrinsics=k.copy(),
                         c2w=metric_camera, g0=np.asarray(1.25), scales=np.ones(5))
        self.inputs = {}
        for name, arrays in (("native",native),("final",final)):
            path = root / (name+".npz")
            with path.open("xb") as f: np.savez_compressed(f, **arrays)
            self.inputs[name] = dict(file=path.name,sha256=io.sha(path))

    @property
    def target(self):
        return self.root / io.SCOPE / self.job / self.attempt

    def prepare(self):
        return io.prepare(self.root,self.job,self.attempt,self.inputs['native'],self.inputs['final'],self.config)

    def verify(self, replay=False):
        return io.verify(self.root,self.job,self.attempt,self.inputs['native'],self.inputs['final'],self.config,replay=replay)


def mock_conversion(*_):
    return dict(native={key:np.ones(2) for key in POINT_FIELDS},
                final={key:np.ones(2)*2 for key in POINT_FIELDS},
                validation=dict(synthetic_mock=True))


class ConversionIOTests(unittest.TestCase):
    def test_production_raster_roundtrip_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=Fixture(Path(temp),full=True)
            record=fixture.prepare()
            receipt=fixture.verify(replay=True)
            self.assertTrue(receipt['conversion_replayed'])
            self.assertFalse(receipt['score_access_authorized'])
            self.assertFalse(record['references_loaded'])
            self.assertFalse(record['model_completion_verified'])
            with np.load(fixture.target/'final.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['points'].shape,DENSE_SHAPE+(3,))
                self.assertEqual(saved['depth_m'].dtype,np.float64)
                np.testing.assert_array_equal(saved['depth_m'],np.full(DENSE_SHAPE,2.5))

    def test_wrong_endpoint_hash_fails_before_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=Fixture(Path(temp))
            fixture.inputs['native']['sha256']='0'*64
            with patch.object(io,'_load',side_effect=AssertionError('must not decode')):
                with self.assertRaisesRegex(ValueError,'Endpoint bytes differ'): fixture.prepare()
            self.assertFalse(fixture.target.exists())

    def test_changed_source_or_dependency_fails_before_decode(self):
        for fault in ('source','dependency'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                fixture=Fixture(Path(temp))
                if fault=='source':
                    (fixture.root/io.SOURCES[1]).write_bytes(b'changed')
                else:
                    fixture.config['dependencies']['numpy']='invented'
                with patch.object(io,'_load',side_effect=AssertionError('must not decode')):
                    with self.assertRaisesRegex(ValueError,'source/runtime'): fixture.prepare()

    def test_path_traversal_and_duplicate_endpoints_fail_before_write(self):
        for fault in ('job','attempt','endpoint','same'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                fixture=Fixture(Path(temp))
                if fault=='job': fixture.job='../escape'
                elif fault=='attempt': fixture.attempt='../escape'
                elif fault=='endpoint': fixture.inputs['native']['file']='../escape.npz'
                else: fixture.inputs['final']=fixture.inputs['native'].copy()
                with self.assertRaises(ValueError): fixture.prepare()

    def test_completed_attempt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=Fixture(Path(temp))
            with patch.object(io,'convert_endpoints',side_effect=mock_conversion): fixture.prepare()
            before=io.sha(fixture.target/'complete.json')
            with self.assertRaisesRegex(ValueError,'Preserve existing'): fixture.prepare()
            self.assertEqual(before,io.sha(fixture.target/'complete.json'))

    def test_conversion_failure_and_postread_input_change_are_preserved(self):
        for fault in ('conversion','mutation'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                fixture=Fixture(Path(temp))
                def conversion(*args):
                    if fault=='conversion': raise ValueError('bad synthetic geometry')
                    (fixture.root/'native.npz').write_bytes(b'changed')
                    return mock_conversion()
                with patch.object(io,'convert_endpoints',side_effect=conversion):
                    with self.assertRaises(ValueError): fixture.prepare()
                self.assertTrue((fixture.target/'failure.json').exists())
                self.assertFalse((fixture.target/'complete.json').exists())
                with self.assertRaisesRegex(ValueError,'Preserve existing'): fixture.prepare()

    def test_changed_output_or_foreign_descriptor_is_rejected(self):
        for fault in ('bytes','path','extra','start','failure'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temp:
                fixture=Fixture(Path(temp))
                with patch.object(io,'convert_endpoints',side_effect=mock_conversion): fixture.prepare()
                if fault=='bytes': (fixture.target/'native.npz').write_bytes(b'changed')
                elif fault=='path':
                    path=fixture.target/'complete.json'; record=io.read(path)
                    record['outputs']['native']=fixture.inputs['native'].copy()
                    path.write_text(json.dumps(record),encoding='utf-8')
                elif fault=='start':
                    path=fixture.target/'started.json'; record=io.read(path)
                    record['job_id']='different'; path.write_text(json.dumps(record),encoding='utf-8')
                elif fault=='extra': (fixture.target/'unexpected').write_bytes(b'extra')
                else: (fixture.target/'failure.json').write_text('{}',encoding='utf-8')
                with self.assertRaises(ValueError): fixture.verify()

    def test_replay_detects_valid_archive_with_changed_values(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=Fixture(Path(temp))
            with patch.object(io,'convert_endpoints',side_effect=mock_conversion): fixture.prepare()
            path=fixture.target/'native.npz'
            with path.open('wb') as f: np.savez_compressed(f,**{key:np.zeros(2) for key in POINT_FIELDS})
            record=io.read(fixture.target/'complete.json')
            record['outputs']['native']['sha256']=io.sha(path)
            (fixture.target/'complete.json').write_text(json.dumps(record),encoding='utf-8')
            with patch.object(io,'convert_endpoints',side_effect=mock_conversion):
                with self.assertRaisesRegex(ValueError,'Recomputed converted field'):
                    io.verify(fixture.root,fixture.job,fixture.attempt,fixture.inputs['native'],
                              fixture.inputs['final'],fixture.config)


if __name__=='__main__':
    unittest.main()
