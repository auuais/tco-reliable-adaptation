"""Exercise the complete TUM prediction seal with small synthetic positive depth heads."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import run_tum_sparse_depth as study
from evaluation_stability import sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import write_arrays


class TumPredictionSealTest(unittest.TestCase):
    def test_normal_entry_point_seals_all_methods(self):
        with TemporaryDirectory() as folder:
            public=Path(folder)/'public'
            raw=Path(folder)/'raw'
            (public/'captures').mkdir(parents=True)
            raw.mkdir()
            sequence='synthetic'
            depth=np.ones((5,4,4))
            k=np.repeat(np.eye(3)[None],5,axis=0)
            poses=np.repeat(np.eye(4)[None],5,axis=0)
            native=point_map(depth,k,poses[:,:3],np.ones(5))
            capture=raw/(sequence+'.npz')
            write_arrays(capture,dict(depth=depth[...,None],intrinsics=k,extrinsics=poses[:,:3],native=native))
            (public/'captures'/(sequence+'.json')).write_text(json.dumps(dict(archive_sha256=sha(capture),rgb_tensor_sha256='synthetic-rgb')))
            xy=np.array([[0,0],[1,0],[2,0],[3,0],[0,2],[1,2],[2,2],[3,2]])
            values=np.column_stack([xy,np.ones(8)])
            sensors={sequence+'_intrinsics':k,sequence+'_poses':poses}
            for grid in study.GRIDS:
                sensors.update({f'{sequence}_{grid}_{i}':values for i in range(5)})
            packet=public/'measurements.npz'
            write_arrays(packet,sensors)
            (public/'inputs.json').write_text(json.dumps(dict(packet_sha256=sha(packet),sequences={sequence:dict(rgb_tensor_sha256='synthetic-rgb')})))
            with patch.object(study,'PUBLIC',public),patch.object(study,'RAW',raw),patch.object(study,'SEQUENCES',(sequence,)):
                study.predict()
                first=(public/'predictions.json').read_bytes()
                study.predict()
                self.assertEqual(first,(public/'predictions.json').read_bytes())
            sealed=json.loads(first)
            self.assertEqual(len(sealed['batches'][sequence]['hashes']),16)
            self.assertEqual(sealed['calibration_solver_sha256'],sha(study.ROOT/'scripts/scale_shift_sparse_depth.py'))
