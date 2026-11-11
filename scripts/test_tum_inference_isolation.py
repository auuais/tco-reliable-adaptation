"""Ensure cross-dataset inference can run with unreadable target-depth labels."""
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import prepare_tum_sparse_packet as packet
from evaluation_stability import sha


def png(value):
    buffer=io.BytesIO()
    Image.fromarray(value).save(buffer,format='PNG')
    return buffer.getvalue()


class TumInferenceIsolationTest(unittest.TestCase):
    def test_target_depth_is_never_opened(self):
        with TemporaryDirectory() as temporary:
            base=Path(temporary)
            public,raw=base/'public',base/'private'
            public.mkdir()
            sequence='synthetic'
            folder='rgbd_dataset_freiburg3_'+sequence
            path=base/(folder+'.tgz')
            pose=np.eye(4).tolist()
            color=png(np.full((480,640,3),128,dtype=np.uint8))
            depth=png(np.full((480,640),10000,dtype=np.uint16))
            entries={}
            targets=[]
            forbidden=[]
            for i in range(5):
                rgb=f'rgb/{i}.png'
                entries[rgb]=color
                labels=f'depth/target_{i}.png'
                entries[labels]=b'UNREADABLE EVALUATION LABEL'
                forbidden.append(labels)
                sources=[]
                for j in range(2):
                    filename=f'depth/source_{i}_{j}.png'
                    entries[filename]=depth
                    sources.append(dict(depth_file=filename,depth_index=2*i+j,c2w=pose))
                targets.append(dict(rgb_file=rgb,rgb_time=float(i),c2w=pose,sources=sources))
            with tarfile.open(path,'w:gz') as archive:
                for name,value in entries.items():
                    info=tarfile.TarInfo(folder+'/'+name)
                    info.size=len(value)
                    archive.addfile(info,io.BytesIO(value))
            (public/'inference_selection.json').write_text(json.dumps({sequence:dict(targets=targets)}))
            (public/(sequence+'_download.json')).write_text(json.dumps(dict(archive_sha256=sha(path))))
            with patch.object(packet,'PUBLIC',public),patch.object(packet,'RAW',raw),patch.object(packet,'SEQUENCES',(sequence,)):
                packet.prepare(base)
            manifest=json.loads((public/'inputs.json').read_text())
            files=manifest['sequences'][sequence]['files']
            self.assertTrue(set(forbidden).isdisjoint(files))
            self.assertEqual(len(files),15)
            with np.load(public/'measurements.npz') as values:
                self.assertEqual(len(values.files),12)
                self.assertEqual(values[sequence+'_poses'].shape,(5,4,4))
            with np.load(raw/'inference'/(sequence+'.npz')) as values:
                self.assertEqual(values['images'].shape,(5,3,392,518))
