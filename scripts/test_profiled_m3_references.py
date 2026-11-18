"""Synthetic byte/camera preparation checks; no study archives are decoded."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_profiled_m3_references as m


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReferencePreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repository"
        self.assets = Path(self.tmp.name) / "assets"
        self.assets.mkdir()
        for name in m.SOURCES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((m.ROOT/name).read_bytes())
        calibration = np.eye(4)
        calibration[:3, 3] = [.02, -.01, .03]
        index = dict(inputs={}, references={})
        poses = dict(frames={})
        audit = dict(reference_cameras={},same_view_checks={},
            source_sha256=m.sha(self.root/"scripts/audit_mast3r_crossover_matches.py",text=True))
        for i,(scene,sequence) in enumerate(m.SCENES.items()):
            key=scene+"_base"
            for kind,parent in (("rgb","inference"),("reference","evaluation")):
                name=key+("_rgb.npz" if kind=="rgb" else ".npz")
                path=self.root/"reports/dense_focal_pilot"/parent/name
                path.parent.mkdir(parents=True,exist_ok=True)
                # Deliberately not an NPZ: decoding must never be attempted.
                path.write_bytes((scene+"/"+kind+"/synthetic opaque bytes").encode())
                record=dict(file=name,sha256=m.sha(path))
                if kind=="rgb":
                    index["inputs"][key]=dict(record,scene=sequence,batch="base")
                else:
                    index["references"][key]=dict(record,valid_pixels=300+i)
            poses["frames"][sequence]={}
            expected=[]; hashes={}
            for j,frame in enumerate(m.FRAMES):
                original=np.eye(4)
                # Nonorthogonal source rotations expose the transpose/inverse convention.
                original[:3,:3]=[[1.0002,.0001,0.],[-.0001,.9998,.0002],[0.,-.0002,1.0001]]
                original[:3,3]=[.2*j,.05*i,1.+.1*j]
                path=self.assets/sequence/f"frame-{frame:06d}.pose.txt"
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savetxt(path,original,fmt="%.18e")
                digest=m.sha(path)
                hashes[str(frame)]=digest
                poses["frames"][sequence][str(frame)]={"original":{"pose.txt":digest}}
                # Independent collapsed algebra of the two historical transpose inversions.
                constructed=original.copy()
                constructed[:3,3]=original[:3,:3]@original[:3,:3].T@original[:3,3]
                expected.append(calibration@np.linalg.inv(constructed))
            audit["reference_cameras"][scene]=dict(world_to_camera=np.stack(expected).tolist(),original_pose_sha256=hashes)
            audit["same_view_checks"][scene]=dict(reference_sha256=index["references"][key]["sha256"])
        for name,value in ((m.INDEX,index),(m.POSE_INDEX,poses),(m.AUDIT,audit),
                           (m.CALIBRATION,dict(calibration=calibration.tolist())),
                           (m.GEOMETRY_PLAN,dict(scenes=m.SCENES,frames=m.FRAMES))):
            dump(self.root/name,value)

    def change(self,name,update):
        value=m.read(self.root/name)
        update(value)
        dump(self.root/name,value)

    def test_complete_preparation_never_decodes_archives(self):
        with patch.object(np,"load",side_effect=AssertionError("Archive decode forbidden")):
            record=m.build(self.root,self.assets)
        self.assertEqual(record["original_pose_text_files_read"],15)
        self.assertEqual(record["image_depth_prediction_reference_archives_decoded"],0)
        self.assertFalse(record["release_authorized"])
        self.assertEqual(record,m.build(self.root,self.assets))
        for row in record["scenes"].values():
            self.assertIsNone(row["fixed_mask"]["tensor_sha256"])
            self.assertEqual(row["alternate_axial_camera"]["status"],"unsupported_not_registered")

    def test_camera_values_are_canonical_archived_metadata(self):
        record=m.build(self.root,self.assets)
        audit=m.read(self.root/m.AUDIT)
        for scene,row in record["scenes"].items():
            cameras=row["primary_axial_camera"]
            self.assertEqual(cameras["world_to_camera"],audit["reference_cameras"][scene]["world_to_camera"])
            self.assertTrue(cameras["historical_camera_reproduction_passed"])
            self.assertFalse(cameras["matrices_orthogonalized"])

    def test_plain_inverse_camera_cannot_replace_reference_convention(self):
        original=np.stack([np.loadtxt(self.assets/"heads/seq-01"/f"frame-{f:06d}.pose.txt") for f in m.FRAMES])
        calibration=np.array(m.read(self.root/m.CALIBRATION)["calibration"])
        wrong=calibration@np.linalg.inv(original)
        self.change(m.AUDIT,lambda value:value["reference_cameras"]["heads"].update(world_to_camera=wrong.tolist()))
        with self.assertRaisesRegex(ValueError,"convention"):
            m.build(self.root,self.assets)

    def test_archive_mutation_rejected_before_any_decode(self):
        path=self.root/"reports/dense_focal_pilot/evaluation/heads_base.npz"
        path.write_bytes(path.read_bytes()+b"changed")
        with patch.object(np,"load",side_effect=AssertionError("Decode forbidden")):
            with self.assertRaisesRegex(ValueError,"archive changed"):
                m.build(self.root,self.assets)

    def test_rgb_mutation_is_detected(self):
        (self.root/"reports/dense_focal_pilot/inference/heads_base_rgb.npz").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"rgb archive changed"):
            m.build(self.root,self.assets)

    def test_original_pose_mutation_is_detected(self):
        path=self.assets/"heads/seq-01/frame-000025.pose.txt"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError,"Original pose changed"):
            m.build(self.root,self.assets)

    def test_old_camera_audit_from_other_reference_is_rejected(self):
        self.change(m.AUDIT,lambda value:value["same_view_checks"]["heads"].update(reference_sha256="0"*64))
        with self.assertRaisesRegex(ValueError,"another dense reference"):
            m.build(self.root,self.assets)

    def test_camera_source_mutation_is_detected(self):
        path=self.root/"scripts/audit_mast3r_crossover_matches.py"
        path.write_bytes(path.read_bytes()+b"# changed\n")
        with self.assertRaisesRegex(ValueError,"camera-audit source"):
            m.build(self.root,self.assets)

    def test_wrong_scene_frames_are_rejected(self):
        self.change(m.GEOMETRY_PLAN,lambda value:value.update(frames=[0,200,400,600,800]))
        with self.assertRaisesRegex(ValueError,"Scene/frame"):
            m.build(self.root,self.assets)

    def test_invalid_camera_and_mask_metadata_rejected(self):
        for field,value in (("valid_pixels",0),("valid_pixels",True),("valid_pixels",5*392*518+1)):
            with self.subTest(value=value):
                original=m.read(self.root/m.INDEX)
                self.change(m.INDEX,lambda data:data["references"]["heads_base"].update({field:value}))
                with self.assertRaisesRegex(ValueError,"mask count"):
                    m.build(self.root,self.assets)
                dump(self.root/m.INDEX,original)
        self.change(m.AUDIT,lambda value:value["reference_cameras"]["heads"]["world_to_camera"][0][3].__setitem__(0,.1))
        with self.assertRaisesRegex(ValueError,"homogeneous"):
            m.build(self.root,self.assets)

    def test_paths_cannot_escape_repository(self):
        for name in ("../outside","C:/outside","/tmp/outside","scripts\\source.py"):
            with self.subTest(name=name),self.assertRaises(ValueError):
                m.relative(self.root,name)

    def test_nonfinite_or_left_handed_camera_rejected(self):
        for matrix in (np.diag([-1.,1.,1.,1.]),np.full((4,4),np.nan)):
            with self.assertRaises(ValueError):
                m.matrix_packet([matrix],1,"fixture")

    def test_tensor_digest_binds_shape_and_dtype(self):
        a=np.arange(16,dtype=np.float64)
        self.assertNotEqual(m.tensor_sha(a),m.tensor_sha(a.reshape(4,4)))
        self.assertNotEqual(m.tensor_sha(a),m.tensor_sha(a.astype(np.float32)))


if __name__ == "__main__":
    unittest.main()
