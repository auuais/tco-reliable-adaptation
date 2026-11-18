"""Temporary synthetic D1 input-boundary tests; no production arrays are read."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

import profiled_d1_artifacts as a


def put(root, name, content=b"opaque synthetic artifact"):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    return a._descriptor(root, name)


def put_json(root, name, value):
    return put(root, name, json.dumps(value, sort_keys=True).encode())


def parents():
    return {run: dict(attempt="fixture01", raw_root=f"reports/lora_profiled_v1/cases/{run}/fixture01",
                      public_root=f"results/lora_profiled_v1/cases/{run}/fixture01") for run in a.RUNS}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.weights = self.root/"weights"; self.weights.mkdir()
        p = parents()
        names = dict(rgb=a.RGB, packet=a.PACKET, splits=a.SPLITS, poses=a.POSES,
                     teacher=p[a.RUNS[1]]["raw_root"]+"/teacher_initial.npz",
                     adapter=p[a.RUNS[1]]["raw_root"]+"/adapters.npz")
        roles = {role: dict(role=role, members=list(a.ROLE_MEMBERS[role]), **put(self.root, path))
                 for role, path in names.items()}
        self.inventory = dict(schema="profiled_d1_artifact_inventory_v1", scene="heads", adaptation_source="earlier",
                              frames=list(a.FRAMES), runs=list(a.RUNS), module_names=list(a.MODULES),
                              parent_evidence=p, roles=roles)
        self.reset_rgb_identity()

    def tearDown(self):
        self.temp.cleanup()

    def reset_rgb_identity(self):
        digest = self.inventory["roles"]["rgb"]["sha256"]
        names = [f'heads_base_{digest[:12]}_{i}' for i in range(5)]
        hashes = [hashlib.md5(name.encode()).hexdigest() for name in names]
        self.inventory.update(rgb_archive_sha256=digest, instances=names, forward={})
        for run in a.RUNS:
            raw = self.inventory["parent_evidence"][run]["raw_root"]
            self.inventory["forward"][run] = [dict(role="fp32_first_camera_xyz", run_id=run, i=i, j=j,
                **put(self.root, raw+f"/endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth")) for i,j in a.PAIRS]

    def npz(self, role, values):
        desc = self.inventory["roles"][role]
        np.savez_compressed(self.root/desc["file"], **values)
        desc.update(a._descriptor(self.root, desc["file"]))
        if role == "rgb": self.reset_rgb_identity()
        return desc

    def adaptation_values(self):
        packet, splits = {}, {}
        # Sixteen native pixels in distinct 8x8 anchor cells, distinct sensor rays.
        for frame in a.FRAMES:
            rows = np.arange(16, dtype=np.int64)
            u = 10+rows*24
            target = np.column_stack((u, np.full(16, 40)))
            source_u, source_v = 40+(rows%8)*80, 30+(rows//8)*60
            values = np.column_stack((target, 1.+rows*.01, np.full(16, frame-4),
                                      source_u, source_v, np.full(16, 1000))).astype(np.float64)
            xy = np.rint((target+.5)*[512/518, 384/392]-.5).astype(np.int64)
            packet[f"heads_{frame}"] = values
            prefix = f"heads_earlier_{frame}"
            splits.update({prefix+"__rows":rows, prefix+"__xy":xy,
                           prefix+"__fold":(rows%2).astype(np.int8)})
        return packet, splits


class LoaderTests(Fixture):
    def test_rgb_only_images_member_is_decoded(self):
        self.npz("rgb", dict(images=np.full((5,3,392,518), .5, dtype=np.float32),
                             dense_reference=np.array([object()], dtype=object)))
        original = np.lib.npyio.NpzFile.__getitem__; accessed=[]
        def guarded(archive, key):
            accessed.append(key)
            if key != "images": raise AssertionError("Unauthorized member access")
            return original(archive, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded):
            result = a.load_rgb(self.root, self.inventory)
        self.assertEqual(accessed, ["images"])
        self.assertEqual(result.shape, (5,3,392,518)); self.assertEqual(result.dtype, np.float32)

    def test_rgb_wrong_dtype_shape_range_and_nonfinite_rejected(self):
        for value in (np.zeros((1,3,392,518), np.float32), np.zeros((5,3,392,518), np.float64),
                      np.full((5,3,392,518), 2., np.float32), np.full((5,3,392,518), np.nan, np.float32)):
            self.npz("rgb", dict(images=value))
            with self.assertRaises(ValueError): a.load_rgb(self.root, self.inventory)

    def test_adaptation_reads_only_five_earlier_members_and_fifteen_split_members(self):
        packet, splits = self.adaptation_values()
        packet["chess_25"] = np.array([object()], dtype=object)
        splits["heads_later_25__xy"] = np.array([object()], dtype=object)
        self.npz("packet", packet); self.npz("splits", splits)
        original = np.lib.npyio.NpzFile.__getitem__; accessed=[]
        def guarded(archive, key):
            accessed.append(key)
            if key not in set(a.PACKET_MEMBERS+a.SPLIT_MEMBERS): raise AssertionError("Unauthorized member")
            return original(archive, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded):
            result = a.load_adaptation(self.root, self.inventory)
        self.assertEqual(set(accessed), set(a.PACKET_MEMBERS+a.SPLIT_MEMBERS))
        self.assertEqual(len(result), 5)
        self.assertEqual([r["frame"] for r in result], list(a.FRAMES))
        for row in result:
            self.assertEqual(row["xy"].dtype, np.int64); self.assertEqual(row["fold"].dtype, np.int8)
            self.assertEqual(row["depth"].dtype, np.float64)

    def test_wrong_signed_source_or_nonpositive_depth_rejected(self):
        for column, value in ((3, 29), (2, 0), (6, 0), (4, 41)):
            packet, splits = self.adaptation_values(); packet["heads_25"][0,column] = value
            self.npz("packet", packet); self.npz("splits", splits)
            with self.subTest(column=column), self.assertRaises(ValueError):
                a.load_adaptation(self.root, self.inventory)

    def test_wrong_xy_rows_folds_and_dtypes_rejected(self):
        for field in ("xy", "rows", "fold", "dtype"):
            packet, splits = self.adaptation_values()
            if field == "xy": splits["heads_earlier_25__xy"][0,0] += 1
            if field == "rows": splits["heads_earlier_25__rows"] = splits["heads_earlier_25__rows"][::-1]
            if field == "fold": splits["heads_earlier_25__fold"][:] = 0
            if field == "dtype": splits["heads_earlier_25__fold"] = splits["heads_earlier_25__fold"].astype(np.int64)
            self.npz("packet", packet); self.npz("splits", splits)
            with self.subTest(field=field), self.assertRaises(ValueError): a.load_adaptation(self.root, self.inventory)

    def test_original_approximately_rigid_pose_is_preserved_without_repair(self):
        pose = np.repeat(np.eye(4)[None], 5, axis=0); pose[0,0,0] += 1e-6
        self.npz("poses", dict(heads_clean=pose, heads_noisy=np.array([object()], dtype=object)))
        actual = a.load_poses(self.root, self.inventory)
        np.testing.assert_array_equal(actual, pose)

    def test_wrong_pose_dtype_homogeneous_row_or_singularity_rejected(self):
        pose = np.repeat(np.eye(4)[None], 5, axis=0)
        variants = [pose.astype(np.float32), pose.copy(), pose.copy()]
        variants[1][0,3,0] = .1; variants[2][0,0] = 0
        for value in variants:
            self.npz("poses", dict(heads_clean=value))
            with self.assertRaises(ValueError): a.load_poses(self.root, self.inventory)

    def test_teacher_exact_two_fields_and_dtype(self):
        values = dict(depth=np.ones((5,384,512), np.float32), confidence=np.ones((5,384,512), np.float32))
        self.npz("teacher", values)
        actual = a.load_teacher(self.root, self.inventory)
        self.assertEqual(set(actual), {"depth", "confidence"})
        values["reference"] = np.zeros(1)
        self.npz("teacher", values)
        with self.assertRaises(ValueError): a.load_teacher(self.root, self.inventory)

    def test_adapter_exact_288_names_orientations_fp32_and_finiteness(self):
        values = {}
        for module in a.MODULES:
            values[module+".lora_A"] = np.ones((4,768), np.float32)
            values[module+".lora_B"] = np.zeros((2304 if module.endswith("attn.qkv") else 768,4), np.float32)
        self.npz("adapter", values)
        actual = a.load_adapter(self.root, self.inventory)
        self.assertEqual(len(actual), 288)
        first = a.FACTOR_MEMBERS[0]
        values[first] = values[first].T
        self.npz("adapter", values)
        with self.assertRaises(ValueError): a.load_adapter(self.root, self.inventory)
        values[first] = np.full((4,768), np.nan, np.float32)
        self.npz("adapter", values)
        with self.assertRaises(ValueError): a.load_adapter(self.root, self.inventory)

    def test_role_member_other_scene_path_and_traversal_tampering_fail_before_load(self):
        mutations = [("member", lambda inv:inv["roles"]["poses"].update(members=["heads_noisy"])),
                     ("side", lambda inv:inv["roles"]["packet"].update(file="results/mast3r_partner_shape/later.npz")),
                     ("path", lambda inv:inv["roles"]["poses"].update(file="../poses.npz")),
                     ("scene", lambda inv:inv.update(scene="chess"))]
        for label, mutate in mutations:
            inv = copy.deepcopy(self.inventory); mutate(inv)
            with self.subTest(label=label), patch.object(np, "load", side_effect=AssertionError("Must not decode")):
                with self.assertRaises(ValueError): a.load_poses(self.root, inv)

    def test_hash_mismatch_before_decode_and_mutation_during_decode(self):
        self.npz("poses", dict(heads_clean=np.repeat(np.eye(4)[None], 5, axis=0)))
        descriptor = self.inventory["roles"]["poses"]; path = self.root/descriptor["file"]
        expected = path.read_bytes(); path.write_bytes(expected+b"changed")
        with patch.object(np, "load", side_effect=AssertionError("Must not decode")):
            with self.assertRaises(ValueError): a.load_poses(self.root, self.inventory)
        path.write_bytes(expected)
        original = np.lib.npyio.NpzFile.__getitem__
        def changed(archive, key):
            result = original(archive, key); path.write_bytes(expected+b"mutated")
            return result
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", changed):
            with self.assertRaisesRegex(ValueError, "changed during decode"):
                a.load_poses(self.root, self.inventory)

    def test_duplicate_unselected_zip_member_is_rejected(self):
        self.npz("poses", dict(heads_clean=np.repeat(np.eye(4)[None],5,axis=0), unused=np.zeros(1)))
        desc = self.inventory["roles"]["poses"]; path = self.root/desc["file"]
        buf=io.BytesIO(); np.save(buf,np.ones(1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(path,"a") as archive: archive.writestr("unused.npy",buf.getvalue())
        desc.update(a._descriptor(self.root,desc["file"]))
        with patch.object(np,"load",side_effect=AssertionError("Must not decode")):
            with self.assertRaisesRegex(ValueError,"Duplicate ZIP"):
                a.load_poses(self.root,self.inventory)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch only needed for synthetic raw-cache test")
    def test_forward_returns_only_first_xyz_with_safe_cpu_load(self):
        import torch
        desc = self.inventory["forward"][a.RUNS[0]][0]; path = self.root/desc["file"]
        first=torch.ones((384,512,3),dtype=torch.float32)
        values=(first,torch.ones((384,512),dtype=torch.float32)*3,
                first*7,torch.ones((384,512),dtype=torch.float32)*9)
        torch.save(values,path); desc.update(a._descriptor(self.root,desc["file"]))
        original = torch.load
        with patch.object(torch,"load",wraps=original) as load:
            actual=a.load_forward(self.root,self.inventory,a.RUNS[0],0,1)
        self.assertEqual(load.call_args.kwargs,dict(map_location="cpu",weights_only=True))
        np.testing.assert_array_equal(actual,np.ones((384,512,3),np.float32))
        self.assertEqual(actual.dtype,np.float32)
        with self.assertRaises(ValueError): a.load_forward(self.root,self.inventory,"heads_earlier_P2",0,1)
        with self.assertRaises(ValueError): a.load_forward(self.root,self.inventory,a.RUNS[0],0,0)
        torch.save(list(values),path); desc.update(a._descriptor(self.root,desc["file"]))
        with self.assertRaises(ValueError): a.load_forward(self.root,self.inventory,a.RUNS[0],0,1)


class InventoryTests(Fixture):
    def metadata(self):
        inv=self.inventory
        checkpoint=put(self.weights,"model.safetensors")
        config=put_json(self.weights,"config.json",dict(dec_depth=12,dec_embed_dim=768))
        inputs=put_json(self.root,a.INPUTS,dict(
            splits_binary_sha256=inv["roles"]["splits"]["sha256"],registration=dict(
                packet_sha256=dict(earlier=inv["roles"]["packet"]["sha256"]),
                immutable_sha256={a.POSES:inv["roles"]["poses"]["sha256"]})))
        rgb_plan=put_json(self.root,a.RGB_PLAN,dict(inputs=dict(heads_base=dict(
            file="heads_base_rgb.npz",sha256=inv["rgb_archive_sha256"],scene="heads/seq-01",batch="base"))))
        jobs=[];cases=[]
        for run in a.RUNS:
            p=inv["parent_evidence"][run];adapt=run==a.RUNS[1]
            job=dict(run_id=run,scene="heads",adaptation_source="earlier",arm=run.split("_")[-1],
                input_packet=a.PACKET,raw_attempt_parent=p["raw_root"].rsplit("/",1)[0],
                public_attempt_parent=p["public_root"].rsplit("/",1)[0],adapt_lora=adapt,graph_weight=0,
                differentiate_solve=adapt)
            raw_files={v["file"][len(p["raw_root"])+1:]:v["sha256"] for v in inv["forward"][run]}
            public_files={}
            adapter=dict(frozen_P0=True)
            if adapt:
                for role in ("teacher","adapter"):
                    value=inv["roles"][role];raw_files[value["file"].rsplit("/",1)[-1]]=value["sha256"]
                adapter={k:inv["roles"]["adapter"][k] for k in ("file","sha256")}
                adapter["state_sha256"]="f"*64
                teacher=put_json(self.root,p["public_root"]+"/teacher.json",dict(decoder_calls=20,
                    checkpoint_sha256=checkpoint["sha256"],initial_adapter_state_sha256="a"*64,
                    full_pair_pointmaps_archived=False,fused_native_file=inv["roles"]["teacher"]["file"],
                    fused_native_sha256=inv["roles"]["teacher"]["sha256"]))
                public_files["teacher.json"]=teacher["sha256"]
            complete=dict(run_id=run,job=job,attempt=p["attempt"],raw_attempt=p["raw_root"],raw_files=raw_files,
                public_files=public_files,checkpoint_sha256=checkpoint["sha256"],original_state_unchanged=True,
                original_state_sha256="b"*64,pre_endpoint_original_state_sha256="b"*64,final_original_state_sha256="b"*64,
                input_sha256={a.INPUTS:inputs["sha256"]},endpoint=dict(capture=dict(input_sha256=inv["rgb_archive_sha256"],
                    prior_cache_files=0,inference_precision="float32")),docker_image="synthetic-pinned-image",
                runtime_policy=dict(deterministic=True),adapter_module_count=144 if adapt else 0,
                adapter_initial_sha256="a"*64 if adapt else None,adapter_final_sha256="f"*64 if adapt else None)
            desc=put_json(self.root,p["public_root"]+"/complete.json",complete)
            jobs.append(job);cases.append(dict(run_id=run,selected=dict(status="complete",attempt=p["attempt"],
                report=desc["file"],report_sha256=desc["sha256"],raw_files=raw_files,adapter=adapter,original_state_sha256="b"*64)))
        source=put(self.root,"scripts/synthetic_source.py",b"# fixture source\n")
        source_map={source["file"]:a.sha(self.root/source["file"],text=True)}
        registry=put_json(self.root,"manifests/synthetic_parent.json",dict(jobs=jobs,source_sha256_lf=source_map,
            checkpoint_sha256=checkpoint["sha256"],input_sha256={a.INPUTS:inputs["sha256"],a.SPLITS:inv["roles"]["splits"]["sha256"]}))
        seal=put_json(self.root,"results/synthetic_seal.json",dict(pass_all=True,completed_states=36,
            common_scientific_registry=registry["file"],common_scientific_registry_sha256=registry["sha256"],cases=cases))
        previous=put_json(self.root,"manifests/synthetic_d0.json",dict(schema="profiled_d0_execution_v1",source_sha256_lf=source_map,
            inventory=dict(schema="profiled_d0_artifact_inventory_v1",trigger=dict(seal=seal),parent_registry=registry,
                upstream_sources={},checkpoint=checkpoint,config=config,rgb_plan=rgb_plan,
                rgb_archive={k:inv["roles"]["rgb"][k] for k in ("file","sha256","bytes")})))
        report=put_json(self.root,"results/synthetic_d0.json",dict(schema="profiled_d0_report_v1",status="complete",
            registration={k:previous[k] for k in ("file","sha256")}))
        design=dict(schema="profiled_lora_d1_design_v1",stage="D1_single_paired_inference_and_calibration_response_probe_without_updates",
            trigger=dict(d0_execution=previous,d0_report=report,exact_d0_replay_required_before_d1_execution=True),
            scope=dict(scene="heads",adaptation_source="earlier",frames=list(a.FRAMES),trained_state=a.RUNS[1],
                frozen_state=a.RUNS[0],directed_pairs=20,support_folds=[0,1],graph_weight=0),
            resources=dict(total_new_decoder_calls=40,frozen_encoder_feature_computations=5,
                pristine_training_path_decoder_calls=20,final_training_path_decoder_calls=20,
                new_FP32_decoder_calls=0,network_backward_calls=0,optimizer_updates=0,rental_usd=0),
            immutable_model=dict(checkpoint=checkpoint,config=config,original_parameters_and_buffers_frozen=True,
                adapter=dict(run_id=a.RUNS[1],**{k:inv["roles"]["adapter"][k] for k in ("file","sha256")})))
        put_json(self.root,"manifests/synthetic_d1.json",design)
        return design

    def test_build_and_verify_hash_opaque_bytes_without_tensor_decode(self):
        self.metadata()
        with patch.object(np,"load",side_effect=AssertionError("No preparation decoding")):
            result=a.build_inventory(self.root,"manifests/synthetic_d1.json",self.weights)
            self.assertEqual(a.verify_inventory(self.root,self.weights,result),result)
        self.assertEqual(set(result["roles"]),set(a.ROLES))
        self.assertEqual(len(result["forward"][a.RUNS[0]]),20)
        self.assertEqual(len(result["forward"][a.RUNS[1]]),20)
        self.assertFalse(result["execution_authorized"]);self.assertFalse(result["arrays_deserialized"])

    def test_changed_source_and_selected_artifact_bytes_rejected(self):
        self.metadata();result=a.build_inventory(self.root,"manifests/synthetic_d1.json",self.weights)
        selected=self.root/result["roles"]["packet"]["file"];original=selected.read_bytes();selected.write_bytes(original+b"changed")
        with self.assertRaises(ValueError): a.verify_inventory(self.root,self.weights,result)
        selected.write_bytes(original)
        (self.root/"scripts/synthetic_source.py").write_bytes(b"# changed\n")
        with self.assertRaises(ValueError): a.verify_inventory(self.root,self.weights,result)

    def test_registration_membership_state_and_design_tampering_rejected(self):
        design=self.metadata();result=a.build_inventory(self.root,"manifests/synthetic_d1.json",self.weights)
        changed=copy.deepcopy(result);changed["forward"][a.RUNS[1]][0]["run_id"]="heads_later_P3"
        with self.assertRaises(ValueError):a.verify_inventory(self.root,self.weights,changed)
        changed=copy.deepcopy(result);changed["roles"]["splits"]["members"].append("heads_later_25__depth")
        with self.assertRaises(ValueError):a.verify_inventory(self.root,self.weights,changed)
        design["scope"]["graph_weight"]=.1;put_json(self.root,"manifests/synthetic_d1.json",design)
        with self.assertRaises(ValueError):a.build_inventory(self.root,"manifests/synthetic_d1.json",self.weights)


if __name__ == "__main__":
    unittest.main()
