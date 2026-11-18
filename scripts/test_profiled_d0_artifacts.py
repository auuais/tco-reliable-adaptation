"""Synthetic D0 provenance/allowlist/loader checks; no research tensors are read."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import profiled_d0_artifacts as artifacts


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def put(root, name, value=b"synthetic opaque bytes"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return artifacts._descriptor(root, name)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.weights = self.root / "weights"
        self.weights.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def npz(self, name, values, **extra):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
        return dict(**artifacts._descriptor(self.root, name), **extra)


class InventoryTests(Fixture):
    def setUp(self):
        super().setUp()
        self.design_file = "manifests/design.json"
        checkpoint = put(self.weights, "model.safetensors")
        write_json(self.weights / "config.json", dict(dec_depth=12, dec_embed_dim=768))
        config = artifacts._descriptor(self.weights, "config.json")
        xy = put(self.root, artifacts.SPLITS)
        rgb = put(self.root, "reports/dense_focal_pilot/inference/heads_base_rgb.npz")
        write_json(self.root / artifacts.RGB_PLAN, dict(inputs=dict(heads_base=dict(
            file="heads_base_rgb.npz", sha256=rgb["sha256"]))))
        names = [f'heads_base_{rgb["sha256"][:12]}_{i}' for i in range(5)]
        hashes = [hashlib.md5(name.encode()).hexdigest() for name in names]
        self.jobs, self.cases = [], []
        for scene in ("heads", "chess", "fire"):
            for side in ("earlier", "later"):
                for arm in range(6):
                    run = f"{scene}_{side}_P{arm}"
                    job = dict(run_id=run, raw_attempt_parent="reports/cases/"+run)
                    self.jobs.append(job)
                    raw = job["raw_attempt_parent"] + "/fixture01"
                    raw_files = {}
                    adapter = dict(frozen_P0=True)
                    if arm:
                        desc = put(self.root, raw+"/adapters.npz")
                        raw_files["adapters.npz"] = desc["sha256"]
                        adapter = {key: desc[key] for key in ("file", "sha256")}
                    if run in artifacts.FUNCTIONAL_RUNS:
                        for i, j in artifacts.PAIRS:
                            name = f"endpoint_cache/forward/{hashes[i]}/{hashes[j]}.pth"
                            raw_files[name] = put(self.root, raw+"/"+name)["sha256"]
                    report = "results/cases/"+run+"/fixture01/complete.json"
                    completion = dict(job=job, run_id=run, attempt="fixture01", raw_attempt=raw,
                        raw_files=raw_files, checkpoint_sha256=checkpoint["sha256"],
                        docker_image="synthetic image", runtime_policy=dict(deterministic=True),
                        input_sha256={artifacts.SPLITS:xy["sha256"]}, endpoint=dict(capture=dict(
                            input_sha256=rgb["sha256"], prior_cache_files=0, inference_precision="float32")))
                    write_json(self.root / report, completion)
                    selected = dict(status="complete", attempt="fixture01", report=report,
                        report_sha256=artifacts.sha(self.root/report), raw_files=raw_files,
                        adapter=adapter, original_state_sha256="same original model")
                    self.cases.append(dict(run_id=run, selected=selected))
        registry_name = "manifests/parent.json"
        self.registry = dict(jobs=self.jobs, source_sha256_lf={},
            input_sha256={artifacts.SPLITS:xy["sha256"]}, checkpoint_sha256=checkpoint["sha256"])
        write_json(self.root/registry_name, self.registry)
        evaluation_sources={}
        for name in ("scripts/seal_profiled_predictions.py", "scripts/profiled_artifacts.py"):
            put(self.root,name,b"synthetic source\n")
            evaluation_sources[name]=artifacts.sha(self.root/name,text=True)
        write_json(self.root/"evaluation.json",dict(schema="profiled_evaluation_execution_v1",
                   source_sha256_lf=evaluation_sources))
        self.seal = dict(common_scientific_registry=registry_name,
            common_scientific_registry_sha256=artifacts.sha(self.root/registry_name),
            evaluation_execution=dict(file="evaluation.json", sha256=artifacts.sha(self.root/"evaluation.json")),
            pass_all=True, completed_states=36, cases=self.cases)
        write_json(self.root/artifacts.SEAL, self.seal)
        seal_sha = artifacts.sha(self.root/artifacts.SEAL)
        score_file = "results/score.json"
        write_json(self.root/score_file, dict(nominee=None, prediction_seal=artifacts.SEAL,
                                             prediction_seal_sha256=seal_sha))
        score_sha = artifacts.sha(self.root/score_file)
        write_json(self.root/artifacts.REPLAY, dict(pass_all=True, summary=dict(nominee=None),
            linkage=dict(prediction_seal=dict(file=artifacts.SEAL,sha256=seal_sha),
                         report=dict(file=score_file,sha256=score_sha))))
        self.design = dict(schema="profiled_lora_diagnosis_design_v1",
            stage="D0_saved_artifact_diagnosis_without_updates",
            trigger=dict(result="no_eligible_M2_nominee", score_file=score_file, score_sha256=score_sha,
                prediction_seal_sha256=seal_sha, independent_replay_sha256=artifacts.sha(self.root/artifacts.REPLAY)),
            adapter_norm_scope=dict(run_ids=list(artifacts.ADAPTER_RUNS),expected_adapters=30,
                expected_modules_per_adapter=144,expected_factor_arrays_per_adapter=288),
            functional_scope=dict(run_ids=list(artifacts.FUNCTIONAL_RUNS[:-1]),
                same_state_forward_comparator=artifacts.FUNCTIONAL_RUNS[-1],scene="heads",adaptation_source="earlier",
                frames=list(artifacts.FRAMES),directed_pairs=20,
                grid=copy.deepcopy(artifacts.GRID),quantiles=list(artifacts.QUANTILES)),
            immutable_model=dict(rank=4,alpha=1,checkpoint_sha256=checkpoint["sha256"],
                prospective_current_config_sha256=config["sha256"]),
            resources=dict(rental_usd=0,model_forward_calls=0,optimizer_updates=0,gpu_requested=False,
                max_cpu_wall_minutes=30,max_concurrent_jobs=1))
        write_json(self.root/self.design_file, self.design)
        self.seal_patch = patch.object(artifacts, "verify_seal", side_effect=lambda root, name, evaluation:
                                       artifacts.read(Path(root)/name))
        self.upstream_patch = patch.object(artifacts,"_upstream_sources",return_value={})
        self.seal_mock = self.seal_patch.start()
        self.upstream_patch.start()
        self.addCleanup(self.seal_patch.stop)
        self.addCleanup(self.upstream_patch.stop)

    def build(self):
        return artifacts.build_inventory(self.root,self.design_file,self.weights)

    def test_byte_only_build_and_exact_reverification(self):
        with patch.object(np,"load",side_effect=AssertionError("Forbidden real array decode")):
            inventory = self.build()
            self.assertEqual(artifacts.verify_inventory(self.root,self.weights,inventory),inventory)
        self.assertEqual(len(inventory["adapters"]),30)
        self.assertEqual([len(row["pairs"]) for row in inventory["functional"]],[20]*5)
        self.assertFalse(inventory["arrays_deserialized"])
        self.assertFalse(inventory["execution_authorized"])
        self.assertEqual(inventory["coordinates"]["members"],list(artifacts.XY_MEMBERS))
        self.seal_mock.assert_called_with(self.root,artifacts.SEAL,self.seal["evaluation_execution"])

    def test_parent_seal_failure_is_not_swallowed(self):
        self.seal_mock.side_effect=ValueError("parent mismatch")
        with self.assertRaisesRegex(ValueError,"parent mismatch"): self.build()

    def test_changed_adapter_byte_rejected(self):
        inventory=self.build()
        (self.root/inventory["adapters"][0]["file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"Changed registered bytes"): self.build()

    def test_appended_native_or_reordered_cases_rejected(self):
        original=self.build()
        for mutation in ("append", "reorder"):
            value=copy.deepcopy(original)
            if mutation=="append": value["functional"][0]["pairs"].append(dict(file="endpoint/native.npz"))
            else: value["adapters"].reverse()
            with self.assertRaisesRegex(ValueError,"inventory changed"):
                artifacts.verify_inventory(self.root,self.weights,value)

    def test_wrong_checkpoint_fails_before_decode(self):
        (self.weights/"model.safetensors").write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError,"Changed registered bytes"): self.build()

    def test_changed_design_scope_rejected(self):
        self.design["functional_scope"]["run_ids"][0]="heads_later_P3"
        write_json(self.root/self.design_file,self.design)
        with self.assertRaisesRegex(ValueError,"functional scope"): self.build()

    def test_grid_and_quantile_changes_rejected_before_parent_audit(self):
        original=copy.deepcopy(self.design)
        for mutation in ("grid", "quantiles", "missing_grid", "extra_grid_field"):
            self.design=copy.deepcopy(original)
            scope=self.design["functional_scope"]
            if mutation=="grid": scope["grid"]["stride"]=8
            elif mutation=="quantiles": scope["quantiles"]=[0,.5,1]
            elif mutation=="missing_grid": del scope["grid"]
            else: scope["grid"]["candidate_mask"]=True
            write_json(self.root/self.design_file,self.design)
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,"fixed grid or quantiles"):
                self.build()
        self.seal_mock.assert_not_called()

    def test_changed_evaluator_source_rejected_before_parent_audit(self):
        (self.root/"scripts/seal_profiled_predictions.py").write_bytes(b"unregistered sealer")
        with self.assertRaisesRegex(ValueError,"Frozen evaluation source changed"):
            self.build()
        self.seal_mock.assert_not_called()

    def test_changed_evaluation_registry_rejected_before_parent_audit(self):
        write_json(self.root/"evaluation.json",dict(schema="profiled_evaluation_execution_v1",source_sha256_lf={}))
        with self.assertRaisesRegex(ValueError,"Changed registered bytes"):
            self.build()
        self.seal_mock.assert_not_called()

    def test_changed_trigger_is_rejected(self):
        write_json(self.root/"results/score.json",dict(nominee="P3"))
        with self.assertRaisesRegex(ValueError,"Changed registered bytes"): self.build()

    def test_parent_completion_mismatch_rejected(self):
        selected=self.cases[0]["selected"]
        completion=artifacts.read(self.root/selected["report"])
        completion["raw_attempt"]="reports/other"
        write_json(self.root/selected["report"],completion)
        with self.assertRaisesRegex(ValueError,"raw inventory"): self.build()


class LoaderTests(Fixture):
    def adapter_values(self):
        values={}
        for module in artifacts.MODULES:
            values[module+".lora_A"]=np.zeros((4,768),dtype=np.float32)
            values[module+".lora_B"]=np.zeros((2304 if module.endswith("attn.qkv") else 768,4),dtype=np.float32)
        return values

    def adapter(self,values):
        return self.npz("adapter.npz",values,modules=144,factor_arrays=288,rank=4,alpha=1)

    def test_adapter_shapes_keys_float64_and_invalid_preservation(self):
        values=self.adapter_values()
        values[artifacts.MODULES[0]+".lora_A"][0,0]=np.nan
        loaded=artifacts.load_adapter(self.root,self.adapter(values))
        self.assertEqual(list(loaded),list(artifacts.MODULES))
        self.assertTrue(np.isnan(loaded[artifacts.MODULES[0]]["A"][0,0]))
        self.assertEqual(loaded[artifacts.MODULES[0]]["B"].shape,(2304,4))
        self.assertEqual(loaded[artifacts.MODULES[-1]]["B"].dtype,np.dtype("float64"))
        self.assertEqual(loaded[artifacts.MODULES[-1]]["base_key"],artifacts.MODULES[-1]+".weight")

    def test_missing_extra_transposed_and_wrong_dtype_factor_rejected(self):
        for mutation in ("missing","extra","transpose","dtype"):
            values=self.adapter_values()
            key=artifacts.MODULES[0]+".lora_A"
            if mutation=="missing": del values[key]
            elif mutation=="extra": values["unregistered.weight"]=np.zeros(1)
            elif mutation=="transpose": values[key]=values[key].T
            else: values[key]=values[key].astype(np.float64)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                artifacts.load_adapter(self.root,self.adapter(values))

    def test_wrong_rank_and_byte_mutation_rejected(self):
        descriptor=self.adapter(self.adapter_values())
        with self.assertRaisesRegex(ValueError,"factor contract"):
            artifacts.load_adapter(self.root,dict(descriptor,rank=8))
        (self.root/descriptor["file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"before decode"):
            artifacts.load_adapter(self.root,descriptor)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            artifacts._verified_path(self.root,dict(file="../outside.npz",sha256="0"*64))

    def test_xy_reads_only_five_allowed_members(self):
        coordinates=np.tile(np.array([[8,8],[24,24]],dtype=np.int32),(8,1))
        values={member:coordinates for member in artifacts.XY_MEMBERS}
        values["forbidden_depth"]=np.array([object()],dtype=object)
        descriptor=self.npz("splits.npz",values,members=list(artifacts.XY_MEMBERS),
            frames=list(artifacts.FRAMES),scene="heads",source="earlier")
        loaded=artifacts.load_xy(self.root,descriptor)
        self.assertEqual(len(loaded),5)
        self.assertTrue(all(np.array_equal(item,coordinates) for item in loaded))
        with self.assertRaisesRegex(ValueError,"allowlist"):
            artifacts.load_xy(self.root,dict(descriptor,source="later"))

    def test_xy_bounds_and_dtype_rejected(self):
        for value in (np.zeros((16,2),dtype=np.float32),np.full((16,2),900,dtype=np.int64)):
            descriptor=self.npz("splits.npz",{member:value for member in artifacts.XY_MEMBERS},
                members=list(artifacts.XY_MEMBERS),frames=list(artifacts.FRAMES),scene="heads",source="earlier")
            with self.assertRaises(ValueError): artifacts.load_xy(self.root,descriptor)

    def test_changed_bytes_after_decode_rejected(self):
        descriptor=self.adapter(self.adapter_values())
        original=artifacts._unchanged
        def mutate(path,desc):
            path.write_bytes(b"midread mutation")
            original(path,desc)
        with patch.object(artifacts,"_unchanged",side_effect=mutate),self.assertRaisesRegex(ValueError,"during decode"):
            artifacts.load_adapter(self.root,descriptor)

    def test_duplicate_npz_members_rejected(self):
        path=self.root/"duplicate.npz"
        data=io.BytesIO()
        np.save(data,np.zeros(1),allow_pickle=False)
        with zipfile.ZipFile(path,"w") as archive:
            archive.writestr("same.npy",data.getvalue())
            with self.assertWarns(UserWarning): archive.writestr("same.npy",data.getvalue())
        with np.load(path,allow_pickle=False) as archive,self.assertRaisesRegex(ValueError,"Duplicate"):
            artifacts._archive_members(archive)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch absent in this runtime")
class ForwardTests(Fixture):
    def setUp(self):
        super().setUp()
        import torch
        self.torch=torch
        xyz=torch.zeros((384,512,3),dtype=torch.float32)
        xyz[...,2]=2
        other=xyz.clone()
        other[...,2]=123
        self.values=(xyz,torch.ones((384,512)),other,torch.ones((384,512)))

    def save(self,values):
        self.torch.save(values,self.root/"pair.pth")
        return dict(**artifacts._descriptor(self.root,"pair.pth"),i=0,j=1)

    def test_first_branch_only_and_invalids_preserved(self):
        self.values[0][0,0,2]=float("nan")
        self.values[0][0,1,2]=-2
        descriptor=self.save(self.values)
        with patch.object(self.torch,"load",wraps=self.torch.load) as loader:
            depth=artifacts.load_forward(self.root,descriptor)
        loader.assert_called_once_with(self.root/"pair.pth",map_location="cpu",weights_only=True)
        self.assertTrue(np.isnan(depth[0,0]))
        self.assertEqual(depth[0,1],-2)
        self.assertEqual(depth[1,1],2)
        self.assertEqual(depth.dtype,np.dtype("float64"))

    def test_non_tuple_shape_and_dtype_rejected(self):
        variants=(list(self.values),self.values[:3],
                  (self.values[0][:1],)+self.values[1:],
                  (self.values[0].double(),)+self.values[1:])
        for values in variants:
            with self.subTest(kind=type(values)),self.assertRaises(ValueError):
                artifacts.load_forward(self.root,self.save(values))

    def test_invalid_pair_rejected(self):
        descriptor=self.save(self.values)
        with self.assertRaisesRegex(ValueError,"directed pair"):
            artifacts.load_forward(self.root,dict(descriptor,i=1,j=1))


class BaseNormTests(Fixture):
    def test_streamed_exact_modules_and_frobenius(self):
        descriptor=put(self.weights,"model.safetensors")
        shapes={module:(2304 if module.endswith("attn.qkv") else 768,768) for module in artifacts.MODULES}
        calls=[]
        class FakeArchive:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def keys(self): return [module+".weight" for module in artifacts.MODULES]+["unrelated.weight"]
            def get_tensor(self,name):
                calls.append(name)
                module=name[:-7]
                return np.ones(shapes[module],dtype=np.float32)
        fake=types.SimpleNamespace(safe_open=lambda *args,**kwargs:FakeArchive())
        with patch.dict("sys.modules",{"safetensors":fake}):
            norms=artifacts.load_base_weight_norms(self.weights,descriptor,shapes)
        self.assertEqual(calls,[module+".weight" for module in artifacts.MODULES])
        self.assertEqual(norms[artifacts.MODULES[0]],float(np.sqrt(2304*768)))
        self.assertEqual(norms[artifacts.MODULES[-1]],768.)
        with patch.dict("sys.modules",{"safetensors":fake}),self.assertRaisesRegex(ValueError,"exactly 144"):
            artifacts.load_base_weight_norms(self.weights,descriptor,{})


class UpstreamTests(Fixture):
    def test_safe_directory_and_pinned_blob_comparison(self):
        revisions={checkout:"a"*40 for checkout,_ in artifacts.UPSTREAM}
        for checkout,name in artifacts.UPSTREAM:
            put(self.root,checkout+"/"+name,b"source\r\n")
        with patch.object(artifacts.subprocess,"check_output",return_value=b"source\n") as git:
            result=artifacts._upstream_sources(self.root,dict(upstream_revisions=revisions))
        self.assertEqual(len(result),3)
        for call,(checkout,name) in zip(git.call_args_list,artifacts.UPSTREAM):
            self.assertEqual(call.args[0],["git","-c","safe.directory="+str((self.root/checkout).resolve()),
                "-C",str((self.root/checkout).resolve()),"show","a"*40+":"+name])

    def test_changed_source_and_missing_directory_rejected(self):
        revisions={checkout:"a"*40 for checkout,_ in artifacts.UPSTREAM}
        with self.assertRaisesRegex(ValueError,"checkout directory"):
            artifacts._upstream_sources(self.root,dict(upstream_revisions=revisions))
        for checkout,name in artifacts.UPSTREAM:
            put(self.root,checkout+"/"+name,b"changed\n")
        with patch.object(artifacts.subprocess,"check_output",return_value=b"source\n"),self.assertRaisesRegex(ValueError,"pinned upstream Git"):
            artifacts._upstream_sources(self.root,dict(upstream_revisions=revisions))


if __name__ == "__main__":
    unittest.main()
