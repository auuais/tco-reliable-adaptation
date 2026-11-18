"""Opaque-byte synthetic registry fixtures; no scientific arrays or scores."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import profiled_m3_registry as registry
from profiled_m3_inventory import NOMINATION_FIELDS
from register_profiled_execution import SOURCES, SCIENTIFIC_SOURCES


ROOT = Path(__file__).resolve().parents[1]


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Fixture:
    """Reusable temporary full360 registry for controller/attempt-audit tests."""
    def __init__(self, root, nominee="P3"):
        self.root = Path(root)
        self.registry_name = "manifests/synthetic_m3_execution.json"
        self.weights = self.root / "fixture_weights"
        self.weights.mkdir(parents=True)
        (self.weights / "model.safetensors").write_bytes(b"synthetic opaque weight bytes")
        (self.weights / "config.json").write_bytes(b'{"fixture":true}')
        checkpoint = registry.sha(self.weights / "model.safetensors")
        config = registry.sha(self.weights / "config.json")
        names = set(SOURCES) | set(SCIENTIFIC_SOURCES) | set(registry.EXTRA_SOURCES)
        self.sources = {}
        for name in sorted(names):
            self.put(name, (ROOT / name).read_bytes())
            self.sources[name] = registry.sha(self.root / name, text=True)
        runtime = dict(torch="2.7.1+cu128", cuda="12.8", gpu="NVIDIA GeForce RTX 3080",
                       policy=deepcopy(registry.EXPECTED_RUNTIME))
        environment = {key: runtime[key] for key in ("torch", "cuda", "gpu")}
        environment.update(docker_image=registry.IMAGE, checkpoint_sha256=checkpoint)
        self.put(registry.ENVIRONMENT, environment)
        self.put(registry.BRIDGE, dict(checkpoint_sha256=checkpoint, original_state_hash=h("original"),
                                      final_state_hash=h("original"), pass_all=True))
        self.put(registry.BASELINE, dict(scenes={scene:dict(scene=scene, fixture=True) for scene in registry.SCENES}))
        endpoints = dict(native_coarse_iterations=300, native_fine_iterations=300, native_pose_weight=.01,
                         final_graph_weight=.1, primary="common final", secondary="native")
        m2 = dict(schema="profiled_scientific_v1", phase="scientific", scientific_ready_at_registration=True,
            source_sha256_lf={name:self.sources[name] for name in (*SOURCES,*SCIENTIFIC_SOURCES)},
            artifact_sha256={name:registry.sha(self.root/name) for name in (registry.ENVIRONMENT,registry.BASELINE)},
            docker_image=registry.IMAGE, checkpoint_sha256=checkpoint,
            runtime_policy=deepcopy(registry.EXPECTED_RUNTIME), endpoints=endpoints)
        self.put(registry.M2,m2)
        rgb = dict(inputs={})
        rgb_names = []
        for scene in registry.SCENES:
            name = f"reports/dense_focal_pilot/inference/{scene}_base_rgb.npz"
            self.put(name, (scene + "/opaque RGB not NPZ").encode())
            rgb_names.append(name)
            rgb["inputs"][scene+"_base"] = dict(file=Path(name).name,sha256=registry.sha(self.root/name),
                rgb_sha256=h(scene+"/rgb tensor"),batch="base",
                scene={"heads":"heads/seq-01","chess":"chess/seq-03","fire":"fire/seq-03"}[scene])
        self.put(registry.RGB,rgb)
        base = f"results/lora_profiled_m3_v1/input_attempts/{registry.INPUT_ATTEMPT}"
        input_files = ["results/lora_profiled_v1/inputs.json", "results/lora_profiled_v1/splits.npz",
                       base+"/poses/pose_packets.npz"]
        for side in registry.SIDES:
            input_files.extend((f"results/mast3r_partner_shape/{side}.npz",base+f"/{side}/shifted_packet.npz",
                                base+f"/{side}/shifted_splits.npz"))
        for name in input_files:
            self.put(name,(name+"/opaque fixture").encode())
        plan_sources = {"scripts/prepare_profiled_m3_inputs.py":self.sources["scripts/prepare_profiled_m3_inputs.py"]}
        plan = dict(metadata_sha256={name:registry.sha(self.root/name) for name in input_files},source_sha256_lf=plan_sources)
        self.put(registry.INPUT_PLAN,plan)
        plan_hash = registry.sha(self.root/registry.INPUT_PLAN)
        poses = {scene:{pose:{str(r):h(scene+"/"+pose) for r in range(3)} for pose in registry.POSES}
                 for scene in registry.SCENES}
        for role,name in registry.PREPARATION.items():
            value = dict(pass_all=True,plan_sha256=plan_hash,source_sha256_lf=plan_sources)
            value.update(repeat_matrix_sha256=poses) if role=="poses" else value.update(side=role)
            self.put(name,value)
        cases, rows = {},[]
        for scene,side,grid,pose in itertools.product(registry.SCENES,registry.SIDES,registry.GRIDS,registry.POSES):
            key = registry.case_id(scene,side,grid,pose)
            selected = [registry.INPUT_PLAN,base+"/poses/complete.json",base+"/poses/pose_packets.npz"]
            selected += ([f"results/mast3r_partner_shape/{side}.npz","results/lora_profiled_v1/splits.npz",
                          "results/lora_profiled_v1/inputs.json"] if grid=="centered" else
                         [base+f"/{side}/complete.json",base+f"/{side}/shifted_packet.npz",base+f"/{side}/shifted_splits.npz"])
            inputs = {name:registry.sha(self.root/name) for name in selected}
            cases[key] = dict(input_sha256=inputs,case_input_digest=h(key))
            rows.append(dict(scene=scene,side=side,grid=grid,pose_condition=pose,counts=[32]*5,input_digest=h(key),
                             input_sha256=inputs,original_m2_loader_compared=grid=="centered" and pose=="clean"))
        self.case_bridge = "results/fixture_case_bridge.json"
        self.put(self.case_bridge,dict(schema="profiled_m3_cpu_case_bridge_v1",pass_all=True,cases=rows,cases_checked=24,
            centered_clean_m2_cases_exact=6,endpoint_body_ast_exact=True,source_sha256_lf=plan_sources,
            device="cpu",gpu_used=False,prediction_arrays_opened=False,accuracy_scoring_performed=False,
            endpoint_gpu_execution_verified=False))
        proof = {}
        for key in NOMINATION_FIELDS:
            name = "results/synthetic_nomination/"+key+".opaque"
            self.put(name,("Must never parse this score/proof payload: "+key).encode())
            proof[key] = dict(file=name,sha256=registry.sha(self.root/name))
        nomination = dict(nominee=nominee,**{key:item["sha256"] for key,item in proof.items()})
        inputs = dict(plan_file=registry.INPUT_PLAN,attempt=registry.INPUT_ATTEMPT,plan_sha256=plan_hash,
            preparation_receipt_sha256={role:registry.sha(self.root/name) for role,name in registry.PREPARATION.items()},
            case_input_sha256={key:case["case_input_digest"] for key,case in cases.items()},
            pose_matrix_sha256={scene+"/"+pose:h(scene+"/"+pose) for scene,pose in itertools.product(registry.SCENES,registry.POSES)},
            runtime_sha256=registry.canonical_sha256(runtime),endpoint_contract_sha256=registry.canonical_sha256(endpoints),
            checkpoint_sha256=checkpoint)
        artifacts = {name:registry.sha(self.root/name) for name in
                     (registry.M2,registry.ENVIRONMENT,registry.BRIDGE,registry.BASELINE,registry.RGB,registry.INPUT_PLAN,
                      self.case_bridge,*registry.PREPARATION.values(),*rgb_names)}
        core = dict(source_sha256_lf=self.sources,artifact_sha256=artifacts,input_plan=registry.INPUT_PLAN,
            input_attempt=registry.INPUT_ATTEMPT,cases=cases,docker_image=registry.IMAGE,checkpoint_sha256=checkpoint,
            checkpoint_config_sha256=config,original_state_sha256=h("original"),runtime_identity=runtime)
        prerequisites = dict(nomination_proof=proof,case_bridge=dict(file=self.case_bridge,sha256=artifacts[self.case_bridge]),
                             engineering_acceptance=None)
        self.registry = registry.build_execution(nomination,inputs,core,prerequisites)
        self.write()

    def put(self,name,value):
        path = self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+"\n").encode())

    def write(self):
        self.put(self.registry_name,self.registry)

    def verify(self,weights=True):
        return registry.verify_execution(self.root,self.registry_name,weights=self.weights if weights else None)


class RegistryFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture=Fixture(Path(self.tmp.name))

    def test_complete_byte_check_never_authorizes_and_does_not_parse_proof_payloads(self):
        f=self.fixture
        report=f.verify()
        self.assertEqual(report["registry"],f.registry)
        self.assertEqual(report["states_checked"],360)
        self.assertEqual(report["cases_checked"],24)
        for key in ("launch_authorized","reuse_authorized","label_access_authorized","nomination_verified",
                    "engineering_acceptance_verified","release_prerequisites_verified","arrays_decoded","score_payloads_read"):
            self.assertFalse(report[key])
        self.assertTrue(report["weights_verified"])
        self.assertFalse(f.verify(weights=False)["weights_verified"])

    def test_pure_builder_does_not_mutate_or_write(self):
        before=deepcopy(self.fixture.registry)
        actual=registry.build_execution(before["inventory"]["nomination"],before["inventory"]["inputs"],
                                        before["model_core"],before["prerequisites"])
        self.assertEqual(actual,self.fixture.registry)
        self.assertEqual(before,self.fixture.registry)

    def test_truncated_reordered_or_changed_training_inventory_rejected(self):
        f=self.fixture
        for mutate in (lambda r:r["jobs"].pop(),lambda r:r["jobs"].reverse(),
                       lambda r:r["jobs"][1]["training"].update(macrosteps=7),
                       lambda r:r["inventory"]["stages"][0]["cells"].pop()):
            original=deepcopy(f.registry)
            mutate(f.registry);f.write()
            with self.assertRaises(ValueError):f.verify()
            f.registry=original;f.write()

    def test_actual_source_input_rgb_and_config_mutation_rejected(self):
        f=self.fixture
        names=["scripts/profiled_network.py","reports/dense_focal_pilot/inference/heads_base_rgb.npz",
               "results/mast3r_partner_shape/earlier.npz","fixture_weights/config.json","fixture_weights/model.safetensors"]
        for name in names:
            path=f.root/name;original=path.read_bytes();path.write_bytes(original+b"changed")
            with self.subTest(name=name),self.assertRaises(ValueError):f.verify()
            path.write_bytes(original)

    def test_missing_model_config_or_required_source_binding_rejected(self):
        f=self.fixture
        original=deepcopy(f.registry)
        for mutate in (lambda r:r["model_core"].pop("checkpoint_config_sha256"),
                       lambda r:r["model_core"]["source_sha256_lf"].pop(registry.SELF),
                       lambda r:r["model_core"]["artifact_sha256"].pop(registry.BASELINE),
                       lambda r:r["model_core"]["cases"].pop("heads_earlier_centered_clean")):
            mutate(f.registry);f.write()
            with self.assertRaises(ValueError):f.verify()
            f.registry=deepcopy(original);f.write()

    def test_case_tensor_identity_cannot_be_replaced_with_a_file_hash(self):
        f=self.fixture
        f.registry["model_core"]["cases"]["heads_earlier_centered_clean"]["case_input_digest"]=h("wrong")
        f.write()
        with self.assertRaisesRegex(ValueError,"CPU fitting receipt"):f.verify()

    def test_cpu_bridge_rejects_duplicate_case_even_when_receipt_is_rebound(self):
        f=self.fixture
        report=registry.read(f.root/f.case_bridge)
        report["cases"][-1]=deepcopy(report["cases"][0]);f.put(f.case_bridge,report)
        digest=registry.sha(f.root/f.case_bridge)
        f.registry["model_core"]["artifact_sha256"][f.case_bridge]=digest
        f.registry["prerequisites"]["case_bridge"]["sha256"]=digest
        f.write()
        with self.assertRaisesRegex(ValueError,"Duplicate"):f.verify()

    def test_nomination_hash_tamper_and_path_escape_rejected_without_scoring(self):
        f=self.fixture
        proof=f.registry["prerequisites"]["nomination_proof"]
        item=proof[NOMINATION_FIELDS[0]]
        original=item["file"]
        item["file"]="../outside";f.write()
        with self.assertRaises(ValueError):f.verify()
        item["file"]=original;item["sha256"]=h("wrong");f.write()
        with self.assertRaises(ValueError):f.verify()

    def test_acceptance_file_presence_cannot_release_execution(self):
        f=self.fixture
        name="results/synthetic_acceptance.json"
        f.put(name,dict(pass_all=True,launch_authorized=True,nominee="P3"))
        f.registry["prerequisites"]["engineering_acceptance"]=dict(file=name,sha256=registry.sha(f.root/name))
        f.write();report=f.verify()
        self.assertFalse(report["launch_authorized"])
        self.assertFalse(report["engineering_acceptance_verified"])

    def test_top_level_authorization_claim_is_rejected(self):
        f=self.fixture;f.registry["launch_authorized"]=True;f.write()
        with self.assertRaisesRegex(ValueError,"non-authorizing"):f.verify()

    def test_another_fixed_noise_draw_cannot_change_only_one_repeat(self):
        f=self.fixture;name=registry.PREPARATION["poses"]
        record=registry.read(f.root/name)
        record["repeat_matrix_sha256"]["heads"]["clean"]["2"]=h("another pose")
        f.put(name,record)
        with self.assertRaisesRegex(ValueError,"bytes changed"):f.verify()
        # Rebinding a changed receipt does not bypass the fixed-draw policy.
        value=registry.sha(f.root/name)
        inputs=deepcopy(f.registry["inventory"]["inputs"])
        inputs["preparation_receipt_sha256"]["poses"]=value
        core=deepcopy(f.registry["model_core"])
        core["artifact_sha256"][name]=value
        f.registry=registry.build_execution(f.registry["inventory"]["nomination"],inputs,core,f.registry["prerequisites"])
        f.write()
        with self.assertRaisesRegex(ValueError,"Pose draw differs"):f.verify()


if __name__=="__main__":
    unittest.main()
