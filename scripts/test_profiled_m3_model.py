"""CPU-only orchestration fixtures using explicit tiny model/optimizer proxies.

These do not reconstruct, exercise MASt3R or establish numerical/model fidelity.
Proxy numel/module inventories stand in for the real registered adapter shape;
actual placement remains an independently required model acceptance check.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import profiled_m3_model as core
from register_profiled_execution import SOURCES
from test_profiled_m3_inventory import synthetic_plan


ROOT = Path(__file__).resolve().parents[1]
PAIRS = tuple(itertools.permutations(range(5), 2))


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def state_digest(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


class ParameterProxy:
    def __init__(self, value):
        self.value = torch.tensor([value], dtype=torch.float32)
        self.requires_grad = True
        self.grad = None

    def numel(self):
        return 516096  # Explicit inventory proxy, not an actual neural model.

    def detach(self):
        return self.value.detach()


class ModelProxy:
    def __init__(self, events):
        self.events = events
        self.weight = torch.tensor([.25], dtype=torch.float32)
        self.buffer = torch.tensor([.75], dtype=torch.float32)
        self.adapters = []

    def _decoder(self, phase):
        self.events.append(("decoder", phase))

    def state_dict(self):
        return {"weight": self.weight, "buffer": self.buffer}

    def parameters(self):
        return [self.weight] + [p for _, p in self.adapters]

    def buffers(self):
        return [self.buffer]

    def named_parameters(self):
        return [("weight", self.weight)] + self.adapters

    def eval(self):
        return self


class Fixture:
    def __init__(self, root, arm="P3", **faults):
        self.root, self.events, self.faults = root, [], faults
        plan = synthetic_plan("P4" if arm == "P4" else "P3")
        self.job = deepcopy(next(job for job in plan["jobs"] if job["arm"] == arm))
        self.job["checkpoint_sha256"] = hash_text("synthetic weights")
        self.model = None
        self.weights = root / "weights"
        self.weights.mkdir()
        (self.weights / "model.safetensors").write_text("synthetic weights", encoding="utf-8")
        (self.weights / "config.json").write_text('{"synthetic":true}', encoding="utf-8")
        self.config_hash = core.sha(self.weights / "config.json")
        required = set(SOURCES) | {core.CORE, "scripts/run_profiled_lora.py", "scripts/profiled_m3_network.py",
            "scripts/prepare_profiled_m3_inputs.py", "scripts/profiled_artifacts.py", "scripts/profiled_m3_gate.py"}
        self.sources = {}
        for name in sorted(required):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes() if name == core.CORE else ("mock source " + name).encode())
            self.sources[name] = core.sha(path, text=True)
        self.input_map = {"results/lora_profiled_m3_v1/synthetic_input.json": hash_text("synthetic fitting input")}
        for name in self.input_map:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fitting input", encoding="utf-8")
        self.case = dict(scene=self.job["scene"], side=self.job["side"], grid=self.job["grid"],
                         pose_condition=self.job["pose"], input_sha256=deepcopy(self.input_map), marker=0)
        self.case["prepared_input_digest"] = self.input_digest(self.case)
        self.runtime = dict(torch="synthetic-cpu-proxy", cuda="synthetic", gpu="synthetic-cpu-proxy",
                            policy=deepcopy(core.EXPECTED_RUNTIME))
        self.original_hash = state_digest(ModelProxy([]).state_dict().items())
        rgb_file = self.job["scene"] + "_base_rgb.npz"
        self.rgb_name = "reports/dense_focal_pilot/inference/" + rgb_file
        (root / self.rgb_name).parent.mkdir(parents=True, exist_ok=True)
        (root / self.rgb_name).write_bytes(b"synthetic RGB byte fixture, not a decoded array")
        rgb_hash = core.sha(root / self.rgb_name)
        artifacts = {
            core.INPUT_PLAN: {"synthetic_fixture": True},
            core.BRIDGE: {"original_state_hash": self.original_hash},
            core.ENVIRONMENT: dict(torch=self.runtime["torch"], cuda=self.runtime["cuda"], gpu=self.runtime["gpu"],
                                   docker_image="synthetic-cpu-fixture", checkpoint_sha256=self.job["checkpoint_sha256"]),
            core.RGB_MANIFEST: {"inputs": {self.job["scene"] + "_base": dict(file=rgb_file, sha256=rgb_hash,
                rgb_sha256=hash_text("synthetic RGB tensor"), batch="base",
                scene={"heads":"heads/seq-01", "chess":"chess/seq-03", "fire":"fire/seq-03"}[self.job["scene"]])}}}
        self.artifacts = {}
        for name, value in artifacts.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            self.artifacts[name] = core.sha(path)
        self.artifacts[self.rgb_name] = rgb_hash
        self.registry_name = "manifests/synthetic_m3_model_registry.json"
        self.registry = dict(schema="profiled_m3_execution_v1", jobs=[self.job], model_core=dict(
            source_sha256_lf=self.sources, artifact_sha256=self.artifacts,
            input_plan=core.INPUT_PLAN, input_attempt=core.INPUT_ATTEMPT,
            docker_image="synthetic-cpu-fixture", checkpoint_sha256=self.job["checkpoint_sha256"],
            checkpoint_config_sha256=self.config_hash,
            original_state_sha256=self.original_hash, runtime_identity=self.runtime,
            cases={self.job["case_id"]: dict(input_sha256=self.input_map,
                                            case_input_digest=self.case["prepared_input_digest"])}))
        self.commit = "1" * 40
        self.context = self.context_from_registry()

    def context_from_registry(self):
        path = self.root / self.registry_name
        path.write_text(json.dumps(self.registry), encoding="utf-8")
        return core.ModelContext(job=deepcopy(self.job), attempt="cpu_mock01", registry_file=self.registry_name,
            registry_sha256=core.sha(path), source_sha256_lf=deepcopy(self.sources), artifact_sha256=deepcopy(self.artifacts),
            input_sha256=deepcopy(self.input_map), case_input_digest=self.case["prepared_input_digest"],
            docker_image="synthetic-cpu-fixture", checkpoint_sha256=self.job["checkpoint_sha256"],
            checkpoint_config_sha256=self.config_hash,
            original_state_sha256=self.original_hash, weights=self.weights, execution_commit=self.commit,
            runtime_identity=deepcopy(self.runtime))

    @staticmethod
    def input_digest(case):
        return hash_text(json.dumps({key: case[key] for key in ("scene", "side", "grid", "pose_condition", "marker")}, sort_keys=True))

    def backend(self):
        fixture = self

        class AdamProxy:
            def __init__(self, parameters, lr):
                fixture.events.append(("optimizer_created", lr))
                self.parameters, self.lr = parameters, lr

            def zero_grad(self, set_to_none):
                fixture.events.append(("zero", set_to_none))
                for parameter in self.parameters:
                    parameter.grad = None

            def step(self):
                fixture.events.append(("optimizer_step",))
                for parameter in self.parameters:
                    parameter.value.sub_(self.lr * parameter.grad)

        def clip(parameters, maximum, error_if_nonfinite):
            self.events.append(("clip", maximum))
            norms = torch.stack([parameter.grad.double().square().sum() for parameter in parameters])
            norm = norms.sum().sqrt()
            if error_if_nonfinite and not torch.isfinite(norm):
                raise FloatingPointError("nonfinite fixture gradient")
            for parameter in parameters:
                parameter.grad.mul_(min(1., maximum / (float(norm) + 1e-6)))
            return norm

        cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: self.runtime["gpu"],
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0, synchronize=lambda: None, empty_cache=lambda: None)
        torch_proxy = SimpleNamespace(__version__=self.runtime["torch"], version=SimpleNamespace(cuda=self.runtime["cuda"]),
            cuda=cuda, manual_seed=lambda seed: self.events.append(("seed", seed)),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)),
            isfinite=lambda value: torch.isfinite(value.value if isinstance(value, ParameterProxy) else value),
            optim=SimpleNamespace(Adam=AdamProxy), nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip)))

        def load_case(plan, attempt, scene, side, grid, pose, device):
            self.events.append(("case", scene, side, grid, pose, device))
            result = deepcopy(self.case)
            if self.faults.get("input_digest"):
                result["marker"] = 1
            return result

        def model_factory(weights):
            self.events.append(("model", str(weights)))
            self.model = ModelProxy(self.events)
            if self.faults.get("original_state"):
                self.model.weight.add_(1.)
            return self.model

        def install(model, path):
            self.events.append(("install",))
            model.adapters = [("decoder.lora_A", ParameterProxy(.1)), ("decoder.lora_B", ParameterProxy(0.))]
            if self.faults.get("nonzero_B"):
                model.adapters[1][1].value.add_(.1)
            if self.faults.get("original_trainable"):
                model.weight.requires_grad_(True)
            return ["module" + str(i) for i in range(144)]

        def collect_points(model, features):
            phase = "collect" if model.adapters else "teacher"
            count = 19 if self.faults.get("short_" + phase) else 20
            for _ in range(count):
                model._decoder(phase)
            return ({pair: (torch.ones(1), torch.ones(1)) for pair in PAIRS}, {pair: torch.ones(1) for pair in PAIRS})

        def prepare_training(case, initial, confidence):
            self.events.append(("prepare", case["pose_condition"]))
            state = dict(gains=torch.ones(5), sensor_normalizers=torch.ones(5), ridge=.01,
                         support=[dict(xy=torch.zeros((8, 2))) for _ in range(5)], graph_filtering={})
            return dict(graph_audit={"synthetic": True}, states=[state, deepcopy(state)],
                        queries=[[dict(xy=torch.zeros((8, 2))) for _ in range(5)] for _ in range(2)],
                        gauges={pair: torch.tensor(1.) for pair in PAIRS})

        def output_adjoints(points, prepared, arm, step):
            self.events.append(("adjoints", arm, step))
            return torch.tensor(1.), None if arm == "P1" else torch.ones(5), {pair: None for pair in PAIRS}

        def replay_backward(model, features, adjoints):
            count = 19 if self.faults.get("short_replay") else 20
            for _ in range(count):
                model._decoder("replay")
            for _, parameter in model.adapters:
                parameter.grad = torch.ones_like(parameter.value)
                if self.faults.get("zero_gradient"):
                    parameter.grad.zero_()
            if self.faults.get("buffer_mutation"):
                model.buffer.add_(1.)

        def capture_endpoint(model, case, cache, raw, public):
            self.events.append(("endpoint", case["pose_condition"]))
            for path in (cache, raw, public):
                path.mkdir(parents=True, exist_ok=False)
            for _ in range(19 if self.faults.get("short_endpoint") else 20):
                model._decoder("endpoint")
            native, final = raw / "native.npz", raw / "final.npz"
            for path in (native, final):
                with path.open("xb") as stream:
                    np.savez(stream, synthetic=np.array([1.]))
            if self.faults.get("adapter_mutation") and model.adapters:
                model.adapters[0][1].value.add_(1.)
            if self.faults.get("input_mutation"):
                case["marker"] += 1
            if self.faults.get("source_mutation"):
                (self.root / "scripts/profiled_network.py").write_text("changed", encoding="utf-8")
            if self.faults.get("config_mutation"):
                (self.weights / "config.json").write_text('{"synthetic":"changed"}', encoding="utf-8")
            if self.faults.get("rgb_mutation"):
                (self.root / self.rgb_name).write_bytes(b"changed synthetic RGB")
            if self.faults.get("object_replacement"):
                model.weight = model.weight.clone()
            evidence = dict(capture={"peak_allocated_mib": 0.}, native_file=native.relative_to(self.root).as_posix(),
                native_sha256=core.sha(native), final_file=final.relative_to(self.root).as_posix(), final_sha256=core.sha(final))
            return {}, {}, {}, {}, evidence

        return SimpleNamespace(np=np, torch=torch_proxy, configure_runtime=lambda: deepcopy(core.EXPECTED_RUNTIME),
            verify_runtime=lambda: deepcopy(core.EXPECTED_RUNTIME), git_commit=lambda root: self.commit,
            load_case=load_case, input_digest=self.input_digest, model_factory=model_factory,
            digest=state_digest, install=install, PAIRS=PAIRS, collect_points=collect_points,
            views_for=lambda key: (None, None, ["views"]), training_features=lambda model, views: ["features"],
            to_device=lambda value, device: value, prepare_training=prepare_training,
            fuse=lambda points, confidence: (torch.ones((5, 1, 1)), torch.ones((5, 1, 1))),
            output_adjoints=output_adjoints, replay_backward=replay_backward, capture_endpoint=capture_endpoint)

    def run(self):
        with redirect_stdout(io.StringIO()):
            return core._run(self.context, self.backend(), self.root)


class ModelCoreFixtures(unittest.TestCase):
    def test_macrostep_arithmetic_ast_is_original(self):
        old = ast.parse((ROOT / "scripts/run_profiled_lora.py").read_text(encoding="utf-8"))
        new = ast.parse((ROOT / core.CORE).read_text(encoding="utf-8"))
        def loop(tree):
            return next(node for node in ast.walk(tree) if isinstance(node, ast.For) and
                        isinstance(node.target, ast.Name) and node.target.id == "step")
        previous, current = deepcopy(loop(old)), deepcopy(loop(new))
        # The sole allowed reporting rename; mathematical statements unchanged.
        for node in ast.walk(previous):
            if isinstance(node, ast.Constant) and node.value == "run_id":
                node.value = "job_id"
        self.assertEqual(ast.dump(previous, include_attributes=False), ast.dump(current, include_attributes=False))
        # Include teacher, frozen-confidence preparation, adapter installation,
        # optimizer creation and saved adapter outputs, not only the loop.
        old_adaptation = next(node for node in ast.walk(old) if isinstance(node, ast.If) and
                              isinstance(node.test, ast.Subscript) and "adapt_lora" in ast.dump(node.test))
        new_adaptation = next(node for node in ast.walk(new) if isinstance(node, ast.If) and
                              isinstance(node.test, ast.Subscript) and "lora_enabled" in ast.dump(node.test))
        old_body, new_body = deepcopy(old_adaptation.body), deepcopy(new_adaptation.body)
        for statement in old_body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Constant) and node.value == "run_id":
                    node.value = "job_id"
        self.assertEqual([ast.dump(node, include_attributes=False) for node in old_body],
                         [ast.dump(node, include_attributes=False) for node in new_body])
        for name in ("gradient_summary", "to_device"):
            first = next(node for node in ast.walk(old) if isinstance(node, ast.FunctionDef) and node.name == name)
            second = next(node for node in ast.walk(new) if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(first, include_attributes=False), ast.dump(second, include_attributes=False))

    def test_import_has_no_heavy_model_import_or_cli(self):
        tree = ast.parse((ROOT / core.CORE).read_text(encoding="utf-8"))
        imports = [ast.dump(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("mast3r" in value or "torch" in value for value in imports))
        self.assertFalse(any(isinstance(node, ast.If) for node in tree.body))

    def test_each_arm_keeps_exact_teacher_collect_replay_update_order(self):
        for arm in ("P1", "P2", "P3", "P4", "P5"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), arm)
                report = fixture.run()
                self.assertEqual(report["optimizer_updates"], 8)
                self.assertEqual(report["teacher_decoder_calls"], 20)
                self.assertEqual(report["training_graph_free_decoder_calls"], 160)
                self.assertEqual(report["training_differentiable_decoder_calls"], 160)
                self.assertEqual(report["endpoint_decoder_calls"], 20)
                actions = [event[0] for event in fixture.events]
                self.assertLess(actions.index("decoder"), actions.index("install"))
                self.assertEqual(actions.count("model"), 1)
                self.assertEqual(actions.count("optimizer_step"), 8)
                for index, event in enumerate(fixture.events):
                    if event[0] == "optimizer_step":
                        self.assertEqual(fixture.events[index - 1][0], "clip")
                        self.assertEqual([item for item in fixture.events[index - 21:index - 1]], [("decoder", "replay")] * 20)
                self.assertEqual([row["support_fold"] for row in report["steps"]],
                                 [None] * 8 if arm == "P1" else [0, 1] * 4)
                self.assertTrue(report["original_state_unchanged"])
                self.assertNotEqual(report["adapter_initial_sha256"], report["adapter_final_sha256"])
                self.assertFalse(report["stage_release_authorized"])
                self.assertEqual(report["process"]["model_constructions"], 1)

    def test_frozen_p0_has_no_teacher_adapter_or_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp), "P0")
            report = fixture.run()
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(report["teacher_decoder_calls"], 0)
            self.assertEqual(report["endpoint_decoder_calls"], 20)
            self.assertEqual(report["trainable_parameters"], 0)
            self.assertIsNone(report["adapter_final_sha256"])
            self.assertFalse(any(event[0] in ("install", "optimizer_step", "prepare", "adjoints") for event in fixture.events))

    def test_failures_preserve_attempt_and_do_not_emit_completion(self):
        for fault in ("input_digest", "original_state", "nonzero_B", "original_trainable", "short_teacher",
                      "short_collect", "short_replay", "zero_gradient", "buffer_mutation", "short_endpoint",
                      "adapter_mutation", "input_mutation", "source_mutation", "object_replacement",
                      "config_mutation", "rgb_mutation"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), **{fault: True})
                with self.assertRaises((ValueError, FloatingPointError)):
                    fixture.run()
                public = Path(tmp) / "results/lora_profiled_m3_v1/cases" / fixture.job["job_id"] / fixture.context.attempt
                self.assertTrue((public / "started.json").is_file())
                failure = json.loads((public / "failure.json").read_text(encoding="utf-8"))
                self.assertFalse(failure["pass_all"])
                self.assertFalse(failure["stage_release_authorized"])
                self.assertFalse((public / "complete.json").exists())

    def test_existing_attempt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp), "P0")
            report = fixture.run()
            with self.assertRaises(FileExistsError):
                fixture.run()
            complete = Path(tmp) / report["public_attempt"] / "complete.json"
            self.assertEqual(json.loads(complete.read_text(encoding="utf-8")), report)

    def test_bound_bytes_and_registry_membership_precede_model_creation(self):
        for fault in ("source", "input", "checkpoint", "config", "rgb", "registry", "job", "path"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), "P0")
                if fault == "source":
                    (Path(tmp) / "scripts/profiled_network.py").write_text("changed", encoding="utf-8")
                elif fault == "input":
                    (Path(tmp) / next(iter(fixture.input_map))).write_text("changed", encoding="utf-8")
                elif fault == "checkpoint":
                    (fixture.weights / "model.safetensors").write_text("changed", encoding="utf-8")
                elif fault == "config":
                    (fixture.weights / "config.json").write_text("changed", encoding="utf-8")
                elif fault == "rgb":
                    (Path(tmp) / fixture.rgb_name).write_bytes(b"changed")
                elif fault == "registry":
                    (Path(tmp) / fixture.registry_name).write_text("changed", encoding="utf-8")
                elif fault == "job":
                    changed = deepcopy(fixture.job); changed["stage_id"] = "invented"
                    fixture.context = replace(fixture.context, job=changed)
                else:
                    fixture.context = replace(fixture.context, attempt="../escape")
                with self.assertRaises(ValueError):
                    fixture.run()
                self.assertFalse(any(event[0] == "model" for event in fixture.events))

    def test_p0_rechecks_rgb_and_config_after_endpoint(self):
        for fault in ("rgb_mutation", "config_mutation"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), "P0", **{fault: True})
                with self.assertRaisesRegex(ValueError, "(Registered bytes changed|configuration bytes differ)"):
                    fixture.run()
                public = Path(tmp) / "results/lora_profiled_m3_v1/cases" / fixture.job["job_id"] / fixture.context.attempt
                self.assertTrue((public / "failure.json").exists())
                self.assertFalse((public / "complete.json").exists())

    def test_public_execute_has_no_dependency_argument_and_is_once_per_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp), "P0")
            with patch.object(core, "ROOT", Path(tmp)), patch.object(core, "_EXECUTION_ATTEMPTED", False), \
                 patch.object(core, "_production_backend", return_value=fixture.backend()):
                core.execute(fixture.context)
                with self.assertRaisesRegex(ValueError, "One model attempt"):
                    core.execute(replace(fixture.context, attempt="second"))
            with self.assertRaises(TypeError):
                core.execute(fixture.context, backend=fixture.backend())


if __name__ == "__main__":
    unittest.main()
