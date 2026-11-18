"""Synthetic complete-record and opaque-byte tamper checks; no array decoding."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import profiled_m3_attempt_audit as audit
from profiled_artifacts import resolve_logical
from test_profiled_m3_registry import Fixture as RegistryFixture


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Fixture:
    def __init__(self, root, arm="P3", colon=False):
        self.root = Path(root)
        self.registration = RegistryFixture(self.root, "P4" if arm == "P4" else "P3")
        self.registry = self.registration.registry
        self.registry_name = self.registration.registry_name
        self.job = next(job for job in self.registry["jobs"] if job["arm"] == arm)
        self.attempt = "opaque_fixture01"
        self.raw_name = f"reports/lora_profiled_m3_v1/cases/{self.job['job_id']}/{self.attempt}"
        self.public_name = f"results/lora_profiled_m3_v1/cases/{self.job['job_id']}/{self.attempt}"
        self.raw, self.public = self.root / self.raw_name, self.root / self.public_name
        self.raw.mkdir(parents=True); self.public.mkdir(parents=True)
        core = self.registry["model_core"]
        case = core["cases"][self.job["case_id"]]
        rgb = audit._read(self.root / audit.RGB_MANIFEST)["inputs"][self.job["scene"] + "_base"]
        self.started = dict(schema="profiled_m3_model_attempt_v1", job=deepcopy(self.job), job_id=self.job["job_id"],
            attempt=self.attempt, registry_file=self.registry_name, registry_sha256=audit.sha(self.root / self.registry_name),
            **{key: deepcopy(core[key]) for key in ("source_sha256_lf", "artifact_sha256", "docker_image", "checkpoint_sha256",
                "checkpoint_config_sha256", "runtime_identity")}, **deepcopy(case),
            input_plan=core["input_plan"], input_attempt=core["input_attempt"], runtime_policy=deepcopy(core["runtime_identity"]["policy"]),
            seed=42, execution_commit="1" * 40, raw_attempt=self.raw_name, public_attempt=self.public_name,
            rental_compute_usd=0, disk_free_bytes_before=2**30, held_source_development_arrays_loaded=False,
            target_depth_arrays_loaded=False, accuracy_scoring_performed=False, stage_release_authorized=False,
            started_at_utc="2026-11-18T00:00:00+00:00", process=dict(pid=1, hostname="synthetic-container",
                process_run_uuid="1" * 32, model_initialization_uuid="2" * 32, model_constructions=0))
        self.put_public("started.json", self.started)
        for name in ("native", "final"):
            self.put_raw(f"endpoint/{name}.npz", ("opaque " + name + " bytes, deliberately not NPZ").encode())
        names = [f"{self.job['scene']}_base_{rgb['sha256'][:12]}_{i}" for i in range(5)]
        ids = [hashlib.md5(name.encode()).hexdigest() for name in names]
        for first, second in itertools.permutations(ids, 2):
            for name in (f"forward/{first}/{second}.pth", f"{audit.MATCH_FOLDER}/{first}-{second}.pth"):
                self.put_raw("endpoint_cache/" + name, ("opaque " + name).encode())
        canonical = []
        for index, identifier in enumerate(ids):
            suffix = "synthetic:1" if colon else "synthetic"
            name = f"endpoint_cache/canon_views/{identifier}_subsample=8_{suffix}.pth"
            self.put_raw(name, ("canonical " + str(index)).encode())
            canonical.append(dict(file=self.raw_name + "/" + name, sha256=audit.sha(resolve_logical(self.raw, name, must_exist=True))))
        edges = []
        for i, j in itertools.combinations(range(5), 2):
            first = f"endpoint_cache/{audit.MATCH_FOLDER}/{ids[i]}-{ids[j]}.pth"
            reverse = f"endpoint_cache/{audit.MATCH_FOLDER}/{ids[j]}-{ids[i]}.pth"
            edges.append(dict(i=i, j=j, count=12, pixel_pairs_sha256_int64_le=h(f"{i}/{j}"),
                file=self.raw_name + "/" + first, sha256=audit.sha(self.raw / first),
                reverse_file=self.raw_name + "/" + reverse, reverse_sha256=audit.sha(self.raw / reverse)))
        graph = dict(names=names, canonical_files=canonical, edges=edges, directed_forward_files=20, directed_match_files=20,
            cache_files=audit._inventory(self.raw / "endpoint_cache"), canonical_ratio_used_as_depth_or_confidence=False,
            fused_confidence_source="third tensor of ((canon, canon2, cconf), focal) in this fresh native canonical cache")
        self.put_public("endpoint/native_graph.json", graph)
        capture = dict(input_sha256=rgb["sha256"], output_sha256=audit.sha(self.raw / "endpoint/native.npz"),
            prior_cache_files=0, pairs=20, coarse_iterations=300, fine_iterations=300, native_size=[384,512],
            inference_precision="float32", device=core["runtime_identity"]["gpu"], seconds=4., peak_allocated_mib=9.,
            nonpositive_depth_pixels=[0] * 5)
        self.put_public("endpoint/native.json", capture)
        self.put_public("endpoint/native_trace.json", [dict(iteration=i, finite_gradient=True, received_gradient=1.,
            metric_scale=1., unit_m=1., depth=.1, rotation=.1, translation=.1, weighted=.003, intrinsic=.1, total_weighted=.004)
            for i in range(300)])
        final = dict(sample_counts=[32]*5, support_counts=[32]*5, gauge_ratio_count=160, sample_exclusions=0,
            nonpositive_fit_predictions=[0]*5, nonpositive_dense_pixels=[0]*5, g0=1., scales=[1.]*5,
            metric_scales=[1.]*5, positive_scales=True, graph_weight=.1, fit_is_accuracy_metric=False,
            gauge_estimator="pooled positive measured/native ratios; midpoint median",
            endpoint="native KDP followed by all-allowed-observation five-scale refinement",
            sensor_normalizers=[.02]*5, ridge=.01, singular_values=[1.]*5, data_rank=5, singular_ratio=1.,
            every_view_supported=True, graph_components=[[0,1,2,3,4]], graph_connected=True, graph_filtering=[],
            rejected_graph_confidences=0, native_origin=[0.,0.,0.], weighted_fit_before=1., weighted_fit_after=.5,
            ridge_penalty_after=0., penalized_fit_after=.5, normal_equation_max_residual=1e-12)
        self.put_public("endpoint/final.json", final)
        endpoint = dict(capture=capture, final=final, sample_counts=[32]*5, original_prior_rows_preserved=True,
            final_compute="CPU float64", graph_file=self.public_name + "/endpoint/native_graph.json",
            graph_sha256=audit.sha(self.public / "endpoint/native_graph.json"))
        for kind in ("native", "final"):
            endpoint[kind+"_file"] = self.raw_name + "/endpoint/" + kind + ".npz"
            endpoint[kind+"_sha256"] = audit.sha(self.raw / "endpoint" / (kind + ".npz"))
        trained = arm != "P0"
        steps = []
        if trained:
            self.put_raw("teacher_initial.npz", b"opaque synthetic teacher")
            self.put_raw("adapters.npz", b"opaque synthetic adapters")
            states = [dict(support_fold=fold, gains=[1.]*5, sensor_normalizers=[.02]*5, ridge=.01,
                           support_counts=[16]*5, query_counts=[16]*5, graph_filtering=[]) for fold in (0,1)]
            teacher = dict(decoder_calls=20, checkpoint_sha256=core["checkpoint_sha256"],
                initial_adapter_state_sha256=h("initial adapter"), fused_native_file=self.raw_name + "/teacher_initial.npz",
                fused_native_sha256=audit.sha(self.raw / "teacher_initial.npz"), full_pair_pointmaps_archived=False,
                graph_audit=audit._read(self.root / audit.BASELINE_GRAPH)["scenes"][self.job["scene"]], states=states,
                legacy_pair_gauges={f"{i}_{j}":1. for i,j in itertools.permutations(range(5),2)}, confidence_and_graph_frozen=True)
            self.put_public("teacher.json", teacher)
            for index in range(8):
                row = dict(step=index, optimizer_steps=1, optimizer_updates_cumulative=index+1,
                    graph_free_decoder_calls=20, differentiable_decoder_calls=20, support_fold=None if arm=="P1" else index%2,
                    loss_before_update=.1, scales_before_update=None if arm=="P1" else [1.]*5,
                    seconds=.1, peak_allocated_mib=10., peak_reserved_mib=15., adapter_update_l2=.001,
                    unclipped_gradient_norm=.5, gradient_clip_norm=1., adapter_state_sha256=h("step "+str(index)),
                    gradient=dict(norm=.5, parameters_with_gradient=288, parameters_with_nonzero_gradient=1,
                                  active_names=["decoder.attn.lora_B"], missing_names=[]))
                self.put_public(f"step_{index:02d}.json", row); steps.append(row)
        self.complete = deepcopy(self.started)
        self.complete["process"].update(model_constructions=1, model_constructed_at_utc="2026-11-18T00:00:01+00:00")
        self.complete.update(status="model_outputs_complete_without_scoring", pass_all=True, accuracy_improvement_established=False,
            completed_at_utc="2026-11-18T00:00:20+00:00", original_state_unchanged=True,
            original_state_sha256=core["original_state_sha256"], pre_endpoint_original_state_sha256=core["original_state_sha256"],
            final_original_state_sha256=core["original_state_sha256"], optimizer_updates=8 if trained else 0, steps=steps,
            teacher_decoder_calls=20 if trained else 0, training_graph_free_decoder_calls=160 if trained else 0,
            training_differentiable_decoder_calls=160 if trained else 0, endpoint_decoder_calls=20,
            adapter_module_count=144 if trained else 0, trainable_parameters=1032192 if trained else 0,
            adapter_initial_sha256=h("initial adapter") if trained else None,
            adapter_final_sha256=h("step 7") if trained else None, endpoint=endpoint,
            seconds=19., training_and_teacher_seconds=10. if trained else 1., endpoint_seconds=5.,
            preparation_peak_allocated_mib=5., preparation_peak_reserved_mib=6., endpoint_peak_reserved_mib=14.,
            peak_allocated_mib=10. if trained else 9., peak_reserved_mib=15. if trained else 14.)
        self.reseal()

    def put_raw(self, name, value):
        path = resolve_logical(self.raw, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def put_public(self, name, value):
        path = self.public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def reseal(self):
        self.complete["raw_files"] = audit._inventory(self.raw)
        self.complete["public_files"] = audit._inventory(self.public, exclude=("complete.json",))
        self.put_public("complete.json", self.complete)

    def verify(self):
        return audit.audit_m3_complete(self.root, self.registry_name, self.job["job_id"], self.attempt)


class AttemptAuditFixtures(unittest.TestCase):
    def test_all_arms_pass_opaque_byte_audit_without_decoding(self):
        for arm in ("P0","P1","P2","P3","P4","P5"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), arm)
                with patch("numpy.load", side_effect=AssertionError("No array decoding permitted")):
                    result = fixture.verify()
                self.assertTrue(result["pass_all"])
                self.assertFalse(result["arrays_deserialized"])
                self.assertFalse(result["actual_container_identity_verified"])
                self.assertFalse(result["stage_release_authorized"])
                expected = deepcopy(result); digest = expected.pop("receipt_sha256")
                self.assertEqual(digest, audit.canonical_sha256(expected))
                self.assertEqual(fixture.verify(), result)

    def test_logical_colon_cache_names_survive_windows_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(Fixture(Path(tmp), "P0", colon=True).verify()["pass_all"])

    def test_missing_complete_and_conflicting_failure_rejected(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), "P0")
                if failure:
                    fixture.put_public("failure.json", {"error": "synthetic"})
                else:
                    (fixture.public / "complete.json").unlink()
                with self.assertRaises(ValueError): fixture.verify()

    def test_changed_or_extra_bytes_rejected_even_with_success_flag(self):
        for action in ("native", "teacher", "step", "extra", "source", "rgb"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp))
                if action == "native": fixture.put_raw("endpoint/native.npz", b"changed")
                elif action == "teacher": fixture.put_raw("teacher_initial.npz", b"changed")
                elif action == "step": fixture.put_public("step_00.json", {"pass_all": True})
                elif action == "extra": fixture.put_raw("unregistered.npz", b"extra")
                elif action == "source": (fixture.root / "scripts/profiled_network.py").write_bytes(b"changed source")
                else: (fixture.root / "reports/dense_focal_pilot/inference/heads_base_rgb.npz").write_bytes(b"changed RGB")
                with self.assertRaises(ValueError): fixture.verify()

    def test_metadata_tampering_rejected_after_rehash(self):
        mutations = {
            "wrong_job": lambda f: f.complete["job"].update(arm="P0"),
            "config": lambda f: f.complete.update(checkpoint_config_sha256="0"*64),
            "frozen": lambda f: f.complete.update(final_original_state_sha256="0"*64),
            "adapter": lambda f: f.complete.update(adapter_final_sha256="0"*64),
            "calls": lambda f: f.complete.update(training_differentiable_decoder_calls=159),
            "endpoint_calls": lambda f: f.complete.update(endpoint_decoder_calls=19),
            "model_count": lambda f: f.complete["process"].update(model_constructions=2),
            "uuid": lambda f: f.complete["process"].update(model_initialization_uuid="0"*32),
            "time": lambda f: f.complete.update(completed_at_utc="2026-11-17T00:00:00Z"),
            "scoring": lambda f: f.complete.update(accuracy_scoring_performed=True),
            "endpoint_path": lambda f: f.complete["endpoint"].update(native_file="reports/other/native.npz"),
            "phase_time": lambda f: f.complete.update(seconds=1.),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp)); mutate(fixture); fixture.reseal()
                with self.assertRaises(ValueError): fixture.verify()

    def test_step_teacher_graph_hook_coverage_cannot_be_forged_with_updated_hashes(self):
        for action in ("fold", "zero_update", "hook", "support", "graph", "extra_resealed"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp))
                if action in ("fold", "zero_update"):
                    row = fixture.complete["steps"][0]
                    row.update(support_fold=1) if action=="fold" else row.update(adapter_update_l2=0.)
                    fixture.put_public("step_00.json", row)
                elif action == "hook":
                    trace = audit._read(fixture.public / "endpoint/native_trace.json")
                    trace[10]["received_gradient"] = 0.
                    fixture.put_public("endpoint/native_trace.json", trace)
                elif action == "support":
                    teacher = audit._read(fixture.public / "teacher.json")
                    teacher["states"][0]["support_counts"][0] -= 1
                    fixture.put_public("teacher.json", teacher)
                elif action == "graph":
                    graph = audit._read(fixture.public / "endpoint/native_graph.json")
                    graph["edges"][0]["reverse_file"] = graph["edges"][0]["file"]
                    fixture.put_public("endpoint/native_graph.json", graph)
                    fixture.complete["endpoint"]["graph_sha256"] = audit.sha(fixture.public / "endpoint/native_graph.json")
                else: fixture.put_raw("unexpected.pth", b"opaque extra")
                fixture.reseal()
                with self.assertRaises(ValueError): fixture.verify()

    def test_nonfinite_and_duplicate_json_keys_fail(self):
        for text in ('{"pass_all":true,"pass_all":false}', '{"seconds":NaN}'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), "P0")
                (fixture.public / "complete.json").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError): fixture.verify()

    def test_internally_consistent_wrong_fitting_population_rejected(self):
        for arm in ("P0", "P3"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                fixture = Fixture(Path(tmp), arm)
                endpoint = fixture.complete["endpoint"]
                endpoint["sample_counts"] = [30] * 5
                endpoint["final"].update(sample_counts=[30] * 5, support_counts=[30] * 5,
                                         gauge_ratio_count=150)
                fixture.put_public("endpoint/final.json", endpoint["final"])
                if arm != "P0":
                    teacher = audit._read(fixture.public / "teacher.json")
                    for state in teacher["states"]:
                        state.update(support_counts=[15] * 5, query_counts=[15] * 5)
                    fixture.put_public("teacher.json", teacher)
                fixture.reseal()
                with self.assertRaisesRegex(ValueError, "registered CPU case bridge"):
                    fixture.verify()

    def test_archived_binding_does_not_require_live_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp), "P0")
            (fixture.registration.weights / "config.json").unlink()
            (fixture.registration.weights / "model.safetensors").unlink()
            result = fixture.verify()
            self.assertFalse(result["live_weight_files_verified"])
            self.assertEqual(result["checkpoint_config_sha256"], fixture.registry["model_core"]["checkpoint_config_sha256"])


if __name__ == "__main__":
    unittest.main()
