"""Independent compact-replay fixtures; generated arrays and opaque provenance."""
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

import replay_profiled_public_scores as replay


ROOT=Path(__file__).resolve().parents[1]


def synthetic():
    samples={};rows=[];runtime={};failures={}
    errors={"P0":1.2,"P1":1.,"P2":.95,"P3":.8,"P4":.8,"P5":.9}
    for scene,side,frame in itertools.product(replay.SCENES,replay.SIDES,replay.FRAMES):
        n=1+replay.FRAMES.index(frame);prefix=f"{scene}_{side}_{frame}__"
        samples[prefix+"xy"]=np.column_stack((np.arange(n),np.zeros(n,dtype=int))).astype(np.int64)
        samples[prefix+"measured_m"]=np.full(n,2.,dtype=np.float64)
        for arm in replay.ARMS:
            row=dict(scene=scene,side=side,arm=arm,frame=frame,mapped_opposite_count=n+1,
                     overlapping_excluded=1,held_count=n,adaptation_count=8)
            for endpoint in replay.ENDPOINTS:
                values=np.full(n,2.+errors[arm],dtype=np.float64)
                samples[prefix+arm+"_"+endpoint+"_m"]=values
                row[endpoint]=replay.metric(values,samples[prefix+"measured_m"])
            rows.append(row);runtime[f"{scene}_{side}_{arm}"]=10.;failures[f"{scene}_{side}_{arm}"]=0
    report=dict(schema="profiled_held_source_scores_v1",rows=rows,runtimes_seconds=runtime,
                numerical_failures_by_run_id=failures,all_failed_attempt_counts=failures.copy())
    refresh(report,samples)
    return report,samples


def refresh(report,samples):
    for row in report["rows"]:
        prefix=f"{row['scene']}_{row['side']}_{row['frame']}__"
        for endpoint in replay.ENDPOINTS:
            row[endpoint]=replay.metric(samples[prefix+row['arm']+"_"+endpoint+"_m"],samples[prefix+"measured_m"])
    report.update(replay.independent_summary(report["rows"],report["runtimes_seconds"],report["numerical_failures_by_run_id"]))


class Files:
    def __init__(self,root):
        self.root=root;self.report,self.samples=synthetic()
        self.report_name="results/report.json";self.samples_name="results/samples.npz";self.out="results/replay.json"
        for name in replay.SOURCES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        rules=json.loads((ROOT/"manifests/lora_profiled_evaluation_v2.json").read_text(encoding="utf-8"))
        rules_hash=self.put("manifests/rules.json",rules)
        registry_hash=self.put("manifests/registry.json",dict(phase="scientific",scientific_ready_at_registration=True,
                       artifact_sha256={"manifests/rules.json":rules_hash}))
        self.evaluation=dict(schema="profiled_evaluation_execution_v1",synthetic_fixtures_pass=True,
            study_arrays_loaded=False,study_scores_read=False,prepared_at_utc="2026-11-18T01:00:00+00:00",
            rules="manifests/rules.json",rules_sha256=rules_hash,failure_policy=replay.FAILURE_POLICY,
            input_sha256={"results/lora_profiled_v1/splits.npz":"a"*64},
            packet_sha256={f"results/mast3r_partner_shape/{side}.npz":str(i+1)*64 for i,side in enumerate(replay.SIDES)})
        identity=dict(file="manifests/evaluation.json",sha256=self.put("manifests/evaluation.json",self.evaluation))
        cases=[]
        for scene,side,arm in itertools.product(replay.SCENES,replay.SIDES,replay.ARMS):
            key=f"{scene}_{side}_{arm}"
            cases.append(dict(run_id=key,scene=scene,adaptation_source=side,arm=arm,
                selected=dict(seconds=10.),unclassified_failure_count=0,numerical_failure_count=0,failed_attempt_count=0))
        self.seal=dict(schema="profiled_global_prediction_seal_v1",pass_all=True,required_states=36,completed_states=36,
            cases=cases,evaluation_execution=identity,evaluation_rules="manifests/rules.json",evaluation_rules_sha256=rules_hash,
            common_scientific_registry="manifests/registry.json",common_scientific_registry_sha256=registry_hash,
            failure_policy=replay.FAILURE_POLICY,failure_review_required=False)
        seal_hash=self.put("results/seal.json",self.seal)
        self.access=dict(evaluation_execution=identity,prediction_seal="results/seal.json",prediction_seal_sha256=seal_hash,
                         first_access_at_utc="2026-11-18T02:00:00+00:00")
        access_hash=self.put("results/access.json",self.access)
        self.report.update(prediction_seal="results/seal.json",prediction_seal_sha256=seal_hash,evaluation_execution=identity,
            score_access_receipt="results/access.json",score_access_receipt_sha256=access_hash,evaluation_rules_sha256=rules_hash,
            failure_policy=replay.FAILURE_POLICY,failure_review_required=False,candidate_dependent_masks=False,
            target_depth_labels_loaded=False,finite_error_means_are_diagnostics_only=True,split_sha256="a"*64,
            source_packet_sha256={side:str(i+1)*64 for i,side in enumerate(replay.SIDES)},replay_samples=self.samples_name)
        self.save()

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")
        return replay.sha(path.read_bytes())

    def save(self):
        with (self.root/self.samples_name).open("wb") as stream:np.savez_compressed(stream,**self.samples)
        self.report["replay_samples_sha256"]=replay.sha((self.root/self.samples_name).read_bytes())
        self.put(self.report_name,self.report)

    def run(self):return replay.replay(self.root,self.report_name,self.out)


class ArithmeticTests(unittest.TestCase):
    def setUp(self):self.report,self.samples=synthetic()

    def test_complete_inventory_and_analytical_mean(self):
        summary=replay.validate_arithmetic(self.report,self.samples)
        self.assertEqual(len(self.samples),420);self.assertEqual(len(self.report["rows"]),180)
        self.assertEqual(summary["nominee"],"P3")
        self.assertAlmostEqual(summary["candidates"]["P3"]["directional_gain"]["earlier"],.2)
        self.assertFalse(summary["rental_gate_passed"])

    def test_unequal_counts_do_not_weight_views_or_scenes(self):
        for scene,side,frame in itertools.product(replay.SCENES,replay.SIDES,replay.FRAMES):
            prefix=f"{scene}_{side}_{frame}__";n=len(self.samples[prefix+"measured_m"])
            error=float(replay.FRAMES.index(frame)+1+10*replay.SCENES.index(scene))
            self.samples[prefix+"P0_final_m"]=np.full(n,2.+error)
        refresh(self.report,self.samples)
        result=replay.validate_arithmetic(self.report,self.samples)
        self.assertEqual(result["case_mae_m"]["final"]["heads_earlier_P0"],3.)
        self.assertEqual(result["directional_equal_scene_mae_m"]["final"]["earlier"]["P0"],13.)
        self.report["case_mae_m"]["final"]["heads_earlier_P0"]=55/15
        with self.assertRaisesRegex(ValueError,"Arithmetic mismatch"):replay.validate_arithmetic(self.report,self.samples)

    def test_missing_duplicate_rows_and_missing_extra_fields(self):
        for fault in ("missing_row","duplicate_row","missing_field","extra_field"):
            with self.subTest(fault=fault):
                report,samples=synthetic()
                if fault=="missing_row":report["rows"].pop()
                elif fault=="duplicate_row":report["rows"][-1]=deepcopy(report["rows"][0])
                elif fault=="missing_field":samples.pop(next(iter(samples)))
                else:samples["extra"]=np.zeros(1)
                with self.assertRaises(ValueError):replay.validate_arithmetic(report,samples)

    def test_invalid_predictions_retained_and_disqualify_primary(self):
        for value in (float("nan"),float("inf"),0.,-1.):
            with self.subTest(value=value):
                report,samples=synthetic();samples["heads_earlier_25__P3_final_m"][0]=value;refresh(report,samples)
                result=replay.validate_arithmetic(report,samples)
                self.assertFalse(result["candidates"]["P3"]["eligible"])
                self.assertEqual(result["nominee"],"P4")
                row=next(row for row in report["rows"] if (row['scene'],row['side'],row['arm'],row['frame'])==('heads','earlier','P3',25))
                self.assertEqual(row['final']['count'],1);self.assertEqual(row['final']['candidate_dependent_exclusions'],0)

    def test_secondary_invalidity_does_not_change_primary_nomination(self):
        self.samples["heads_earlier_25__P3_native_m"][0]=np.nan;refresh(self.report,self.samples)
        self.assertEqual(replay.validate_arithmetic(self.report,self.samples)["nominee"],"P3")

    def test_overflow_errors_remain_invalid(self):
        value=replay.metric(np.array([-1e308]),np.array([1e308]))
        self.assertIsNone(value['mae_m']);self.assertEqual(value['unrepresentable_error_count'],1)

    def test_fixed_pixels_and_mask_accounting_reject_tamper(self):
        for fault in ('xy','truth','count','candidate_count','prediction_count'):
            with self.subTest(fault=fault):
                report,samples=synthetic();prefix='heads_earlier_225__'
                if fault=='xy':samples[prefix+'xy'][1]=samples[prefix+'xy'][0]
                elif fault=='truth':samples[prefix+'measured_m'][0]=0.
                elif fault=='count':report['rows'][0]['held_count']+=1
                elif fault=='candidate_count':report['rows'][1]['adaptation_count']+=1
                else:samples[prefix+'P3_final_m']=np.zeros(1)
                with self.assertRaises(ValueError):replay.validate_arithmetic(report,samples)

    def test_ties_use_runtime_then_arm_id(self):
        for scene,side in itertools.product(replay.SCENES,replay.SIDES):self.report['runtimes_seconds'][f'{scene}_{side}_P4']=9.
        refresh(self.report,self.samples)
        self.assertEqual(replay.validate_arithmetic(self.report,self.samples)['nominee'],'P4')
        self.report['nominee']='P3'
        with self.assertRaisesRegex(ValueError,'Exact mismatch'):replay.validate_arithmetic(self.report,self.samples)

    def test_twelve_decimal_gain_tie_precedes_runtime(self):
        for scene,side,frame in itertools.product(replay.SCENES,replay.SIDES,replay.FRAMES):
            prefix=f'{scene}_{side}_{frame}__';n=len(self.samples[prefix+'measured_m'])
            self.samples[prefix+'P4_final_m']=np.full(n,2.8+1e-13)
            self.report['runtimes_seconds'][f'{scene}_{side}_P4']=9.
        refresh(self.report,self.samples);result=replay.validate_arithmetic(self.report,self.samples)
        first,second=(result['candidates'][arm] for arm in ('P3','P4'))
        self.assertGreater(first['minimum_directional_gain'],second['minimum_directional_gain'])
        self.assertEqual(first['ranking_gain_rounded_12'],second['ranking_gain_rounded_12'])
        self.assertEqual(result['nominee'],'P4')

    def test_one_failed_direction_cannot_be_pooled_away(self):
        for scene,frame in itertools.product(replay.SCENES,replay.FRAMES):
            prefix=f'{scene}_later_{frame}__';n=len(self.samples[prefix+'measured_m'])
            self.samples[prefix+'P3_final_m']=np.full(n,3.01)
        refresh(self.report,self.samples);result=replay.validate_arithmetic(self.report,self.samples)
        self.assertGreater(result['candidates']['P3']['directional_gain']['earlier'],.01)
        self.assertLess(result['candidates']['P3']['directional_gain']['later'],0.)
        self.assertFalse(result['candidates']['P3']['eligible'])

    def test_reported_tolerance_does_not_waive_exact_eligibility(self):
        self.report['candidates']['P3']['eligible']=False
        with self.assertRaisesRegex(ValueError,'Exact mismatch'):replay.validate_arithmetic(self.report,self.samples)

    def test_same_scenes_required_in_both_directions(self):
        for scene,side,frame in itertools.product(replay.SCENES,replay.SIDES,replay.FRAMES):
            gain_scene=scene in ({'heads','chess'} if side=='earlier' else {'chess','fire'})
            prefix=f'{scene}_{side}_{frame}__';n=len(self.samples[prefix+'measured_m'])
            self.samples[prefix+'P3_final_m']=np.full(n,2.+(.5 if gain_scene else 1.1))
        refresh(self.report,self.samples);result=replay.validate_arithmetic(self.report,self.samples)
        self.assertGreater(result['candidates']['P3']['minimum_directional_gain'],.01)
        self.assertEqual(result['candidates']['P3']['same_scenes_improving_both'],['chess'])
        self.assertFalse(result['candidates']['P3']['eligible'])

    def test_numerical_failure_disqualifies_even_good_arithmetic(self):
        self.report['numerical_failures_by_run_id']['heads_earlier_P3']=1
        self.report['all_failed_attempt_counts']['heads_earlier_P3']=1;refresh(self.report,self.samples)
        self.assertEqual(replay.validate_arithmetic(self.report,self.samples)['nominee'],'P4')


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Files(Path(self.tmp.name))

    def test_linked_replay_and_preserved_receipt(self):
        result=self.f.run();self.assertTrue(result['pass_all']);self.assertFalse(result['raw_endpoints_opened'])
        self.assertEqual(result['fields_checked'],420)
        with self.assertRaisesRegex(ValueError,'Preserve'):self.f.run()

    def test_sample_byte_tamper_rejected_before_decode(self):
        (self.f.root/self.f.samples_name).write_bytes(b'changed')
        with patch.object(np,'load',side_effect=AssertionError('decode forbidden')):
            with self.assertRaisesRegex(ValueError,'before decode'):self.f.run()

    def test_access_required_and_linkage_tampering_fails_before_decode(self):
        (self.f.root/'results/access.json').unlink()
        with patch.object(np,'load',side_effect=AssertionError('decode forbidden')):
            with self.assertRaises(FileNotFoundError):self.f.run()

    def test_duplicate_npz_member_rejected_even_with_rebound_hash(self):
        path=self.f.root/self.f.samples_name;name=next(iter(self.f.samples))+'.npy'
        payload=io.BytesIO();np.save(payload,np.zeros(1),allow_pickle=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(path,'a') as archive:archive.writestr(name,payload.getvalue())
        self.f.report['replay_samples_sha256']=replay.sha(path.read_bytes());self.f.put(self.f.report_name,self.f.report)
        with self.assertRaisesRegex(ValueError,'duplicate compact'):self.f.run()

    def test_report_scalar_tampering_detected(self):
        self.f.report['directional_equal_scene_mae_m']['final']['earlier']['P3']=.1
        self.f.put(self.f.report_name,self.f.report)
        with self.assertRaisesRegex(ValueError,'Arithmetic mismatch'):self.f.run()

    def test_rebound_access_with_wrong_seal_still_rejected(self):
        self.f.access['prediction_seal_sha256']='f'*64
        self.f.report['score_access_receipt_sha256']=self.f.put('results/access.json',self.f.access)
        self.f.put(self.f.report_name,self.f.report)
        with patch.object(np,'load',side_effect=AssertionError('decode forbidden')):
            with self.assertRaisesRegex(ValueError,'linkage'):self.f.run()

    def test_runtime_and_failure_counts_bound_to_seal(self):
        self.f.report['runtimes_seconds']['heads_earlier_P4']=0.
        self.f.put(self.f.report_name,self.f.report)
        with self.assertRaisesRegex(ValueError,'sealed runtime'):self.f.run()

    def test_path_escape_rejected(self):
        self.f.report['score_access_receipt']='../outside';self.f.put(self.f.report_name,self.f.report)
        with self.assertRaises(ValueError):self.f.run()


if __name__=='__main__':unittest.main()
