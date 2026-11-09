"""Predeclared 61-job screen; fresh controls account for a migrated physical GPU."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = "diagnostics_v2"


def build_jobs():
    manifest = json.loads((ROOT / "manifests/midterm_3scene_kf200.json").read_text())
    jobs = []
    def add(scene, profile, steps, weight, label):
        jobs.append(dict(run_id=f"diag2_{scene.replace('/', '_')}_{label}", scene=scene,
                         profile=profile, steps=steps, weight=weight, frames=manifest[scene],
                         seed=42, prior_seed=1729, size=518))
    add("heads/seq-01", "clean", 0, 1, "bridge_clean_s0")
    add("heads/seq-01", "mixed_severe", 0, 1, "bridge_corrupt_s0")
    add("heads/seq-01", "clean", 40, 1, "bridge_clean_s40")
    for scene in manifest:
        if scene != "heads/seq-01":
            for steps in [0,40]:
                add(scene,"clean",steps,1,f"bridge_clean_s{steps}")
    for scene in manifest:
        for steps in [5, 10, 20]:
            add(scene, "clean", steps, 1, f"clean_s{steps}")
        for profile in ["rotation5", "translation10", "focal_plus10", "mixed_moderate", "mixed_severe"]:
            add(scene, profile, 40, 1, f"{profile}_s40")
        add(scene, "mixed_severe", 40, 0.1, "mixed_severe_weak_s40")
        add(scene, "clean", 40, 0, "visual_only_s40")
    assert len(jobs) == 61 and len({j['run_id'] for j in jobs}) == 61
    return jobs


def bridge_checks():
    import numpy as np
    clean=ROOT/'runs/diag2_heads_seq-01_bridge_clean_s0/predictions.npy'
    noisy=ROOT/'runs/diag2_heads_seq-01_bridge_corrupt_s0/predictions.npy'
    assert np.array_equal(np.load(clean),np.load(noisy)), "Zero-step prior invariance failed; investigate before proceeding"
    for steps in [0,40]:
        old=json.loads((ROOT/f'runs/cloud_heads_seq-01_518_s{steps}_seed42/run.json').read_text())
        new=json.loads((ROOT/f'runs/diag2_heads_seq-01_bridge_clean_s{steps}/run.json').read_text())
        for key in ['input_sha256','reference_sha256','upstream_method_sha256','versions','configuration','checkpoint_download','gsplat_backend','gpu']:
            assert old[key]==new[key], f"Bridge provenance mismatch: {key}"
        for key in ['accuracy_mean_m','completeness_mean_m']:
            assert abs(new['metrics'][key]/old['metrics'][key]-1)<=0.01, f"Bridge metric mismatch: {key}"
        assert abs(new['metrics']['normal_consistency_mean']-old['metrics']['normal_consistency_mean'])<=0.005
    return {'status':'passed','zero_step_corruption_bitwise_invariant':True,
            'clean_bridge_error_relative_tolerance':0.01,'normal_consistency_absolute_tolerance':0.005}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=pathlib.Path,default=pathlib.Path('/workspace/tco-assets'))
    parser.add_argument('--max-hours',type=float,default=2.5)
    parser.add_argument('--per-job-minutes',type=float,default=6)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    jobs=build_jobs()
    plan={'study_id':STUDY,'scope':'exploratory_on_previously_inspected_sequences','jobs':jobs,
          'profiles':json.loads((ROOT/'configs/diagnostic_profiles.json').read_text()),
          'max_hours':args.max_hours,'per_job_minutes':args.per_job_minutes,'session_cap_usd':3,
          'stop_conditions':['first failure','bridge mismatch','per-job timeout','batch deadline','61 jobs finished'],
          'same_session_controls':'Fresh 0/40-step controls on all five scenes; v1 superseded before execution because the GPU required migration',
          'seed_limit':'one optimization seed and one synthetic corruption seed; no inferential statistics',
          'original_runs_prefix':'cloud','runtime_note':'independent fixed-step runs, not saved optimizer checkpoints'}
    report=ROOT/'reports'/STUDY
    report.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'manifests/diagnostics_v2.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())==plan,'Frozen plan differs; assign a new study ID'
    else:
        manifest_path.write_text(json.dumps(plan,indent=2)+'\n')
    if args.dry_run:
        print(json.dumps({'jobs':len(jobs),'manifest':str(manifest_path),'plan_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()},indent=2));return
    deadline=time.monotonic()+args.max_hours*3600
    status=[]
    for index,j in enumerate(jobs):
        run=ROOT/'runs'/j['run_id']; record=run/'run.json'
        assert not run.exists(),f"Preserve previous evidence and choose a new study/run ID: {run}"
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError('Batch deadline reached')
        cmd=[sys.executable,'-u',str(ROOT/'scripts/run_diagnostic.py'),'--run-id',j['run_id'],
             '--study-id',STUDY,'--scene',j['scene'],'--size',str(j['size']),'--steps',str(j['steps']),
             '--seed',str(j['seed']),'--prior-profile',j['profile'],'--prior-seed',str(j['prior_seed']),
             '--prior-weight',str(j['weight']),'--data-root',str(args.data_root/'data/7scenes'),
             '--dataset-cache',str(args.data_root/'data/7scenes/full_sequence_metadata.npy'),
             '--checkpoint',str(args.data_root/'checkpoints/VGGT-1B'),'--frames',*map(str,j['frames'])]
        print(f"Starting {index+1}/{len(jobs)} {j['run_id']}",flush=True)
        log=ROOT/'runs/logs'/f"{j['run_id']}.log"; log.parent.mkdir(exist_ok=True)
        entry={'run_id':j['run_id'],'status':'running','command':cmd};status.append(entry)
        (report/'batch_status.json').write_text(json.dumps(status,indent=2))
        with log.open('w') as stream:
            try:
                result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=min(remaining,args.per_job_minutes*60))
                entry['exit_code']=result.returncode
                assert result.returncode==0, f"Job failed: {j['run_id']}"
                assert json.loads(record.read_text())['status']=='complete'
                entry['status']='complete'
            except BaseException as exc:
                entry.update(status='failed_or_timed_out',error=repr(exc))
                if record.exists():
                    failed=json.loads(record.read_text());failed.update(status='failed_or_timed_out',batch_error=repr(exc));record.write_text(json.dumps(failed,indent=2))
                raise
            finally:
                (report/'batch_status.json').write_text(json.dumps(status,indent=2))
        if index==10:
            (report/'bridge_checks.json').write_text(json.dumps(bridge_checks(),indent=2)+'\n')
            print('Bridge and zero-step invariance passed',flush=True)
        print(f"Completed {index+1}/{len(jobs)} {j['run_id']}",flush=True)


if __name__=='__main__': main()
