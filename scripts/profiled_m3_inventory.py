"""Pure prospective M3 inventory and serial stopping-policy scaffolding.

No files, scores, arrays, subprocesses or GPUs are accessed. ``build_plan`` takes
descriptors supplied by a future byte-verifying registration layer. A nominee
descriptor must link P3/P4 to M2's seal, score report, rule and verification
receipt; this module cannot certify those external facts. Synthetic descriptors
are suitable for fixtures only. No real nominee is selected here.

``inspect_progress`` checks explicitly supplied completion/reuse/review receipts
and returns a simulated next action. Stage reviews are necessary-condition
checks, never a complete M3 pass. Even a complete inventory ends at
``ready_for_full_inventory_gate``; the separately registered full gate and byte
verification remain mandatory. All planned command scopes are unimplemented,
non-executable descriptions. This module provides no launch authorization.
"""
from copy import deepcopy
import itertools

from profiled_m3_gate import (SCENES, SIDES, GRIDS, POSES, REPEATS, CONTROLS, REFERENCES,
                             ENDPOINTS, canonical_sha256, cell_id, digest, policy,
                             require, state_id)


INPUT_PLAN="manifests/lora_profiled_m3_inputs_v2.json"
INPUT_ATTEMPT="inputs02"
NOMINATION_FIELDS=("m2_prediction_seal_sha256","m2_score_report_sha256",
                   "nomination_rules_sha256","verification_receipt_sha256")
IDENTITIES=("runtime_sha256","endpoint_contract_sha256","checkpoint_sha256")
REUSE_CHECKS=("original_isolated_process","original_completion_and_endpoint_hashes",
              "pretrained_checkpoint_and_zero_function_initialization","seed42_and_optimizer_schedule",
              "allowed_packet_and_selected_original_rows","spatial_folds_and_native_pixels",
              "supplied_clean_camera_and_intrinsic_values","strict_runtime_and_precision",
              "unchanged_common_native_and_final_endpoint","original_M2_seal_membership")
CELL_ORDER=tuple((side,grid,pose) for grid,pose in
                 (("centered","clean"),("centered","noisy"),("shifted","clean"),("shifted","noisy"))
                 for side in SIDES)


def case_id(scene,side,grid,pose):
    return "_".join((scene,side,grid,pose))


def build_plan(nomination,inputs):
    """Build all360 states; retain30 reuse decisions as explicitly unresolved.

    ``nomination``: nominee plus four hashes in NOMINATION_FIELDS. ``inputs``:
    fixed plan_file/attempt, plan_sha256, preparation_receipt_sha256 keyed
    earlier/later/poses, case_input_sha256 keyed all24 scene/side/grid/pose cases,
    pose_matrix_sha256 keyed all6 scene/pose combinations, and three common
    hashes in IDENTITIES. Their bytes are not opened here.
    """
    require(isinstance(nomination,dict) and set(nomination)=={"nominee",*NOMINATION_FIELDS},
            "A complete externally verifiable M2 nomination linkage is required")
    nominee=nomination["nominee"]
    require(nominee in ("P3","P4"),"Only the M2-selected P3/P4 nominee is eligible")
    for field in NOMINATION_FIELDS:
        digest(nomination[field],field)
    require(set(inputs)=={"plan_file","attempt","plan_sha256","preparation_receipt_sha256",
                         "case_input_sha256","pose_matrix_sha256",*IDENTITIES},"Incomplete prospective input descriptors")
    require(inputs["plan_file"]==INPUT_PLAN and inputs["attempt"]==INPUT_ATTEMPT,
            "Expected preserved input plan v2 / inputs02")
    for field in ("plan_sha256",*IDENTITIES):
        digest(inputs[field],field)
    require(set(inputs["preparation_receipt_sha256"])=={"earlier","later","poses"},
            "Signed-source and fixed-pose preparation receipts are required")
    for key,value in inputs["preparation_receipt_sha256"].items():
        digest(value,key+" preparation receipt")
    expected={case_id(*parts) for parts in itertools.product(SCENES,SIDES,GRIDS,POSES)}
    require(set(inputs["case_input_sha256"])==expected,"Exactly24 paired case input identities are required")
    for key,value in inputs["case_input_sha256"].items():
        digest(value,key)
    require(set(inputs["pose_matrix_sha256"])=={"/".join(parts) for parts in itertools.product(SCENES,POSES)},
            "Exactly one clean and one fixed noisy matrix packet per scene are required")
    for key,value in inputs["pose_matrix_sha256"].items():
        digest(value,key+" pose matrices")
    require(all(inputs["pose_matrix_sha256"][scene+"/clean"]!=inputs["pose_matrix_sha256"][scene+"/noisy"]
                for scene in SCENES),"Clean and noisy matrix packets must differ")
    arms=tuple(sorted(CONTROLS+(nominee,),key=lambda value:int(value[1:])))
    stages=[]
    # Both directional centered/clean checks share the first stop barrier. No
    # new condition starts until both have been sealed, scored and reviewed.
    specs=[(0,CELL_ORDER[:2])]+[(0,(cell,)) for cell in CELL_ORDER[2:]]
    specs.extend((repeat,(cell,)) for repeat in (1,2) for cell in CELL_ORDER)
    jobs=[]
    previous_stage=None
    for stage_index,(repeat,cells) in enumerate(specs):
        stage_id=("r0_centered_clean_both_directions" if stage_index==0
                  else "r"+str(repeat)+"_"+cell_id(*cells[0]))
        stage_jobs=[]
        previous_job=None
        for (side,grid,pose),scene,arm in itertools.product(cells,SCENES,arms):
            job_id=state_id(scene,side,grid,pose,repeat,arm)
            eligible=repeat==0 and grid=="centered" and pose=="clean"
            dependencies=["verified_nomination","scientific_registration","input_endpoint_acceptance"]
            if previous_stage is not None:
                dependencies.append("stage_review:"+previous_stage)
            if previous_job is not None:
                dependencies.append("completed_job:"+previous_job)
            jobs.append(dict(job_id=job_id,stage_id=stage_id,stage_index=stage_index,
                scene=scene,side=side,grid=grid,pose=pose,repeat=repeat,arm=arm,
                case_id=case_id(scene,side,grid,pose),
                input_identity_sha256=inputs["case_input_sha256"][case_id(scene,side,grid,pose)],
                pose_identity_sha256=inputs["pose_matrix_sha256"][scene+"/"+pose],
                **{field:inputs[field] for field in IDENTITIES},dependencies=dependencies,
                reuse=dict(eligible=eligible,status="unresolved" if eligible else "not_eligible",
                           m2_run_id=f"{scene}_{side}_{arm}" if eligible else None,
                           requires=list(REUSE_CHECKS) if eligible else []),
                training=dict(fresh_pretrained_process=True,original_weights_frozen=True,seed=42,
                    lora_enabled=arm!="P0",rank=4 if arm!="P0" else None,alpha=1 if arm!="P0" else None,
                    zero_function_initialization=arm!="P0",macrosteps=0 if arm=="P0" else 8,
                    ordered_pairs_per_macrostep=0 if arm=="P0" else 20,
                    collect_replay_passes_per_macrostep=0 if arm=="P0" else 2,
                    optimizer="none" if arm=="P0" else "Adam",learning_rate=None if arm=="P0" else 0.0001,
                    clip_gradient_norm=None if arm=="P0" else 1.),
                outputs=dict(endpoints=list(ENDPOINTS),references=list(REFERENCES),
                    raw_scope=f"reports/lora_profiled_m3_v1/cases/{job_id}/<fresh_attempt>",
                    public_scope=f"results/lora_profiled_m3_v1/cases/{job_id}/<fresh_attempt>"),
                command_scope=dict(status="prospective_unimplemented",executable=False,
                    operation="one_fresh_local_case_no_scoring",
                    planned_argv=["python","scripts/run_profiled_m3.py","--registry","<registered_M3_registry>",
                        "--job-id",job_id,"--attempt","<fresh_attempt>","--weights","<registered_local_weights>",
                        "--image","<registered_pinned_image>"])))
            stage_jobs.append(job_id)
            previous_job=job_id
        stages.append(dict(stage_id=stage_id,stage_index=stage_index,repeat=repeat,
            cells=[cell_id(*cell) for cell in cells],job_ids=stage_jobs,
            requires_previous_review=previous_stage,required_states=len(stage_jobs),
            required_reference_rows=len(stage_jobs)*2,required_endpoint_metrics=len(stage_jobs)*4,
            score_only_after="all_stage_predictions_and_both_endpoints_are_sealed",
            review_scope="r0_necessary_conditions" if repeat==0 else "repeat_safety" if repeat==1 else "repeat_safety_and_full_cell_mechanism",
            control_policy="select_primary_r0_only" if repeat==0 else "retain_primary_r0_control",
            pass_scope="necessary_stage_precheck_only_never_full_inventory_qualification"))
        previous_stage=stage_id
    result=dict(schema="profiled_m3_inventory_v1",nomination=deepcopy(nomination),inputs=deepcopy(inputs),
        policy=policy(),arms=list(arms),matched_detached_arm={"P3":"P2","P4":"P5"}[nominee],
        cell_order=[cell_id(*cell) for cell in CELL_ORDER],stages=stages,jobs=jobs,
        required_states=360,required_reference_rows=720,required_endpoint_metrics=1440,
        eligible_M2_reuse_states=30,new_states_if_all_reuse_is_approved=330,
        new_states_if_no_reuse_is_approved=360,resource=dict(local_gpu_count=1,max_concurrent_jobs=1,rental_usd=0),
        launch_authorized=False,source_and_byte_verification_performed=False,
        future_controller_requirements=["verify_M2_nomination_linkage","freeze_runner_evaluator_and_runtime",
            "verify_preparation_and_centered_clean_endpoint_equivalence","audit_each_reused_M2_state",
            "seal_each_complete_stage_before_dense_reference_access","preserve_original_attempt_and_failure_receipts",
            "apply_full360_state_gate_after_all_staged_prechecks"])
    result["plan_sha256"]=canonical_sha256(result)
    return result


def _validate_plan(plan):
    require(plan==build_plan(plan["nomination"],plan["inputs"]),"Prospective inventory or policy changed")


def _identity(plan,receipt,schema):
    require(receipt.get("schema")==schema and receipt.get("plan_sha256")==plan["plan_sha256"],
            "Receipt belongs to another schema or plan")


def inspect_progress(plan,*,completed=None,reuse_audits=None,stage_reviews=None,failed=None):
    """Simulate the next serial action; never execute or certify external bytes.

    Maps are keyed job_id, except stage_reviews keyed stage_id. Complete receipts
    bind both endpoints, input/runtime/endpoint/checkpoint and a fresh-process
    receipt. Approved reuse audits bind the original M2 completion and endpoints
    and every REUSE_CHECKS predicate. Stage reviews include the full paired metric
    inventory and a decision/control per cell. Their decisions must be produced
    by the future registered early-cell evaluator, not hand-selected.
    """
    _validate_plan(plan)
    completed=completed or {}
    reuse_audits=reuse_audits or {}
    stage_reviews=stage_reviews or {}
    failed=failed or {}
    jobs={job["job_id"]:job for job in plan["jobs"]}
    stages={stage["stage_id"]:stage for stage in plan["stages"]}
    require(set(completed)<=set(jobs) and set(failed)<=set(jobs) and set(reuse_audits)<=set(jobs),"Unknown job receipt")
    require(set(stage_reviews)<=set(stages),"Unknown stage review")
    require(not set(completed)&set(failed),"A failed attempt cannot silently become a completed state")
    for job_id,audit in reuse_audits.items():
        job=jobs[job_id]
        _identity(plan,audit,"profiled_m3_reuse_audit_v1")
        require(job["reuse"]["eligible"] and audit.get("job_id")==job_id and
                audit.get("m2_run_id")==job["reuse"]["m2_run_id"],"Ineligible or mismatched M2 reuse")
        require(set(audit.get("checks",{}))==set(REUSE_CHECKS) and
                all(type(value) is bool for value in audit["checks"].values()),"Incomplete reuse audit predicates")
        require(audit.get("decision") in ("approved","rejected") and
                (audit["decision"]=="approved")==all(audit["checks"].values()),"Reuse verdict disagrees with exact-match predicates")
        digest(audit.get("audit_receipt_sha256"),"reuse audit receipt")
        if audit["decision"]=="approved":
            for field in ("m2_completion_sha256","native_sha256","final_sha256"):
                digest(audit.get(field),field)
    processes={}
    for job_id,receipt in completed.items():
        job=jobs[job_id]
        _identity(plan,receipt,"profiled_m3_state_completion_v1")
        require(receipt.get("job_id")==job_id and receipt.get("origin") in ("fresh","m2_reuse"),"Wrong completed-state identity")
        for field in ("input_identity_sha256","pose_identity_sha256",*IDENTITIES):
            require(receipt.get(field)==job[field],"Completed state changed paired inputs/runtime/endpoint: "+job_id)
        for field in ("native_sha256","final_sha256","completion_receipt_sha256","process_receipt_sha256"):
            digest(receipt.get(field),job_id+"/"+field)
        process=receipt["process_receipt_sha256"]
        require(process not in processes,"Fresh-process receipt reused across states")
        processes[process]=job_id
        if job["reuse"]["eligible"]:
            require(job_id in reuse_audits,"Resolve conditional reuse before spending on an eligible fresh state")
        if receipt["origin"]=="m2_reuse":
            audit=reuse_audits.get(job_id,{})
            require(audit.get("decision")=="approved","Reuse needs its own approved exact-match audit")
            require(receipt["completion_receipt_sha256"]==audit["m2_completion_sha256"] and
                    all(receipt[field]==audit[field] for field in ("native_sha256","final_sha256")),
                    "Reused endpoints/receipt differ from the original M2 evidence")
    for job_id,receipt in failed.items():
        _identity(plan,receipt,"profiled_m3_failure_receipt_v1")
        require(receipt.get("job_id")==job_id and receipt.get("classification") in
                ("numerical","infrastructure","unclassified"),"Unknown failure classification")
        digest(receipt.get("failure_receipt_sha256"),"failure receipt")
    controls={}
    reference_masks={}
    reviewed=[]
    active=None
    terminal=None
    for stage in plan["stages"]:
        stage_id=stage["stage_id"]
        if terminal is not None or active is not None:
            require(stage_id not in stage_reviews,"Later stage reviewed before its predecessor passed")
            require(not(set(stage["job_ids"])&(set(completed)|set(failed))),"Later states executed beyond a conditional stop barrier")
            continue
        failures={job_id:failed[job_id] for job_id in stage["job_ids"] if job_id in failed}
        if failures:
            require(stage_id not in stage_reviews,"Failed/incomplete stage cannot receive a passing score review")
            terminal=dict(status="stopped_numerical_failure" if any(row["classification"]=="numerical" for row in failures.values())
                          else "failure_review_required",stage_id=stage_id,failures=deepcopy(failures))
            continue
        if stage_id not in stage_reviews:
            found_missing=False
            for job_id in stage["job_ids"]:
                if job_id not in completed:
                    found_missing=True
                elif found_missing and completed[job_id]["origin"]=="fresh":
                    raise ValueError("Fresh jobs violate the registered one-GPU serial dependency order")
            active=stage
            continue
        review=stage_reviews[stage_id]
        _identity(plan,review,"profiled_m3_stage_review_v1")
        require(review.get("stage_id")==stage_id and set(stage["job_ids"])<=set(completed),
                "A stage review requires every paired completed state")
        inventory=review.get("metrics",[])
        expected=set(itertools.product(stage["job_ids"],REFERENCES))
        require(isinstance(inventory,list) and len(inventory)==len(expected),"Missing paired stage reference metrics")
        seen=set()
        for row in inventory:
            key=(row.get("job_id"),row.get("reference"))
            require(key in expected and key not in seen,"Duplicate or wrong paired stage metric")
            seen.add(key)
            for field in ("final_metrics_sha256","native_metrics_sha256","fixed_mask_sha256"):
                digest(row.get(field),field)
            for kind in ENDPOINTS:
                require(row.get(kind+"_endpoint_sha256")==completed[key[0]][kind+"_sha256"],
                        "Stage metrics belong to a different completed endpoint")
            mask_key=(jobs[key[0]]["scene"],key[1])
            mask=row["fixed_mask_sha256"]
            require(mask_key not in reference_masks or reference_masks[mask_key]==mask,
                    "Fixed reference mask differs across arms or conditions")
            reference_masks[mask_key]=mask
        decisions=review.get("cells",{})
        require(set(decisions)==set(stage["cells"]),"Stage review omitted a directional/grid/pose cell")
        for cell,decision in decisions.items():
            require(decision.get("status") in ("passed_precheck","failed"),"Partial stage review cannot count as passing")
            digest(decision.get("evaluation_receipt_sha256"),"cell evaluation receipt")
            control=decision.get("selected_control")
            require(control in CONTROLS or (control is None and decision["status"]=="failed" and stage["repeat"]==0),
                    "Control must come from the primary-r0 eligible pool")
            if stage["repeat"]==0:
                controls[cell]=control
            else:
                require(control==controls[cell],"Primary-r0 control identity changed in a later repetition")
            if stage["repeat"]==2:
                require(type(decision.get("mechanism_confirmed")) is bool and
                        (decision["status"]!="passed_precheck" or decision["mechanism_confirmed"]),
                        "Final repeat needs the complete three-repeat mechanism decision")
            else:
                require(decision.get("mechanism_confirmed") is None,"Early precheck cannot claim full-repeat mechanism confirmation")
        reviewed.append(stage_id)
        if any(row["status"]=="failed" for row in decisions.values()):
            terminal=dict(status="stopped_mandatory_gate_failure",stage_id=stage_id,cell_decisions=deepcopy(decisions))
    if terminal is not None:
        next_action=dict(operation="preserve_failure_and_conditional_omissions",**terminal)
        status=terminal["status"]
    elif active is None:
        status="ready_for_full_inventory_gate"
        next_action=dict(operation="run_separately_registered_complete_inventory_gate",required_states=360,required_reference_rows=720)
    else:
        missing=[job_id for job_id in active["job_ids"] if job_id not in completed]
        if not missing:
            status="awaiting_stage_seal_metrics_and_review"
            next_action=dict(operation="seal_then_score_all_stage_references_and_endpoints",stage_id=active["stage_id"],
                             job_ids=list(active["job_ids"]))
        else:
            job_id=missing[0]
            job=jobs[job_id]
            if job["reuse"]["eligible"] and job_id not in reuse_audits:
                operation="audit_conditional_M2_reuse"
            elif reuse_audits.get(job_id,{}).get("decision")=="approved":
                operation="bind_verified_original_M2_receipt"
            else:
                operation="plan_one_fresh_local_process"
            status="awaiting_job"
            next_action=dict(operation=operation,stage_id=active["stage_id"],job_id=job_id,
                             command_scope=deepcopy(job["command_scope"]))
    remaining=[job["job_id"] for job in plan["jobs"] if job["job_id"] not in completed and job["job_id"] not in failed]
    return dict(schema="profiled_m3_progress_simulation_v1",plan_sha256=plan["plan_sha256"],status=status,
        next_action=next_action,completed_states=len(completed),failed_states=len(failed),required_states=360,
        reviewed_stages=reviewed,frozen_r0_controls=controls,conditional_unexecuted_job_ids=remaining,
        preserved_failures=deepcopy(failed),preserved_stage_reviews=deepcopy(stage_reviews),
        all_model_states_completed=len(completed)==360,final_gate_passed=False,
        partial_inventory_qualified=False,launch_authorized=False,simulation_only=True,
        external_source_byte_nomination_and_registration_verification_required=True)
