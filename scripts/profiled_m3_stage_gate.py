"""Non-authorizing M3 early-cell necessary conditions, using decimal fractions.

Pure dictionaries only: no source bytes, arrays, files or scores are opened.
The early registration freezes the full gate's policy/input/reference/runtime
metadata but has schema ``profiled_m3_early_registration_v1`` and omits the
not-yet-existing global ``prediction_seal_sha256`` from provenance. Individual
stage prediction-seal descriptors are supplied separately as repeats finish.

``evaluate_cell(registration, rows, cell=..., repeat=..., stage_seals=...,
prior_r0=...)`` requires all30 paired scene/arm/reference rows per available
repeat0..repeat. A descriptor may seal both initial centered/clean directions;
only the requested cell's metric subset is consumed here. The future byte
verifier/controller must establish the complete stage, authenticate the seals,
compute the metrics and honor previous stop decisions. This module checks
declared identities, never manufactures external verification receipt hashes.

r0/r1 require positive full-minus-detached contrasts, deferring the fixed1e-6
margin and execution floor until r2. Those early checks are necessary, not
sufficient. Later calls require the exact previous passing r0 report, replayed
against unchanged canonical r0 rows and seal identity. No partial or complete
single-cell result can qualify the full study, release money or launch a job.
"""
from copy import deepcopy
from fractions import Fraction
import itertools

import profiled_m3_gate as full


EARLY_PROVENANCE=tuple(key for key in full.PROVENANCE if key!="prediction_seal_sha256")
CELLS={full.cell_id(*parts):parts for parts in itertools.product(full.SIDES,full.GRIDS,full.POSES)}


def _registration(record):
    full.require(record.get("schema")=="profiled_m3_early_registration_v1","Wrong early registration schema")
    full.require(record.get("nominee") in ("P3","P4") and record.get("policy")==full.policy(),"Early nominee/policy differs")
    full.require(set(record.get("provenance",{}))==set(EARLY_PROVENANCE),"Early provenance must omit the future global seal")
    for key,value in record["provenance"].items():
        full.digest(value,key)
    full.require(set(record.get("scenes",{}))==set(full.SCENES),"Wrong early scene inventory")
    for index,scene in enumerate(full.SCENES):
        spec=record["scenes"][scene]
        full.require(set(spec)=={"rgb_sha256","clean_pose_sha256","noisy_pose_sha256","noise_seed"},"Wrong scene identities")
        full.require(type(spec["noise_seed"]) is int and spec["noise_seed"]==20261112+index,"Wrong fixed noise seed")
        for field in ("rgb_sha256","clean_pose_sha256","noisy_pose_sha256"):
            full.digest(spec[field],scene+"/"+field)
        full.require(spec["clean_pose_sha256"]!=spec["noisy_pose_sha256"],"Clean/noisy poses must differ")
    expected={"/".join(parts) for parts in itertools.product(full.SCENES,full.SIDES,full.GRIDS)}
    full.require(set(record.get("observations",{}))==expected,"Wrong signed/grid input inventory")
    for key,spec in record["observations"].items():
        full.require(set(spec)=={"packet_sha256","folds_sha256","native_rows_sha256"},"Wrong observation identities")
        for field,value in spec.items():
            full.digest(value,key+"/"+field)
    expected={"/".join(parts) for parts in itertools.product(full.SCENES,full.REFERENCES)}
    full.require(set(record.get("references",{}))==expected,"Wrong dense reference inventory")
    for key,spec in record["references"].items():
        full.require(set(spec)=={"reference_sha256","mask_sha256","mask_count"},"Wrong reference identities")
        for field in ("reference_sha256","mask_sha256"):
            full.digest(spec[field],key+"/"+field)
        full.require(full.count(spec["mask_count"],key)>0,"Empty scene reference mask")
    for scene in full.SCENES:
        first,second=(record["references"][scene+"/"+reference] for reference in full.REFERENCES)
        full.require(first["mask_sha256"]==second["mask_sha256"] and first["mask_count"]==second["mask_count"],
                     "Both reference conventions must use the same archived scene mask")
    return record["nominee"]


def _validate(registration,rows,cell,repeat,stage_seals):
    nominee=_registration(registration)
    full.require(cell in CELLS and type(repeat) is int and repeat in full.REPEATS,"Unknown cell or repetition")
    side,grid,pose=CELLS[cell]
    arms=tuple(sorted(full.CONTROLS+(nominee,),key=lambda arm:int(arm[1:])))
    repeats=tuple(range(repeat+1))
    registry_hash=full.canonical_sha256(registration)
    full.require(isinstance(stage_seals,dict) and set(stage_seals)=={str(r) for r in repeats},"Missing or unexpected stage seal")
    for r in repeats:
        seal=stage_seals[str(r)]
        full.require(seal.get("schema")=="profiled_m3_stage_seal_descriptor_v1" and
                     seal.get("registration_sha256")==registry_hash and type(seal.get("repeat")) is int and seal["repeat"]==r,
                     "Stage seal belongs to another registry or repetition")
        full.digest(seal.get("prediction_seal_sha256"),"stage prediction seal")
        cells=seal.get("cells",[])
        full.require(isinstance(cells,list) and all(isinstance(value,str) and value in CELLS for value in cells)
                     and cell in cells and len(set(cells))==len(cells),"Stage seal does not cover this cell")
        expected={full.state_id(scene,*CELLS[scope],r,arm)
                  for scene,scope,arm in itertools.product(full.SCENES,cells,arms)}
        states=seal.get("state_ids",[])
        full.require(isinstance(states,list) and len(states)==len(expected) and set(states)==expected,
                     "Incomplete declared stage prediction inventory")
        full.require(seal.get("endpoints")==list(full.ENDPOINTS),"Stage must seal both main and native endpoints")
    full.require(len({stage_seals[str(r)]["prediction_seal_sha256"] for r in repeats})==len(repeats),
                 "Distinct repeat stages cannot reuse one prediction seal identity")
    expected=set(itertools.product(full.SCENES,repeats,arms,full.REFERENCES))
    full.require(isinstance(rows,list) and len(rows)==len(expected),"Cell requires every paired row for repeats0 through current")
    index,states,receipts={},{},{}
    for row in rows:
        full.require(isinstance(row,dict),"Metric row must be an object")
        full.require((row.get("side"),row.get("grid"),row.get("pose"))==(side,grid,pose),"Metric row belongs to another cell")
        key=tuple(row.get(field) for field in ("scene","repeat","arm","reference"))
        full.require(type(row.get("repeat")) is int and key in expected and key not in index,"Missing/duplicate/wrong paired cell metric")
        scene,r,arm,reference=key
        sid=full.state_id(scene,side,grid,pose,r,arm)
        full.require(row.get("state_id")==sid,"State ID differs from row pairing")
        identity=row.get("identity",{})
        full.require(set(identity)==set(full.PAIR_HASHES)|{"optimizer_seed","process_receipt_sha256"},"Incomplete state identity")
        scene_spec=registration["scenes"][scene]
        ref=registration["references"][scene+"/"+reference]
        want=dict(rgb_sha256=scene_spec["rgb_sha256"],pose_sha256=scene_spec[pose+"_pose_sha256"],
            **registration["observations"]["/".join((scene,side,grid))],
            **{field:ref[field] for field in ("reference_sha256","mask_sha256")},
            **{field:registration["provenance"][field] for field in
               ("runtime_sha256","original_checkpoint_sha256","initialization_contract_sha256")})
        full.require(all(identity.get(field)==value for field,value in want.items()),"Unpaired input/reference/runtime identity")
        full.require(type(identity.get("optimizer_seed")) is int and identity["optimizer_seed"]==42,"Wrong optimizer seed")
        receipt=full.digest(identity.get("process_receipt_sha256"),"process receipt")
        endpoints=row.get("endpoint_sha256",{})
        full.require(set(endpoints)==set(full.ENDPOINTS),"Missing main/native endpoint hash")
        for kind,value in endpoints.items():
            full.digest(value,kind)
        full.digest(row.get("metrics_sha256"),"metric receipt")
        proof=dict(identity={field:value for field,value in identity.items() if field not in ("reference_sha256","mask_sha256")},
                   endpoint_sha256=endpoints)
        full.require(sid not in states or states[sid]==proof,"References describe different state/endpoint evidence")
        full.require(receipt not in receipts or receipts[receipt]==sid,"Process receipt reused across distinct states")
        states[sid],receipts[receipt]=proof,sid
        validity=row.get("validity",{})
        full.require(set(validity)==set(full.VALIDITY),"Wrong raw validity inventory")
        for field,value in validity.items():
            full.count(value,field)
        full.require(validity["mask_count"]==validity["evaluated_mask_count"]==ref["mask_count"],"Fixed mask changed or invalid pixels were dropped")
        for field in ("main_nonpositive_depth","native_nonpositive_depth"):
            full.require(validity[field]<=ref["mask_count"],"Invalid depth count exceeds mask")
        for kind in full.ENDPOINTS:
            metrics=row.get(kind,{})
            full.require(set(metrics)=={"accuracy","completeness"},"Missing A/C components")
            for field,value in metrics.items():
                full.number(value,kind+"/"+field)
        index[key]=row
    full.require(set(index)==expected and len(states)==15*len(repeats),"Missing paired state")
    for scene,r,arm in itertools.product(full.SCENES,repeats,arms):
        pair=[index[(scene,r,arm,reference)] for reference in full.REFERENCES]
        # All these counts concern the same output on the same shared mask.
        full.require(pair[0]["validity"]==pair[1]["validity"],"Shared-mask validity counts differ across references")
    ordered=[deepcopy(index[key]) for key in sorted(index,key=lambda k:(k[1],full.SCENES.index(k[0]),int(k[2][1:]),full.REFERENCES.index(k[3])))]
    return nominee,arms,index,ordered


def evaluate_cell(registration,rows,*,cell,repeat,stage_seals,prior_r0=None):
    """Necessary stage review, with no fabricated rows or external receipts."""
    nominee,arms,index,ordered=_validate(registration,rows,cell,repeat,stage_seals)
    side,grid,pose=CELLS[cell]
    repeats=tuple(range(repeat+1))
    detached={"P3":"P2","P4":"P5"}[nominee]
    if repeat==0:
        full.require(prior_r0 is None,"r0 cannot inherit a separately chosen control")
    else:
        full.require(isinstance(prior_r0,dict),"Later repetition requires explicit prior-r0 evidence")
        expected=evaluate_cell(registration,[row for row in ordered if row["repeat"]==0],cell=cell,repeat=0,
                               stage_seals={"0":stage_seals["0"]})
        full.require(full.canonical_sha256(prior_r0)==full.canonical_sha256(expected),
                     "Prior-r0 report/control/raw identities or original seal changed")
        full.require(prior_r0["pass_precheck"],"A failed r0 precheck cannot authorize a later repetition")
    normalized,aggregate,table={},{},[]
    for r,reference,kind,arm in itertools.product(repeats,full.REFERENCES,full.ENDPOINTS,arms):
        values=[]
        for scene in full.SCENES:
            row=index[(scene,r,arm,reference)]
            p0=index[(scene,r,"P0",reference)]
            a,c=(full.number(row[kind][field],field) for field in ("accuracy","completeness"))
            a0,c0=(full.number(p0[kind][field],"P0 "+field) for field in ("accuracy","completeness"))
            full.require(a0>0 and c0>0,"Zero P0 normalization denominator")
            an,cn=a/a0,c/c0
            value=dict(A=an,C=cn,S=(an+cn)/2,A_m=a,C_m=c)
            normalized[(scene,r,arm,reference,kind)]=value
            values.append(value)
            table.append(dict(state_id=row["state_id"],cell=cell,scene=scene,repeat=r,arm=arm,reference=reference,endpoint=kind,
                accuracy_m=float(a),completeness_m=float(c),P0_accuracy_m=float(a0),P0_completeness_m=float(c0),
                accuracy_normalized=full.scalar(an),completeness_normalized=full.scalar(cn),S=full.scalar(value["S"]),
                validity=deepcopy(row["validity"])))
        aggregate[(r,reference,kind,arm)]={field:full.mean([value[field] for value in values]) for field in ("A","C","S","A_m","C_m")}
    def score(arm,r,reference=full.REFERENCES[0],kind="final"):
        return aggregate[(r,reference,kind,arm)]
    best=min(full.CONTROLS,key=lambda arm:(score(arm,0)["S"],int(arm[1:])))
    decisions=[]
    contrasts=[]
    for r in repeats:
        f,b=score(nominee,r),score(best,r)
        full.require(b["S"]>0,"Zero selected-control score")
        gain=1-f["S"]/b["S"]
        scenes=[]
        for scene in full.SCENES:
            fs=normalized[(scene,r,nominee,full.REFERENCES[0],"final")]["S"]
            bs=normalized[(scene,r,best,full.REFERENCES[0],"final")]["S"]
            scenes.append(dict(scene=scene,nominee_S=full.scalar(fs),control_S=full.scalar(bs),
                ratio=full.scalar(fs/bs) if bs else None,ratio_status="defined" if bs else "zero_control_scene_score",no_regression=fs<=bs))
        all_rows=[index[(scene,r,arm,reference)] for scene,arm,reference in itertools.product(full.SCENES,arms,full.REFERENCES)]
        af,ab=score(nominee,r,full.REFERENCES[1]),score(best,r,full.REFERENCES[1])
        checks=dict(primary_gain_positive=gain>0,primary_r0_gain_at_least_5pct=(gain>=Fraction(1,20) if r==0 else True),
            no_scene_regression=all(row["no_regression"] for row in scenes),accuracy_component_safe=f["A"]<=Fraction(51,50)*b["A"],
            completeness_component_safe=f["C"]<=Fraction(51,50)*b["C"],alternate_reference_no_regression=af["S"]<=ab["S"],
            all_artifact_fields_finite=all(row["validity"][field]==0 for row in all_rows for field in ("main_nonfinite_fields","native_nonfinite_fields")),
            candidate_fixed_mask_main_depth_positive=all(row["validity"]["main_nonpositive_depth"]==0 for row in all_rows if row["arm"]==nominee))
        delta=score(detached,r)["S"]-f["S"]
        contrasts.append(delta)
        defined=[Fraction(row["ratio"]["exact"]) for row in scenes if row["ratio"] is not None]
        decisions.append(dict(repeat=r,selected_control=best,primary_gain=full.scalar(gain),primary_nominee_S=full.scalar(f["S"]),
            primary_control_S=full.scalar(b["S"]),scenes=scenes,
            worst_defined_scene_ratio=full.scalar(max(defined)) if defined else None,undefined_scene_ratios=3-len(defined),
            accuracy_component=dict(nominee=full.scalar(f["A"]),control=full.scalar(b["A"]),limit=full.scalar(Fraction(51,50)*b["A"])),
            completeness_component=dict(nominee=full.scalar(f["C"]),control=full.scalar(b["C"]),limit=full.scalar(Fraction(51,50)*b["C"])),
            alternate_reference=dict(reference=full.REFERENCES[1],nominee_S=full.scalar(af["S"]),control_S=full.scalar(ab["S"])),
            validity=[dict(state_id=row["state_id"],reference=row["reference"],**row["validity"]) for row in all_rows],
            checks=checks,pass_all=all(checks.values()),matched_detached_delta_S=full.scalar(delta),
            necessary_matched_detached_positive=delta>0))
    mechanism=None
    if repeat==2:
        f_floor=max(abs(score(nominee,r)["S"]-score(nominee,0)["S"]) for r in repeats)
        d_floor=max(abs(score(detached,r)["S"]-score(detached,0)["S"]) for r in repeats)
        floor,margin=f_floor+d_floor,Fraction(1,1000000)
        checks=dict(full_beats_detached_all_repeats=all(value>0 for value in contrasts),
                    r0_contrast_exceeds_execution_floor_plus_margin=contrasts[0]>floor+margin)
        mechanism=dict(detached_arm=detached,contrasts=[dict(repeat=r,delta_S=full.scalar(value)) for r,value in zip(repeats,contrasts)],
            nominee_execution_floor_term=full.scalar(f_floor),detached_execution_floor_term=full.scalar(d_floor),
            execution_floor=full.scalar(floor),required_strict_margin=full.scalar(margin),
            r0_contrast_minus_floor_and_margin=full.scalar(contrasts[0]-floor-margin),checks=checks,pass_all=all(checks.values()))
    passed=all(row["pass_all"] and row["necessary_matched_detached_positive"] for row in decisions) and (mechanism is None or mechanism["pass_all"])
    confirmed=mechanism["pass_all"] if mechanism is not None else None
    result=dict(schema="profiled_m3_cell_precheck_v1",registration_sha256=full.canonical_sha256(registration),
        provenance=deepcopy(registration["provenance"]),policy=full.policy(),cell=cell,repeat=repeat,nominee=nominee,
        matched_detached_arm=detached,selected_control=best,
        stage_seals=deepcopy(stage_seals),prior_r0_semantic_sha256=full.canonical_sha256(prior_r0) if prior_r0 is not None else None,
        raw_rows_semantic_sha256=full.canonical_sha256(ordered),raw_metric_rows=ordered,
        primary_r0_control_pool={arm:full.scalar(score(arm,0)["S"]) for arm in full.CONTROLS},
        repetitions=decisions,mechanism=mechanism,mechanism_confirmed=confirmed,
        scene_table=table,aggregate_table=[dict(cell=cell,repeat=r,reference=reference,endpoint=kind,arm=arm,
            **{field:full.scalar(value) for field,value in values.items()}) for (r,reference,kind,arm),values in sorted(aggregate.items())],
        pass_precheck=passed,cell_decision=dict(status="passed_precheck" if passed else "failed",selected_control=best,mechanism_confirmed=confirmed),
        required_rows_this_cell=30*(repeat+1),observed_rows=len(rows),full_study_pass=False,rental_authorized=False,launch_authorized=False,
        early_mechanism_rule="positive contrasts only; fixed1e-6 margin and execution floor deferred until r2",
        arrays_loaded=False,external_byte_verification_performed=False,
        limitation="Declared cell evidence only. Controller must verify full stage seals and external receipts; full360-state gate remains mandatory.")
    result["report_semantic_sha256"]=full.canonical_sha256(result)
    return result
