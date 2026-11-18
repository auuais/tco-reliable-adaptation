"""Pure, prospective M3 aggregation; never load predictions or dense references.

This module is not a registered dense scorer and does not authorize study-score
access. A future sealed evaluator must supply already computed scene metrics and
verify the underlying byte hashes. ``evaluate(registration, packet)`` checks their
pairing and applies the November 18 M3 preparation contract. Malformed, missing,
nonfinite or mismatched evidence raises ``ValueError``; valid evidence receives a
complete passing/failed gate report. No incomplete inventory can pass.

Registration schema ``profiled_m3_gate_registration_v1`` contains ``nominee``;
``provenance`` hashes (M2 seal, scores, nomination rule, preparation document,
execution registry, evaluation registry, prediction seal, runtime, checkpoint,
initialization contract); the fixed ``policy()``; ``scenes`` (RGB/clean/noisy pose
hashes and noise seed); ``observations`` keyed scene/side/grid (packet/folds/native
rows hashes); and ``references`` keyed scene/reference (archive/mask hashes and
positive mask count). The future wrapper must bind this exact registration.
Both reference labels share the same archived mask hash and count per scene.

Packet schema ``profiled_m3_scene_metrics_v1`` binds its canonical registration
SHA256 and contains 720 rows: eight cells * three repeats * three scenes * five
arms * two references. Each row has scene/side/grid/pose/repeat/arm/reference,
canonical state_id, identities matching that registration, a unique per-state
process_receipt_sha256, endpoint_sha256 for final/native, metrics_sha256, final
and native {accuracy, completeness} in meters, and validity counts. The validity
fields are mask_count, evaluated_mask_count, main_nonpositive_depth,
native_nonpositive_depth, main_nonfinite_fields and native_nonfinite_fields.
Counts never alter a mask. Optional frozen_controls binds previously selected
primary-r0 controls; it must match recomputation exactly in all eight cells.

Comparisons use exact fractions of the supplied decimal representations, with
no tolerance or 12-place rounding. This preserves the specified inclusive 5%
boundary, strict repeat/mechanism inequalities and exact numeric-arm ties. Raw
meters, normalized values and exact rational representations remain auditable.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
import math
import re


SCENES = ("heads", "chess", "fire")
SIDES = ("earlier", "later")
GRIDS = ("centered", "shifted")
POSES = ("clean", "noisy")
REPEATS = (0, 1, 2)
CONTROLS = ("P0", "P1", "P2", "P5")
REFERENCES = ("registered_rgb", "released")
ENDPOINTS = ("final", "native")
PROVENANCE = ("m2_prediction_seal_sha256", "m2_scores_sha256", "nomination_rule_sha256",
              "preparation_document_sha256", "execution_registry_sha256",
              "evaluation_registry_sha256", "prediction_seal_sha256", "runtime_sha256",
              "original_checkpoint_sha256", "initialization_contract_sha256")
PAIR_HASHES = ("rgb_sha256", "pose_sha256", "packet_sha256", "folds_sha256", "native_rows_sha256",
               "reference_sha256", "mask_sha256", "runtime_sha256", "original_checkpoint_sha256",
               "initialization_contract_sha256")
VALIDITY = ("mask_count", "evaluated_mask_count", "main_nonpositive_depth", "native_nonpositive_depth",
            "main_nonfinite_fields", "native_nonfinite_fields")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy():
    """Exact prospective choices; callers cannot relax thresholds or inventory."""
    return dict(version="profiled_m3_dense_gate_v1", scenes=list(SCENES), sides=list(SIDES),
        grids=list(GRIDS), poses=list(POSES), repetitions=list(REPEATS), control_pool=list(CONTROLS),
        references=list(REFERENCES), primary_reference=REFERENCES[0], gate_endpoint="final",
        reported_endpoints=list(ENDPOINTS), optimizer_seed=42, noise_seed_base=20261112,
        noise_draws_per_scene=1, rotation_degrees=5, translation_diameter_fraction="0.02",
        scene_weights="equal", normalization="same_cell_scene_repeat_reference_P0",
        control_selection="primary_reference_repeat0_then_numeric_arm_exact_tie",
        minimum_primary_r0_gain="0.05", minimum_other_repeat_gain="strictly_positive",
        maximum_scene_ratio="1", maximum_component_ratio="1.02", maximum_alternate_ratio="1",
        mechanism_margin_absolute_S="0.000001",
        mechanism_floor="max_r_abs(F_r-F_0)+max_r_abs(D_r-D_0)",
        fixed_mask_policy="never_filter_by_predictions", main_nonpositive_allowed=0,
        arithmetic="exact_fraction_of_input_decimal", required_states=360, required_rows=720)


def canonical_sha256(value):
    try:
        data=json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Evidence must be finite JSON data") from error
    return hashlib.sha256(data).hexdigest()


def digest(value, context):
    require(isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None,
            "Invalid SHA256: "+context)
    return value


def number(value, context):
    try:
        valid=type(value) in (int,float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid=False
    require(valid,"Expected finite nonnegative metric: "+context)
    return Fraction(str(value))


def count(value, context):
    require(type(value) is int and value >= 0,"Expected nonnegative integer count: "+context)
    return value


def cell_id(side,grid,pose):
    return "_".join((side,grid,pose))


def state_id(scene,side,grid,pose,repeat,arm):
    return "_".join((scene,side,grid,pose,arm,"r"+str(repeat)))


def mean(values):
    return sum(values,Fraction(0))/len(values)


def scalar(value):
    try:
        approximate=float(value)
    except OverflowError as error:
        raise ValueError("Derived gate quantity is not finite in the report representation") from error
    require(math.isfinite(approximate),"Derived gate quantity is not finite in the report representation")
    return dict(value=approximate, exact=str(value))


def validate_registration(registration):
    require(registration.get("schema")=="profiled_m3_gate_registration_v1","Wrong M3 gate registration schema")
    require(registration.get("nominee") in ("P3","P4"),"Nominee must come from the unchanged M2 P3/P4 rule")
    require(registration.get("policy")==policy(),"M3 prospective policy differs")
    provenance=registration.get("provenance",{})
    require(set(provenance)==set(PROVENANCE),"Incomplete M3 provenance inventory")
    for key,value in provenance.items():
        digest(value,key)
    scenes=registration.get("scenes",{})
    require(set(scenes)==set(SCENES),"Wrong M3 scene inventory")
    for index,scene in enumerate(SCENES):
        spec=scenes[scene]
        require(set(spec)=={"rgb_sha256","clean_pose_sha256","noisy_pose_sha256","noise_seed"},
                "Wrong scene identity fields: "+scene)
        require(type(spec["noise_seed"]) is int and spec["noise_seed"]==20261112+index,
                "Wrong fixed pose-noise seed: "+scene)
        for key in ("rgb_sha256","clean_pose_sha256","noisy_pose_sha256"):
            digest(spec[key],scene+"/"+key)
        require(spec["clean_pose_sha256"]!=spec["noisy_pose_sha256"],"Clean and noisy pose packets are identical")
    observations=registration.get("observations",{})
    expected={"/".join(parts) for parts in itertools.product(SCENES,SIDES,GRIDS)}
    require(set(observations)==expected,"Wrong signed/grid observation inventory")
    for key,spec in observations.items():
        require(set(spec)=={"packet_sha256","folds_sha256","native_rows_sha256"},"Wrong observation identity fields: "+key)
        for field,value in spec.items():
            digest(value,key+"/"+field)
    references=registration.get("references",{})
    expected={"/".join(parts) for parts in itertools.product(SCENES,REFERENCES)}
    require(set(references)==expected,"Wrong dense-reference inventory")
    for key,spec in references.items():
        require(set(spec)=={"reference_sha256","mask_sha256","mask_count"},"Wrong reference identity fields: "+key)
        for field in ("reference_sha256","mask_sha256"):
            digest(spec[field],key+"/"+field)
        require(count(spec["mask_count"],key)>0,"Empty scene reference mask: "+key)
    for scene in SCENES:
        first,second=(references[scene+"/"+reference] for reference in REFERENCES)
        require(first["mask_sha256"]==second["mask_sha256"] and first["mask_count"]==second["mask_count"],
                "Both reference conventions must use the same archived scene mask: "+scene)
    return registration["nominee"]


def validate_packet(registration,packet):
    nominee=validate_registration(registration)
    require(packet.get("schema")=="profiled_m3_scene_metrics_v1","Wrong metric packet schema")
    require(packet.get("registration_sha256")==canonical_sha256(registration),"Metric packet belongs to another registration")
    rows=packet.get("rows")
    require(isinstance(rows,list) and len(rows)==720,"M3 requires exactly 720 metric rows / 360 paired states")
    arms=tuple(sorted(CONTROLS+(nominee,),key=lambda a:int(a[1:])))
    expected=set(itertools.product(SCENES,SIDES,GRIDS,POSES,REPEATS,arms,REFERENCES))
    index={}
    states={}
    receipts={}
    for row in rows:
        require(isinstance(row,dict),"Metric row must be an object")
        key=tuple(row.get(field) for field in ("scene","side","grid","pose","repeat","arm","reference"))
        require(type(row.get("repeat")) is int and key in expected,"Unexpected scene/cell/repeat/arm/reference")
        require(key not in index,"Duplicate metric row: "+repr(key))
        scene,side,grid,pose,repeat,arm,reference=key
        sid=state_id(scene,side,grid,pose,repeat,arm)
        require(row.get("state_id")==sid,"State identity disagrees with row pairing: "+sid)
        identity=row.get("identity",{})
        require(set(identity)==set(PAIR_HASHES)|{"optimizer_seed","process_receipt_sha256"},"Incomplete identity fields: "+sid)
        scene_spec=registration["scenes"][scene]
        observation=registration["observations"]["/".join((scene,side,grid))]
        reference_spec=registration["references"][scene+"/"+reference]
        want=dict(rgb_sha256=scene_spec["rgb_sha256"],pose_sha256=scene_spec[pose+"_pose_sha256"],
                  **observation,**{field:reference_spec[field] for field in ("reference_sha256","mask_sha256")},
                  **{field:registration["provenance"][field] for field in
                     ("runtime_sha256","original_checkpoint_sha256","initialization_contract_sha256")})
        require(all(identity.get(field)==value for field,value in want.items()),"Unpaired input/reference/runtime identity: "+sid)
        require(type(identity.get("optimizer_seed")) is int and identity["optimizer_seed"]==42,"Wrong optimizer seed: "+sid)
        receipt=digest(identity.get("process_receipt_sha256"),sid+"/process receipt")
        endpoint=row.get("endpoint_sha256",{})
        require(set(endpoint)==set(ENDPOINTS),"Missing main/native endpoint hashes: "+sid)
        for kind,value in endpoint.items():
            digest(value,sid+"/"+kind)
        digest(row.get("metrics_sha256"),sid+"/metric receipt")
        state_proof=dict(identity={key:value for key,value in identity.items()
                                  if key not in ("reference_sha256","mask_sha256")},endpoint_sha256=endpoint)
        require(sid not in states or states[sid]==state_proof,"References used different state/endpoint evidence: "+sid)
        require(receipt not in receipts or receipts[receipt]==sid,"Fresh-process receipt reused by another state")
        states[sid],receipts[receipt]=state_proof,sid
        validity=row.get("validity",{})
        require(set(validity)==set(VALIDITY),"Wrong raw validity-count inventory: "+sid)
        for field,value in validity.items():
            count(value,sid+"/"+field)
        require(validity["mask_count"]==validity["evaluated_mask_count"]==reference_spec["mask_count"],
                "Fixed reference mask changed or prediction pixels were filtered: "+sid)
        for field in ("main_nonpositive_depth","native_nonpositive_depth"):
            require(validity[field]<=validity["mask_count"],"Invalid depth count exceeds reference mask: "+sid)
        for kind in ENDPOINTS:
            metrics=row.get(kind,{})
            require(set(metrics)=={"accuracy","completeness"},"Missing A/C metric components: "+sid)
            for field,value in metrics.items():
                number(value,sid+"/"+kind+"/"+field)
        index[key]=row
    require(set(index)==expected and len(states)==360,"Missing paired M3 state")
    # Both labels use the same output and shared mask, so every validity count
    # (including depth positivity) must be identical, not merely finiteness.
    for sid in states:
        pair=[row for row in rows if row["state_id"]==sid]
        require(pair[0]["validity"]==pair[1]["validity"],
                "Shared-mask validity counts differ across references: "+sid)
    return nominee,arms,index


def evaluate(registration,packet):
    """Return the full eight-cell gate or reject incomplete/inconsistent evidence.

    The caller remains responsible for prospective execution registration, M2
    nomination verification, byte-level sealing and metric kernel correctness.
    A passing object alone is never authorization for rental or score release.
    """
    nominee,arms,index=validate_packet(registration,packet)
    detached={"P3":"P2","P4":"P5"}[nominee]
    normalized={}
    table=[]
    aggregate={}
    for side,grid,pose,repeat,reference,kind in itertools.product(SIDES,GRIDS,POSES,REPEATS,REFERENCES,ENDPOINTS):
        for arm in arms:
            values=[]
            for scene in SCENES:
                key=(scene,side,grid,pose,repeat,arm,reference)
                row=index[key]
                p0=index[(scene,side,grid,pose,repeat,"P0",reference)]
                a,c=(number(row[kind][field],field) for field in ("accuracy","completeness"))
                a0,c0=(number(p0[kind][field],"P0 "+field) for field in ("accuracy","completeness"))
                require(a0>0 and c0>0,"Zero P0 denominator: "+repr((scene,side,grid,pose,repeat,reference,kind)))
                an,cn=a/a0,c/c0
                score=(an+cn)/2
                normalized[key+(kind,)]=dict(A=an,C=cn,S=score,A_m=a,C_m=c)
                values.append(normalized[key+(kind,)])
                table.append(dict(state_id=row["state_id"],cell=cell_id(side,grid,pose),scene=scene,repeat=repeat,
                    arm=arm,reference=reference,endpoint=kind,accuracy_m=float(a),completeness_m=float(c),
                    P0_accuracy_m=float(a0),P0_completeness_m=float(c0),accuracy_normalized=scalar(an),
                    completeness_normalized=scalar(cn),S=scalar(score),validity=deepcopy(row["validity"])))
            aggregate[(side,grid,pose,repeat,reference,kind,arm)]={
                field:mean([value[field] for value in values]) for field in ("A","C","S","A_m","C_m")}
    cells=[]
    chosen={}
    for side,grid,pose in itertools.product(SIDES,GRIDS,POSES):
        cell=cell_id(side,grid,pose)
        def score(arm,repeat,reference=REFERENCES[0],kind="final"):
            return aggregate[(side,grid,pose,repeat,reference,kind,arm)]
        best=min(CONTROLS,key=lambda arm:(score(arm,0)["S"],int(arm[1:])))
        chosen[cell]=best
        repeats=[]
        for repeat in REPEATS:
            f,b=score(nominee,repeat),score(best,repeat)
            require(b["S"]>0,"Zero selected-control primary score: "+cell)
            gain=1-f["S"]/b["S"]
            scenes=[]
            for scene in SCENES:
                f_scene=normalized[(scene,side,grid,pose,repeat,nominee,REFERENCES[0],"final")]["S"]
                b_scene=normalized[(scene,side,grid,pose,repeat,best,REFERENCES[0],"final")]["S"]
                # Zero control scene error permits only zero candidate scene error;
                # its ratio is undefined and is reported, never silently omitted.
                ratio=f_scene/b_scene if b_scene else None
                scenes.append(dict(scene=scene,nominee_S=scalar(f_scene),control_S=scalar(b_scene),
                    ratio=scalar(ratio) if ratio is not None else None,
                    ratio_status="defined" if ratio is not None else "zero_control_scene_score",
                    no_regression=f_scene<=b_scene))
            all_rows=[index[(scene,side,grid,pose,repeat,arm,reference)]
                      for scene,arm,reference in itertools.product(SCENES,arms,REFERENCES)]
            validity=[dict(state_id=row["state_id"],reference=row["reference"],**row["validity"]) for row in all_rows]
            finite=all(row["validity"][field]==0 for row in all_rows
                       for field in ("main_nonfinite_fields","native_nonfinite_fields"))
            positive=all(row["validity"]["main_nonpositive_depth"]==0 for row in all_rows if row["arm"]==nominee)
            alt_f,alt_b=score(nominee,repeat,REFERENCES[1]),score(best,repeat,REFERENCES[1])
            checks=dict(primary_gain_positive=gain>0,primary_r0_gain_at_least_5pct=(gain>=Fraction(1,20) if repeat==0 else True),
                no_scene_regression=all(row["no_regression"] for row in scenes),
                accuracy_component_safe=f["A"]<=Fraction(51,50)*b["A"],
                completeness_component_safe=f["C"]<=Fraction(51,50)*b["C"],
                alternate_reference_no_regression=alt_f["S"]<=alt_b["S"],
                all_artifact_fields_finite=finite,candidate_fixed_mask_main_depth_positive=positive)
            defined=[row["ratio"] for row in scenes if row["ratio"] is not None]
            worst=max((Fraction(item["exact"]) for item in defined),default=None)
            repeats.append(dict(repeat=repeat,selected_control=best,primary_gain=scalar(gain),
                primary_nominee_S=scalar(f["S"]),primary_control_S=scalar(b["S"]),
                scenes=scenes,worst_defined_scene_ratio=scalar(worst) if worst is not None else None,
                undefined_scene_ratios=sum(row["ratio"] is None for row in scenes),
                accuracy_component=dict(nominee=scalar(f["A"]),control=scalar(b["A"]),limit=scalar(Fraction(51,50)*b["A"])),
                completeness_component=dict(nominee=scalar(f["C"]),control=scalar(b["C"]),limit=scalar(Fraction(51,50)*b["C"])),
                alternate_reference=dict(reference=REFERENCES[1],nominee_S=scalar(alt_f["S"]),control_S=scalar(alt_b["S"])),
                validity=validity,checks=checks,pass_all=all(checks.values())))
        contrasts=[score(detached,r)["S"]-score(nominee,r)["S"] for r in REPEATS]
        f_floor=max(abs(score(nominee,r)["S"]-score(nominee,0)["S"]) for r in REPEATS)
        d_floor=max(abs(score(detached,r)["S"]-score(detached,0)["S"]) for r in REPEATS)
        floor=f_floor+d_floor
        margin=Fraction(1,1000000)
        mechanism_checks=dict(full_beats_detached_all_repeats=all(value>0 for value in contrasts),
                              r0_contrast_exceeds_execution_floor_plus_margin=contrasts[0]>floor+margin)
        mechanism=dict(detached_arm=detached,contrasts=[dict(repeat=r,delta_S=scalar(value)) for r,value in zip(REPEATS,contrasts)],
            nominee_execution_floor_term=scalar(f_floor),detached_execution_floor_term=scalar(d_floor),
            execution_floor=scalar(floor),required_strict_margin=scalar(margin),
            r0_contrast_minus_floor_and_margin=scalar(contrasts[0]-floor-margin),
            checks=mechanism_checks,pass_all=all(mechanism_checks.values()))
        cells.append(dict(cell=cell,side=side,grid=grid,pose=pose,selected_control=best,
            primary_r0_control_pool={arm:scalar(score(arm,0)["S"]) for arm in CONTROLS},
            repetitions=repeats,mechanism=mechanism,
            pass_all=all(row["pass_all"] for row in repeats) and mechanism["pass_all"]))
    if "frozen_controls" in packet:
        require(packet["frozen_controls"]==chosen,"Previously frozen primary-r0 controls changed")
    aggregates=[dict(cell=cell_id(side,grid,pose),repeat=repeat,reference=reference,endpoint=kind,arm=arm,
                     **{field:scalar(value) for field,value in values.items()})
                for (side,grid,pose,repeat,reference,kind,arm),values in sorted(aggregate.items())]
    passed=all(cell["pass_all"] for cell in cells)
    return dict(schema="profiled_m3_gate_report_v1",policy=policy(),registration_sha256=canonical_sha256(registration),
        input_packet_sha256=canonical_sha256(packet),nominee=nominee,matched_detached_arm=detached,
        provenance=deepcopy(registration["provenance"]),required_states=360,observed_states=360,
        required_rows=720,observed_rows=len(packet["rows"]),status="passed" if passed else "failed",
        pass_all=passed,conditional_allocation_gate_passed=passed,failed_cells=[cell["cell"] for cell in cells if not cell["pass_all"]],
        frozen_controls=chosen,cells=cells,scene_table=table,aggregate_table=aggregates,
        raw_metric_rows=deepcopy(packet["rows"]),aggregation_only=True,arrays_loaded=False,
        execution_registration_and_sealing_implemented=False,release_authorized=False,
        limitation="Local development gate only; one fixed pose draw and three nonindependent same-scene execution repetitions. "
                   "This pure aggregator verifies declared identities, not source bytes, metric kernels or M2 nomination legitimacy.")
