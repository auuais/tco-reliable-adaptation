"""Independent replay of saved public M2 compact held-source arithmetic.

This module does not import the original scorer, reconstruct overlap masks,
decode full predictions, verify model repeatability, or authorize more research.
The real entry point requires existing report/access/seal/registration linkage
before decoding the compact NPZ. It preserves a new receipt and never overwrites
scores. All arithmetic comparisons use prospectively fixed 1e-12 tolerances;
nomination, eligibility, inventories and invalidity counts must agree exactly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("heads", "chess", "fire")
SIDES = ("earlier", "later")
ARMS = tuple("P" + str(i) for i in range(6))
FRAMES = (25, 225, 425, 625, 825)
ENDPOINTS = ("final", "native")
RTOL = ATOL = 1e-12
SOURCES = ("scripts/replay_profiled_public_scores.py", "scripts/test_replay_profiled_public_scores.py")
FAILURE_POLICY = {
    "automatic_numerical_error_types": ["FloatingPointError"],
    "unclassified_errors": "block all study-score release pending engineering review",
    "infrastructure_errors": "no automatic negative scientific verdict; any disposition must be frozen before scores",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path_at(root, name):
    require(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
            "Expected repository-relative POSIX path")
    parts = name.split("/")
    require(all(part not in ("", ".", "..") and not re.match(r"^[A-Za-z]:", part) for part in parts),
            "Unsafe artifact path")
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact path escaped repository")
    return path


def parse(data):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def bound_json(root, name, expected=None):
    data = path_at(root, name).read_bytes()
    if expected is not None:
        require(sha(data) == expected, "Changed linked JSON bytes: " + name)
    return parse(data), sha(data)


def close(actual, expected, where="report"):
    """Counts/booleans/identities exact; floating arithmetic within fixed bounds."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Wrong fields: " + where)
        for key in expected:
            close(actual[key], expected[key], where + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Wrong list: " + where)
        for index, (first, second) in enumerate(zip(actual, expected)):
            close(first, second, where + "/" + str(index))
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), "Arithmetic mismatch: " + where)
    else:
        require(type(actual) is type(expected) and actual == expected, "Exact mismatch: " + where)


def mean(values):
    return None if any(value is None for value in values) else math.fsum(value / len(values) for value in values)


def metric(values, measured):
    errors = []
    nonfinite = nonpositive = overflow = 0
    for prediction, truth in zip(values.tolist(), measured.tolist()):
        if not math.isfinite(prediction):
            nonfinite += 1
            continue
        nonpositive += prediction <= 0
        error = abs(prediction - truth)
        if math.isfinite(error):
            errors.append(error)
        else:
            overflow += 1
    finite_mean = mean(errors) if errors else None
    return dict(count=len(measured), nonfinite_count=nonfinite, nonpositive_count=nonpositive,
                mae_m=finite_mean if not nonfinite and not overflow else None,
                finite_error_count=len(errors), finite_error_mae_m=finite_mean,
                unrepresentable_error_count=overflow,
                all_samples_valid=not (nonfinite or nonpositive or overflow), candidate_dependent_exclusions=0)


def sample_fields():
    fields = set()
    for scene, side, frame in itertools.product(SCENES, SIDES, FRAMES):
        prefix = f"{scene}_{side}_{frame}__"
        fields.update(prefix + suffix for suffix in ("xy", "measured_m"))
        fields.update(prefix + arm + "_" + endpoint + "_m" for arm, endpoint in itertools.product(ARMS, ENDPOINTS))
    return fields


def independent_summary(rows, runtimes, failures):
    lookup = {(row["scene"], row["side"], row["arm"], row["frame"]): row for row in rows}
    cases, directional, validity = {}, {}, {}
    for endpoint in ENDPOINTS:
        cases[endpoint], validity[endpoint], directional[endpoint] = {}, {}, {}
        for scene, side, arm in itertools.product(SCENES, SIDES, ARMS):
            views = [lookup[scene, side, arm, frame][endpoint] for frame in FRAMES]
            key = f"{scene}_{side}_{arm}"
            cases[endpoint][key] = mean([view["mae_m"] for view in views])
            validity[endpoint][key] = all(view["all_samples_valid"] for view in views)
        for side in SIDES:
            directional[endpoint][side] = {arm: mean([cases[endpoint][f"{scene}_{side}_{arm}"]
                                                     for scene in SCENES]) for arm in ARMS}
    candidates = {}
    for arm in ("P3", "P4"):
        rejected, gains, improving = [], {}, {}
        failure_count = sum(failures[f"{scene}_{side}_{arm}"] for scene, side in itertools.product(SCENES, SIDES))
        if failure_count:
            rejected.append("reported_numerical_failure")
        for side in SIDES:
            baseline, candidate = directional["final"][side]["P1"], directional["final"][side][arm]
            valid_control = all(validity["final"][f"{scene}_{side}_P1"] for scene in SCENES)
            if baseline is None or baseline <= 0 or not valid_control or candidate is None:
                gains[side] = None
                rejected.append("invalid_primary_metric_or_P1_control_" + side)
            else:
                gains[side] = 1.0 - candidate / baseline
                if gains[side] < .01:
                    rejected.append("directional_gain_below_0.01_" + side)
        if not all(validity["final"][f"{scene}_{side}_{arm}"] for scene, side in itertools.product(SCENES, SIDES)):
            rejected.append("invalid_primary_predictions")
        for side in SIDES:
            improving[side] = []
            for scene in SCENES:
                candidate, baseline = (cases["final"][f"{scene}_{side}_{value}"] for value in (arm, "P1"))
                if candidate is not None and baseline is not None and candidate < baseline:
                    improving[side].append(scene)
        common = sorted(set(improving["earlier"]).intersection(improving["later"]))
        if len(common) < 2:
            rejected.append("fewer_than_two_same_scenes_improve_both_directions")
        minimum = None if None in gains.values() else min(gains.values())
        candidates[arm] = dict(eligible=not rejected, reasons=rejected, directional_gain=gains,
            improving_scenes=improving, same_scenes_improving_both=common, numerical_failed_attempts=failure_count,
            minimum_directional_gain=minimum, ranking_gain_rounded_12=None if minimum is None else round(minimum,12),
            total_measured_seconds_six_cases=math.fsum(runtimes[f"{scene}_{side}_{arm}"]
                                                      for scene, side in itertools.product(SCENES,SIDES)))
    ranking = sorted((arm for arm in candidates if candidates[arm]["eligible"]), key=lambda arm:
        (-candidates[arm]["ranking_gain_rounded_12"], candidates[arm]["total_measured_seconds_six_cases"], int(arm[1:])))
    contrasts = {}
    for full, detached in (("P3","P2"),("P4","P5")):
        contrasts[full+"_versus_"+detached] = {}
        for side in SIDES:
            a, b = (directional["final"][side][arm] for arm in (full,detached))
            valid = all(validity["final"][f"{scene}_{side}_{arm}"] for scene,arm in itertools.product(SCENES,(full,detached)))
            contrasts[full+"_versus_"+detached][side] = dict(valid=valid,
                relative_gain=None if not valid or a is None or b is None or b <= 0 else 1-a/b,
                absolute_mae_difference_m=None if a is None or b is None else a-b)
    return dict(case_mae_m=cases,directional_equal_scene_mae_m=directional,case_validity=validity,
                candidates=candidates,nominee=ranking[0] if ranking else None,candidate_ranking=ranking,
                mechanism_contrasts=contrasts,primary_endpoint="final",secondary_endpoint="native",
                accuracy_improvement_established=False,publication_test_performed=False,
                mechanism_repeatability_established=False,rental_gate_passed=False)


def validate_arithmetic(report, samples):
    """Pure compact-array verification; never reconstruct source-mask selection."""
    require(isinstance(samples,dict) and set(samples)==sample_fields() and len(samples)==420,
            "Expected exactly420 compact sample fields")
    rows=report.get("rows")
    require(isinstance(rows,list) and len(rows)==180,"Expected180 scored view/arm rows")
    expected=set(itertools.product(SCENES,SIDES,ARMS,FRAMES)); index={}
    run_ids={f"{s}_{d}_{a}" for s,d,a in itertools.product(SCENES,SIDES,ARMS)}
    for field in ("runtimes_seconds","numerical_failures_by_run_id","all_failed_attempt_counts"):
        require(isinstance(report.get(field),dict) and set(report[field])==run_ids,"Wrong run inventory: "+field)
        for value in report[field].values():
            if field=="runtimes_seconds":
                require(type(value) in (int,float) and math.isfinite(value) and value>=0,"Invalid measured runtime")
            else:
                require(type(value) is int and value>=0,"Invalid failure count")
    counts_by_view={}; recomputed=[]
    for row in rows:
        require(isinstance(row,dict),"Expected metric row")
        key=tuple(row.get(k) for k in ("scene","side","arm","frame"))
        require(type(row.get("frame")) is int and key in expected and key not in index,"Missing/duplicate scored row")
        index[key]=row
        scene,side,arm,frame=key; prefix=f"{scene}_{side}_{frame}__"
        xy,measured=samples[prefix+"xy"],samples[prefix+"measured_m"]
        require(isinstance(xy,np.ndarray) and xy.dtype==np.int64 and xy.ndim==2 and xy.shape[1]==2 and len(xy)>0,
                "Invalid compact native XY")
        require(((xy>=[0,0])&(xy<[512,384])).all() and len(set(map(tuple,xy.tolist())))==len(xy),
                "Duplicate/out-of-bounds compact pixels")
        require(isinstance(measured,np.ndarray) and measured.dtype==np.float64 and measured.shape==(len(xy),)
                and np.isfinite(measured).all() and (measured>0).all(),"Invalid compact observations")
        count_keys=("mapped_opposite_count","overlapping_excluded","held_count","adaptation_count")
        counts={k:row.get(k) for k in count_keys}
        require(all(type(v) is int and v>=0 for v in counts.values()) and counts["adaptation_count"]>0,
                "Invalid fixed-mask accounting")
        require(counts["held_count"]==len(xy) and counts["mapped_opposite_count"]==len(xy)+counts["overlapping_excluded"]
                and counts["overlapping_excluded"]<=counts["adaptation_count"],"Inconsistent held-mask counts")
        view_key=(scene,side,frame)
        require(view_key not in counts_by_view or counts_by_view[view_key]==counts,"Candidate-dependent mask accounting")
        counts_by_view[view_key]=counts
        computed={k:row[k] for k in ("scene","side","arm","frame")}; computed.update(counts)
        for endpoint in ENDPOINTS:
            values=samples[prefix+arm+"_"+endpoint+"_m"]
            require(isinstance(values,np.ndarray) and values.dtype==np.float64 and values.shape==measured.shape,
                    "Wrong compact prediction dtype/count")
            computed[endpoint]=metric(values,measured)
        close(row,computed,"row/"+str(key));recomputed.append(computed)
    require(set(index)==expected,"Missing scored row")
    summary=independent_summary(recomputed,report["runtimes_seconds"],report["numerical_failures_by_run_id"])
    for key,value in summary.items():
        close(report.get(key),value,key)
    return summary


def validate_rules(rules):
    require(rules.get("schema")=="profiled_evaluation_rules_v1" and rules.get("rules_version")==2,
            "Expected unchanged evaluation rules v2")
    require(rules.get("candidate_arms")==["P3","P4"] and rules.get("control_arms")==["P0","P1","P2","P5"],
            "Wrong nominated/control arms")
    n=rules["nomination"]
    require((n["minimum_directional_gain_each"],n["minimum_same_scenes_improving_both_directions"],
             n["numerical_failures_allowed"],n["max_candidates"])==(.01,2,0,1),"Changed nomination thresholds")
    require(n["ranking"]=="largest minimum directional gain, rounded to 12 decimal places; then lower total measured runtime across six cases; then smaller numeric arm ID",
            "Changed nomination tie rule")
    require(rules["global_prediction_seal"]["required_states"]==36 and
            rules["held_source_metric"]["candidate_independent_mask"] is True,"Changed inventory or mask policy")


def linked_evidence(root,report_file):
    report,report_hash=bound_json(root,report_file)
    require(report.get("schema")=="profiled_held_source_scores_v1","Wrong saved report schema")
    seal,seal_hash=bound_json(root,report["prediction_seal"],report["prediction_seal_sha256"])
    access,access_hash=bound_json(root,report["score_access_receipt"],report["score_access_receipt_sha256"])
    identity=report["evaluation_execution"]
    require(set(identity)=={"file","sha256"},"Invalid evaluation descriptor")
    evaluation,_=bound_json(root,identity["file"],identity["sha256"])
    require(seal.get("schema")=="profiled_global_prediction_seal_v1" and seal.get("pass_all") is True
            and seal.get("required_states")==seal.get("completed_states")==36,"Incomplete global seal")
    require(evaluation.get("schema")=="profiled_evaluation_execution_v1" and evaluation.get("synthetic_fixtures_pass") is True
            and evaluation.get("study_arrays_loaded") is False and evaluation.get("study_scores_read") is False,
            "Scorer was not prospectively registered")
    require(seal["evaluation_execution"]==identity==access["evaluation_execution"] and
            access["prediction_seal"]==report["prediction_seal"] and access["prediction_seal_sha256"]==seal_hash,
            "Score-access/evaluation/seal linkage differs")
    for item,field in ((evaluation,"prepared_at_utc"),(access,"first_access_at_utc")):
        value=datetime.fromisoformat(item[field].replace("Z","+00:00"))
        require(value.tzinfo is not None,"Receipt timestamp must be timezone-aware")
    require(datetime.fromisoformat(evaluation["prepared_at_utc"].replace("Z","+00:00")) <=
            datetime.fromisoformat(access["first_access_at_utc"].replace("Z","+00:00")),"Registration follows score access")
    rules,rules_hash=bound_json(root,evaluation["rules"],evaluation["rules_sha256"])
    require(rules_hash==report["evaluation_rules_sha256"]==seal["evaluation_rules_sha256"] and
            seal["evaluation_rules"]==evaluation["rules"],"Evaluation-rule byte linkage differs")
    validate_rules(rules)
    registry,_=bound_json(root,seal["common_scientific_registry"],seal["common_scientific_registry_sha256"])
    require(registry.get("phase")=="scientific" and registry.get("scientific_ready_at_registration") is True and
            registry["artifact_sha256"].get(evaluation["rules"])==rules_hash,"Scientific registry does not bind rules")
    for item in (report,seal,evaluation):
        require(item.get("failure_policy")==FAILURE_POLICY,"Failure policy differs")
    require(report.get("failure_review_required") is False and seal.get("failure_review_required") is False,
            "Unclassified failures require review")
    require(report.get("candidate_dependent_masks") is False and report.get("target_depth_labels_loaded") is False
            and report.get("finite_error_means_are_diagnostics_only") is True,"Wrong report scope")
    require(report["split_sha256"]==evaluation["input_sha256"]["results/lora_profiled_v1/splits.npz"],"Split identity differs")
    require(report["source_packet_sha256"]=={side:evaluation["packet_sha256"][f"results/mast3r_partner_shape/{side}.npz"]
                                           for side in SIDES},"Source packet identity differs")
    cases=seal.get("cases");require(isinstance(cases,list) and len(cases)==36,"Wrong seal case count")
    expected={f"{s}_{d}_{a}" for s,d,a in itertools.product(SCENES,SIDES,ARMS)};seen=set()
    for case in cases:
        key=f"{case['scene']}_{case['adaptation_source']}_{case['arm']}"
        require(key in expected and key not in seen and case["run_id"]==key,"Duplicate/unexpected seal case")
        seen.add(key)
        require(case.get("unclassified_failure_count")==0,"Unclassified sealed failure")
        close(report["runtimes_seconds"][key],float(case["selected"]["seconds"]),"sealed runtime")
        close(report["numerical_failures_by_run_id"][key],case["numerical_failure_count"],"sealed numerical failures")
        close(report["all_failed_attempt_counts"][key],case["failed_attempt_count"],"sealed total failures")
    return report,dict(report=dict(file=report_file,sha256=report_hash),
        prediction_seal=dict(file=report["prediction_seal"],sha256=seal_hash),evaluation_execution=identity,
        rules=dict(file=evaluation["rules"],sha256=rules_hash),
        score_access=dict(file=report["score_access_receipt"],sha256=access_hash))


def replay(root,report_file,output_file):
    root=Path(root).resolve();output=path_at(root,output_file)
    require(not output.exists(),"Preserve existing independent replay receipt")
    report,linkage=linked_evidence(root,report_file)
    samples_path=path_at(root,report["replay_samples"]);data=samples_path.read_bytes()
    require(sha(data)==report["replay_samples_sha256"],"Compact sample bytes changed before decode")
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        require(len(archive.files)==420 and set(archive.files)==sample_fields(),"Missing/duplicate compact NPZ fields")
        samples={name:archive[name].copy() for name in archive.files}
    summary=validate_arithmetic(report,samples)
    again,_=linked_evidence(root,report_file)
    require(again==report and sha(samples_path.read_bytes())==report["replay_samples_sha256"],"Evidence changed during replay")
    receipt=dict(schema="profiled_public_arithmetic_replay_v1",pass_all=True,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),linkage=linkage,
        samples=dict(file=report["replay_samples"],sha256=report["replay_samples_sha256"]),
        source_sha256_lf={name:sha(path_at(root,name).read_bytes().replace(b"\r\n",b"\n")) for name in SOURCES},
        dependencies=dict(numpy=np.__version__),tolerance=dict(rtol=RTOL,atol=ATOL),
        views_checked=30,rows_checked=180,fields_checked=420,summary=summary,
        original_scorer_arithmetic_imported=False,raw_endpoints_opened=False,source_masks_reconstructed=False,
        model_repeatability_verified=False,accuracy_improvement_established=False,rental_authorized=False,
        scope="Public compact arithmetic and saved provenance linkage only; not raw-endpoint/mask-selection reconstruction or model repeatability")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write("\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--report",default="results/lora_profiled_v1/held_source_scores.json")
    parser.add_argument("--output",default="results/lora_profiled_v1/independent_public_replay.json")
    args=parser.parse_args();result=replay(args.root,args.report,args.output)
    print(json.dumps(dict(pass_all=result["pass_all"],nominee=result["summary"]["nominee"],rental_authorized=False)))


if __name__=="__main__":
    main()
