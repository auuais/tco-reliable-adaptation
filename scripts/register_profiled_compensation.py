"""Freeze the executable 36-state compensation screen and its evaluator."""
import argparse
import json
import platform

from profiled_compensation_protocol import (ROOT, DESIGN, ENGINEERING, EXECUTION, EVALUATION, PARENT,
    PUBLIC, ENGINEERING_SOURCES, SCIENTIFIC_SOURCES, file_identity, jobs, require, source_maps,
    validate_design, verify_hashes, verify_identity)
from run_profiled_d0 import committed_identity, now, read, sha, write_new

SOURCES = ENGINEERING_SOURCES + SCIENTIFIC_SOURCES + ("scripts/register_profiled_compensation.py",
    "scripts/profiled_compensation_penalty.py", "scripts/test_profiled_compensation_penalty.py",
    "scripts/verify_compensation_public_scores.py")
BRIDGE = f"{PUBLIC}/engineering/bridge01/report.json"
TERMINAL = f"{PUBLIC}/launches/tco-profiled-compensation-bridge01/terminal.json"
PARENT_RULES = "manifests/lora_profiled_evaluation_v2.json"


def score_runtime():
    import numpy
    return dict(python=platform.python_version(), numpy=numpy.__version__)


def require_scoring_runtime(evaluation):
    require(score_runtime() == evaluation["scoring_runtime"],
            "Scoring needs the prospectively recorded CPU Python and NumPy")


def check_bridge():
    bridge = read(ROOT / BRIDGE)
    require(bridge["pass_all"] is True and bridge["engineering"] == file_identity(ENGINEERING)
            and bridge["optimizer_updates"] == 0 and bridge["decoder_calls"] == 184
            and bridge["zero_adapter_exact"] and bridge["original_state_unchanged"]
            and bridge["nonzero_state_restored"], "Engineering bridge did not pass")
    require([row["fold"] for row in bridge["forward_matched"]] == [0,1]
            and all(row["exact"] for row in bridge["forward_matched"])
            and [row["fold"] for row in bridge["parameter_finite_differences"]] == [0,1]
            and all(row["pass_check"] for row in bridge["parameter_finite_differences"])
            and [(row["fold"], row["arm"]) for row in bridge["retained_replay"]]
            == [(0,"R1"),(0,"R2"),(1,"R1"),(1,"R2")], "Engineering comparison coverage differs")
    terminal = read(ROOT / TERMINAL)
    require(terminal["verified_stopped"] is True and terminal["state"]["ExitCode"] == 0,
            "Engineering container completion is unverified")
    return [file_identity(name) for name in (ENGINEERING, BRIDGE, TERMINAL)]


def payload():
    design = validate_design()
    source, upstream = source_maps(SOURCES)
    parent = read(ROOT / PARENT)
    engineering = read(ROOT / ENGINEERING)
    require(all(source.get(name) == digest for name, digest in engineering["source_sha256_lf"].items())
            and upstream == engineering["upstream_sha256_lf"], "Engineering sources differ")
    evidence = check_bridge()
    artifacts = dict(parent["artifact_sha256"])
    for name in (PARENT, PARENT_RULES, DESIGN, ENGINEERING, BRIDGE, TERMINAL):
        artifacts[name] = sha(ROOT / name)
    verify_hashes(artifacts)
    rows = jobs()
    require(len(rows) == 36 and len({row["run_id"] for row in rows}) == 36
            and sum(row["adapt_lora"] for row in rows) == 30, "Unexpected screen size")
    return dict(schema="profiled_compensation_execution_v1", phase="scientific",
        scientific_ready_at_registration=True, design=file_identity(DESIGN), engineering=evidence,
        execution_commit=read(ROOT / BRIDGE)["execution_commit"], source_sha256_lf=source,
        upstream_sha256_lf=upstream, artifact_sha256=artifacts, input_sha256=parent["input_sha256"],
        docker_image=parent["docker_image"], checkpoint_sha256=engineering["checkpoint_sha256"],
        config_sha256=engineering["config_sha256"], runtime_policy=parent["runtime_policy"],
        study_settings=parent["study_settings"], attempt_id="screen01", jobs=rows,
        counts=dict(total=36, trained=30, frozen=6),
        local_limits=dict(max_concurrent_cases=1, host_memory_gib=8, case_seconds=900,
                          max_cuda_allocated_mib=8192, max_cuda_reserved_mib=9216, rental_usd=0),
        estimate=estimate(), evaluation_required_before_first_case=True, comparison=design["comparison"])


def estimate():
    m2 = read(ROOT / "results/lora_profiled_v1/held_source_scores.json")
    runtimes = m2["runtimes_seconds"]
    return dict(method="Sum the 36 corresponding M2 case runtimes. Map each R arm to P3.",
        modeled_wall_seconds=sum(runtimes[f"{j['scene']}_{j['adaptation_source']}_" +
            (j["arm"] if j["arm"].startswith("P") else "P3")] for j in jobs()),
        additional_launch_audit_time_excluded=True)


def verify_release():
    registry = read(ROOT / EXECUTION)
    committed_identity(ROOT, EXECUTION, registry["source_sha256_lf"])
    committed_identity(ROOT, EVALUATION, ())
    require(registry == payload(), "Compensation executable manifest differs")
    evaluation = read(ROOT / EVALUATION)
    require(evaluation["schema"] == "profiled_compensation_evaluation_v1"
            and evaluation["execution"] == file_identity(EXECUTION)
            and evaluation["parent_rules"] == file_identity(PARENT_RULES)
            and evaluation["comparison"] == registry["comparison"]
            and evaluation["synthetic_acceptance"]["pass_all"] is True,
            "Compensation evaluator does not bind the executable, rules or fixture evidence")
    verify_identity(evaluation["synthetic_acceptance"]["test_source"])
    return registry, evaluation


def prepare():
    import unittest
    from test_profiled_compensation_protocol import ScoreTests
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScoreTests))
    require(result.wasSuccessful() and not result.skipped and result.testsRun >= 4,
            "Registered evaluator fixtures did not pass")
    require(not (ROOT / EXECUTION).exists() and not (ROOT / EVALUATION).exists(),
            "Preserve existing compensation release")
    registry = payload()
    write_new(ROOT / EXECUTION, registry)
    item = dict(schema="profiled_compensation_evaluation_v1", execution=file_identity(EXECUTION),
                parent_rules=file_identity(PARENT_RULES), comparison=registry["comparison"],
                scoring_runtime=score_runtime(), created_at_utc=now(),
                synthetic_acceptance=dict(pass_all=True, test_count=result.testsRun,
                    test_source=file_identity("scripts/test_profiled_compensation_protocol.py")),
                first_access_after_global_seal=True)
    write_new(ROOT / EVALUATION, item)
    print(json.dumps(registry["counts"]), json.dumps(registry["estimate"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "verify"))
    arg = p.parse_args()
    prepare() if arg.mode == "prepare" else verify_release()
