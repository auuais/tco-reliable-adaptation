"""Bind the deterministic same-state check to the five-view study phases."""

import argparse
from pathlib import Path
from unittest.mock import patch

import check_tco_five_view_replay_cache as design
from run_profiled_d0 import read, sha, write_new
from tco_five_view_replay_cache_protocol import PUBLIC, ROOT, job


ANCHOR = "grad_deterministic_capture"
CHECK_SOURCE = "scripts/qualify_five_view_deterministic.py"
EXTENSION_SOURCE = Path(__file__).resolve().relative_to(ROOT).as_posix()
REGISTRATION = PUBLIC / "deterministic_extension.json"


def deterministic_job(policy, configuration=None):
    return job(f"det_match_{policy}", "deterministic_qualification", policy, steps=2, width=294,
               probe="real", backend="deterministic", gradient_mode="replay", source_job=ANCHOR,
               **(configuration or {}))


def structurally_valid(policy, production):
    controls = production["policy_checks"].get(policy, {}).get("controls", {})
    return bool(controls) and all(controls.values())


def profile_camera(rows):
    production = read(PUBLIC / "qualification.json")
    for policy in ("last", "camera"):
        if structurally_valid(policy, production) and all(
            f"baseline_{window}_{policy}_{replica}" in rows
            for window in design.WINDOWS for replica in (1, 2)
        ):
            return policy
    return "camera"


def augmented_read(descriptive):
    def load(path):
        result = read(path)
        sources = {
            "qualification.json": "deterministic_screen_qualification.json",
            "retention_qualification.json": "deterministic_development_qualification.json",
        }
        if path.name in sources:
            supplement = read(PUBLIC / sources[path.name])
            allowed = set(result["eligible"]) | set(supplement["eligible"])
            if descriptive:
                allowed.update(policy for policy in ("camera", "last") if structurally_valid(policy, result))
            return {**result, "eligible": sorted(allowed)}
        return result
    return load


def extend(phase):
    sources = {name: sha(ROOT / name, text=True) for name in (EXTENSION_SOURCE, CHECK_SOURCE)}
    if phase == "register":
        write_new(REGISTRATION, dict(
            source_sha256_lf=sources,
            rationale="A new ordinary Native replay crossed the three-repeat envelope. The supplementary "
                      "check uses the already qualified deterministic native backend's own real-gradient anchor.",
            anchor=ANCHOR, native_reference="grad_deterministic_fallback_fixed_1",
            preserve_production_status=True, reconstruction_margin=1.01, rental_compute_usd=0,
        ))
        return
    if read(REGISTRATION)["source_sha256_lf"] != sources:
        raise ValueError("Deterministic extension source changed")
    original = design.authority

    def write(name, parent, evidence, extra_authorities=(), **payload):
        extra = [REGISTRATION.name]
        if name == "screen":
            payload["jobs"][:0] = [deterministic_job("camera"), deterministic_job("last")]
            payload["deterministic_extension"] = (
                "The production repeat envelope was crossed by unchanged Native. Retain the production verdict "
                "and test the camera arms using identical actual q at the separate verified deterministic anchor."
            )
        elif name == "development":
            extra.append("deterministic_screen_qualification.json")
            configs = {row["policy"]: {key: row[key] for key in ("camera", "retention", "sites", "operation")}
                       for row in payload["jobs"] if row["cohort"] == "qualification"}
            payload["jobs"][:0] = [deterministic_job(policy, config) for policy, config in configs.items()]
            payload["deterministic_extension"] = "Apply the same deterministic check to the selected retention sites."
        elif name == "locked":
            extra += ["deterministic_screen_qualification.json", "deterministic_development_qualification.json"]
        return original(name, parent, evidence, extra_authorities=(*extra_authorities, *extra), **payload)

    if phase == "screen":
        with patch.multiple(design, authority=write, camera_arm=profile_camera):
            design.screen()
    else:
        with patch.multiple(design, authority=write, read=augmented_read(descriptive=phase == "development")):
            {"development": design.development, "lock": design.lock}[phase]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "screen", "development", "lock"))
    extend(parser.parse_args().phase)
