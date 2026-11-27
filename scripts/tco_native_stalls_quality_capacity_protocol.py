"""Register the local five-view stall, quality, and ten-view capacity study."""

import argparse
import random

from run_profiled_d0 import read, sha, write_new
from tco_courtyard_camera_protocol import COURTYARD_FRAMES, COURTYARD_IMAGES, TEN_FRAMES, TEN_WINDOWS, hashes_for
from tco_five_view_replay_cache_protocol import INITIAL as PRIOR, LOCKED as PRIOR_LOCK
from tco_full40_recompute_protocol import runtime_pins
from tco_one_percent_protocol import ROOT
from tco_render_lifetime_protocol import job


STUDY = "tco_native_stalls_quality_capacity_v1"
STEM = STUDY.removesuffix("_v1")
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
INITIAL = ROOT / "manifests" / f"{STUDY}.json"
LOCKED = INITIAL.with_name(f"{STEM}_locked_v1.json")
CAPACITY = INITIAL.with_name(f"{STEM}_capacity_v1.json")
REGISTRATIONS = {"initial": INITIAL, "locked": LOCKED, "capacity": CAPACITY}
CPU_PROFILES = {"fixture", "rescore", "input_audit"}
IMAGE = runtime_pins()["docker_image"]
MARGIN = 1.01
SEEDS = (7, 77, 777)
ORDER_SEED = 20261127
SELECTION = ("ng", "n", "g", "camera")
POLICIES = ("native", "camera", "n", "g", "ng")
GSPLAT = "339c0272e544d6a166814cf49e2d21f500eaf9018e69585f1fcaadb5ed3c940e"
FIVE_BASELINE = "54ab68a3c401862e8ee408e12c2f225ab3241945"
SOURCES = (
    "docs/TCO_NATIVE_STALLS_QUALITY_CAPACITY_HANDOFF.md",
    "scripts/tco_native_stalls_quality_capacity_protocol.py",
    "scripts/run_tco_native_stalls_quality_capacity.py",
    "scripts/run_tco_native_stalls_quality_capacity_local.py",
    "scripts/tco_native_stall_trace.py",
    "scripts/replay_tco_native_stalls_quality_capacity.py",
    "scripts/qualify_tco_native_stalls_quality_capacity.py",
    "scripts/run_tco_render_lifetime.py",
    "scripts/tco_render_lifetime.py",
    "scripts/check_tco_render_lifetime.py",
    "scripts/tco_five_view_gradients.py",
    "scripts/run_tco_courtyard_camera_local.py",
    "scripts/run_tco_one_percent.py",
    "scripts/tco_one_percent_geometry.py",
    "scripts/replay_tco_five_view_replay_cache.py",
)
LOCATION_FIELDS = ("dataset", "scene", "frames", "width", "images_expected")
CHESS = {"scene": "chess/seq-03"}
COURTYARD = dict(scene="courtyard", dataset="eth3d", frames=list(COURTYARD_FRAMES),
                 images_expected=list(COURTYARD_IMAGES))


def rescore(source):
    location = {name: source[name] for name in LOCATION_FIELDS if name in source}
    return job(f"rescore_{source['id']}", "rescore", source["policy"], steps=0,
               profile="rescore", probe="rescore", source_job=source["id"], seed=source["seed"], **location)


def repeat_jobs(rng):
    rows = []
    for name, location in (("chess", CHESS), ("courtyard", COURTYARD)):
        for seed in (42, 44):
            block = [job(f"repeat_{name}_{seed}_{label}", "repeat", policy, seed=seed,
                         score=True, replica=label, **location)
                     for label, policy in (("native_a", "native"), ("native_b", "native"), ("camera", "camera"))]
            rng.shuffle(block)
            rows.extend(block)
    traces = [job(f"trace_{name}_{policy}", "trace", policy, seed=44, profile="timeline", score=True, **location)
              for name, location, policy in (("chess", CHESS, "native"), ("chess", CHESS, "camera"),
                                              ("courtyard", COURTYARD, "native"))]
    traces += [job("trace_heads_native", "trace", seed=44, profile="timeline", score=True, conditional="heads"),
               job("trace_chess_focused", "trace", seed=44, profile="timeline", score=True,
                   conditional="focused", **CHESS)]
    small = [job(f"small_chess_44_{policy}", "small", policy, seed=44, width=294, score=True, **CHESS)
             for policy in ("native", "camera")]
    return rows + traces + small + [rescore(row) for row in rows if row["policy"] == "native"]


def confirmation_jobs(windows, rng):
    pairs = []
    rescoring = []
    for dataset, first_scene in (("7scenes", "chess/seq-05"), ("eth3d", "delivery_area")):
        roster = sorted((window for window in windows if window["dataset"] == dataset),
                        key=lambda window: (window["scene"] != first_scene, window["scene"]))
        for window in roster:
            fields = {key: window[key] for key in LOCATION_FIELDS if key in window}
            room = window["scene"].split("/")[0]
            for seed in SEEDS:
                pair_id = f"{dataset}_{room}_{seed}"
                block = [job(f"conf_{pair_id}_{policy}", "confirmation", policy, seed=seed, score=True,
                             pair_id=pair_id, **fields) for policy in ("native", "camera")]
                rng.shuffle(block)
                pairs.extend(block)
                if window["scene"] == first_scene and seed == SEEDS[0]:
                    rescoring.extend(rescore(source) for source in block)
    return pairs + rescoring


def ten_view_jobs(rng):
    rows = []
    for replica in (1, 2):
        block = [job(f"screen_{policy}_{replica}", "screen", policy, frames=TEN_FRAMES, steps=2, replica=replica)
                 for policy in POLICIES]
        rng.shuffle(block)
        rows.extend(block)
    rows.extend(job(f"capacity_trace_{policy}", "capacity_trace", policy,
                    frames=TEN_FRAMES, steps=2, profile="timeline") for policy in ("camera", "ng"))
    rows.extend(job(f"pilot_{policy}", "pilot", policy, frames=TEN_FRAMES) for policy in SELECTION)
    for scene, seeds in TEN_WINDOWS.items():
        for seed in seeds:
            room = scene.split("/")[0]
            block = [job(f"ten_{room}_{seed}_{policy}", "capacity_control" if policy == "native" else "capacity",
                         policy, scene=scene, frames=TEN_FRAMES, seed=seed, score=True,
                         first_seed=seeds[0]) for policy in POLICIES]
            rng.shuffle(block)
            rows.extend(block)
    return rows
def fixture_jobs():
    def fixture(identifier, stage, policy="native", **kwargs):
        return job("qual_" + identifier, stage + "_fixture", policy, **kwargs)

    q = "quality"
    c = "capacity"
    det = dict(width=294, steps=2, probe="qualification", backend="deterministic")
    qual = dict(width=294, steps=2, probe="qualification")
    rows = [
        fixture("camera_head", q, "camera", steps=0, profile="fixture", probe="camera_fixture"),
        fixture("native_identity", q, steps=0, probe="identity"),
        fixture("camera_identity", q, "camera", steps=0, probe="identity"),
        fixture("anchor", q, gradient_mode="capture", **det),
        fixture("native_replay", q, gradient_mode="replay", source_job="qual_anchor", **det),
        fixture("camera", q, "camera", gradient_mode="replay", source_job="qual_anchor", **det),
        fixture("negative", q, gradient_mode="negative", source_job="qual_anchor", **det),
        fixture("production_native_a", q, gradient_mode="capture", **qual),
    ]
    rows.extend(fixture(identifier, q, policy, gradient_mode="replay", source_job="qual_production_native_a", **qual)
                for identifier, policy in (("production_native_b", "native"), ("production_camera", "camera")))
    rows.extend([
        fixture("leak_base", q, "camera", width=294, steps=2, probe="reference_substitution",
                backend="deterministic", dummy_reference=False),
        fixture("leak_dummy", q, "camera", width=294, steps=2, probe="reference_substitution",
                backend="deterministic", dummy_reference=True),
        fixture("cpu", c, steps=0, profile="fixture", probe="cpu"),
        fixture("capacity_anchor", c, "camera", gradient_mode="capture", **det),
        fixture("renderer", c, "camera", width=294, steps=0, probe="renderer", source_job="qual_capacity_anchor"),
    ])
    rows.extend(fixture(f"capacity_{policy}", c, policy, gradient_mode="replay",
                        source_job="qual_capacity_anchor", **det) for policy in ("camera", "n", "g", "ng"))
    return rows


def job_table(windows):
    rng = random.Random(ORDER_SEED)
    return [*repeat_jobs(rng), *fixture_jobs(), *confirmation_jobs(windows, rng), *ten_view_jobs(rng)]


def source_hashes():
    return {name: sha(ROOT / name, text=True) for name in SOURCES}


def register():
    prior = read(PRIOR)
    sealed = read(PRIOR_LOCK)
    old_evidence = read(ROOT / "results/tco_five_view_replay_cache_v1/locked_evidence.json")
    confirm_ids = {row["id"] for row in sealed["jobs"] if row["cohort"] == "confirmation"}
    if any(row["id"] in confirm_ids and row["attempts"] for row in old_evidence["runs"]):
        raise ValueError("Prior confirmation outcomes overlap the proposed roster")
    windows = prior["planned_windows"]
    jobs = job_table(windows)
    keys = [row["id"] for row in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate registered case")
    hashes = hashes_for(jobs)
    confirmation_hashes = hashes_for(windows)
    if any(prior["input_source_sha256"].get(name) != digest for name, digest in confirmation_hashes.items()):
        raise ValueError("A reserved confirmation asset changed on disk")
    comparison_rows = [row for row in jobs if row["cohort"] not in {"confirmation", "rescore"}]
    exposed = hashes_for(comparison_rows)

    def image_hashes(rows):
        return {digest for name, digest in rows.items()
                if name.lower().endswith((".color.png", ".jpg"))}

    if image_hashes(confirmation_hashes) & image_hashes(exposed):
        raise ValueError("Development and confirmation RGB images overlap")
    write_new(INITIAL, dict(
        **runtime_pins(), study=STUDY, registration="initial", max_attempts=2,
        baseline=FIVE_BASELINE, prior_sha256=sha(PRIOR, text=True), launch_order_seed=ORDER_SEED,
        margin=MARGIN, bootstrap_samples=10_000, bootstrap_seed=20261126,
        planned_windows=windows, expected_confirmation_signatures=prior["expected_confirmation_signatures"],
        historical_exposure=prior["historical_exposure"], input_source_sha256=hashes,
        source_sha256_lf=source_hashes(), jobs=jobs, gsplat_sha256=GSPLAT, torch="2.7.1+cu128",
        selection_order=list(SELECTION), comparison=dict(candidate="camera", reference="native"),
        thresholds=dict(prediction_rtol=1e-5, gradient_rtol=1e-4, atol=1e-6, rescore_mm=0.01),
        launch_rules=dict(deadline_seconds=1200, screen_attempts_per_slot=1, max_attempts=2,
                          quality_stop="first_required_pair_terminal_missing", ten_view_candidate_slots=9),
        trace_rules=dict(clean_stage="repeat", sample_seconds=2, optional_heads="chess_native_trace_completes",
                         optional_focus="ambiguous_boundary", max_focused_repeats=1),
    ))


def verify(registration="initial", *, sources=True):
    plan = read(INITIAL)
    if plan["study"] != STUDY or plan["rental_compute_usd"] != 0 or plan["margin"] != MARGIN:
        raise ValueError("Study, cost or quality margin differ")
    if sources and plan["source_sha256_lf"] != source_hashes():
        raise ValueError("Registered effective source changed")
    if plan["jobs"] != job_table(plan["planned_windows"]):
        raise ValueError("Registered case design changed")
    if registration != "initial":
        lock = read(REGISTRATIONS[registration])
        if lock["initial_sha256"] != sha(INITIAL, text=True) or lock["study"] != STUDY:
            raise ValueError("Lock differs from the initial registration")
    return plan


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("register", "verify"))
    options = cli.parse_args()
    if options.command == "register":
        register()
    else:
        print(f"Verified {len(verify()['jobs'])} prospective cases")
