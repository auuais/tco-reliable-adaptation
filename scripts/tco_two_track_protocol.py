"""Prospective local register for both five-view TCO experiments. No model or score imports."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_two_track_v1"
MANIFEST = ROOT / "manifests" / (STUDY + ".json")
PUBLIC = ROOT / "results" / STUDY
PRIVATE = ROOT / "reports" / STUDY
DEV = ("heads/seq-01", "chess/seq-03", "fire/seq-03")
STRESS = ("office/seq-02", "courtyard")
SEEDS = [42, 43, 44]
NEW_SEEDS = [7, 77, 777]
ARMS = {"Full10": 10, "Full40": 40, "Cache10": 10, "Refresh1": 10, "Refresh1Uncached": 10,
        "Refresh2": 10, "Narrow": 10, "NarrowUncached": 10, "Distilled": 10, "OfflineOriginal": 10}
ORDER_SEED = 2026112801
HANDOFF = "docs/TCO_TWO_TRACK_SOL_HANDOFF.md"
PINS = [HANDOFF, "docs/TCO_SPEED_AND_ACCURACY_NEXT_TESTS.md", "configs/7scenes_baseline.yaml",
        "scripts/tco_frozen_prefix.py", "scripts/run_tco_one_percent.py", "scripts/run_experiment.py",
        "scripts/tco_one_percent_geometry.py", "scripts/anchor_pose_surface.py",
        "scripts/prepare_tco_final_ln_arkit.py", "results/tco_one_percent_v1/evidence.json",
        "results/tco_one_percent_v1/confirmation_evidence.json", "results/tco_one_percent_v1/archive_receipt.json",
        "manifests/tco_final_global_layernorm_v1_replication.json"]


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def pinned_source(name):
    value = digest(ROOT / name)
    # Git normalizes a single CRLF in the historical Windows copy of this file.
    # Both digests were checked against the registered working copy and its Git blob.
    if name == "scripts/anchor_pose_surface.py" and value == "c58cff0dd9198935ec11453d7b2bc4200439453e9425d524093bd2520387bd69":
        return "db4405916220c9c564df61d93a812775cfb11d090e0ab9baf0caf34bd5df8306"
    return value


def seal(path, data):
    payload = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Changed sealed data: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)


def schedule(stage, rooms, seeds, arms, salt=0):
    """Shuffle the blocks, and then arm ordering independently within each block."""
    order = random.Random(ORDER_SEED + salt)
    blocks = list(product(rooms, seeds))
    order.shuffle(blocks)
    result = []
    for room, seed in blocks:
        shuffled = list(arms)
        order.shuffle(shuffled)
        for arm in shuffled:
            result.append(dict(id=f"{stage}_{room.replace('/', '_')}_{seed}_{arm}",
                               room=room, slug=room.replace("/", "__"), seed=seed, arm=arm, steps=ARMS[arm], profile="clean"))
    return result


def locked_inputs(source, rooms):
    records = read(ROOT / source)["rows"]
    fields = ("dataset", "scene", "frames", "raster_shape", "input_sha256", "reference_sha256")
    result = []
    for room in rooms:
        matches = [r for r in records if r["scene"] == room and r["arm"] == "four" and r["steps"] == 40]
        if len(matches) != 3:
            raise ValueError("Expected three previously committed controls: " + room)
        packet = {key: matches[0][key] for key in fields}
        if any(packet != {key: r[key] for key in fields} for r in matches):
            raise ValueError("Different inputs for archived controls: " + room)
        result.append(packet)
    return result


def registration():
    dev = locked_inputs("results/tco_one_percent_v1/evidence.json", DEV)
    stress = locked_inputs("results/tco_one_percent_v1/confirmation_evidence.json", STRESS)
    dev_jobs = schedule("development", DEV, SEEDS, ["Full10", "Full40", "Cache10", "Refresh1", "Refresh1Uncached"])
    dev_inputs = {p["scene"]: p for p in dev}
    for job in dev_jobs:
        packet = dev_inputs[job["room"]]
        job.update({k: packet[k] for k in ("frames","dataset","input_sha256","reference_sha256","raster_shape")})
    pilots = [job["id"] for job in dev_jobs if job["room"] == DEV[0] and job["seed"] == 42 and job["arm"] != "Full40"]
    engineering = []
    for i,(arm,profile) in enumerate([("Refresh1","fixed"),("Refresh1Uncached","fixed"),("Refresh1","stale"),
                                        ("Refresh1","real"),("Refresh1","real")],1):
        engineering.append(dict(id=f"engineering_heads_{profile}_{i}_{arm}", room=DEV[0], slug="heads__seq-01",
                                arm=arm, profile=profile, seed=42, steps=2, width=518,
                                **{k:dev[0][k] for k in ("frames","dataset","input_sha256","reference_sha256","raster_shape")}))
    return dict(study=STUDY, design_revision="68012e64674f51c258a8951d4ca1fe0ae4957bdf",
        pins={p: pinned_source(p) for p in PINS}, runtime=dict(
            gpu="NVIDIA GeForce RTX 3080", cap_mib=8960, memory_fraction=.875,
            allocator="garbage_collection_threshold:0.75,expandable_segments:True", runpod_usd=0,
            checkpoint="f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e",
            image="sha256:541dd1e09a32b8da7851929e73810882415aa30967b0ae2acd5473c9cf5ec3f1",
            upstream="65387333877723cf99086f7f29089379dd50c59a", views=5, width=518,
            precision="baseline production with shared FP32 canonical final surface and alignment",
            lora=dict(rank=4,alpha=16,dropout=0,lr=.0003,clip=1.,optimizer="Adam",heads="frozen"),
            time="Synchronized loaded-CPU-model placement through adapters/cache, all updates, transfers, postprocessing and delivered FP32 geometry; separately cold load and scoring; no scorer beside timed GPU.",
            memory="reset before scene; peak live allocated and reserved; separately report retries, OOM and cold",
            failures="Hard 1200s per model case. One unchanged retry for interruption/provision or ambiguous watchdog. Keep scientific failures in denominators. Longer control deadline fixed at 3600s for 120 updates.",
            geometry="Shared Umeyama then rigid ICP 0.1m, directed mean NN in mm. Priors only for inference; separate evaluation filesystem, fixed support."),
        design=dict(rooms_dev=dev,rooms_stress=stress,dev_seeds=SEEDS,stress_seeds=NEW_SEEDS,order_seed=ORDER_SEED,
                    order="seeded shuffle blocks then arms per block; see schedule(). freeze new phase ledgers before inference",
                    a_development_seed_offset=0, a_followup_seed_offset=1, a_stress_seed_offset=2),
        a=dict(arms=ARMS, dev_jobs=dev_jobs, pilot_ids=pilots, engineering_jobs=engineering,
            clean_order=pilots+[j["id"] for j in dev_jobs if j["id"] not in pilots],
            engineering=dict(scene=DEV[0],seed=42,
                require=["fixed signed derivative equality of cached/uncached twins", "differentiable effective LoRA changes",
                         "explicit early-weight perturbation catches stale cache; invalidated equals recomputed",
                         "two production controls; full-resolution resource pilot before matrix"],
                counters={"Refresh1":[1,10],"Refresh2":[2,10],"Narrow":[1,10]},
                clean_pilot="Heads seed 42 Full10, Cache10, Refresh1 and uncached twin; an identical clean slot is reusable with original receipt"),
            policies={"Full10":"all 24 x10","Full40":"all 24 x40","Cache10":"suffix stages 12..23 x10",
                      "Refresh1":"all at update 1, suffix for 2..10, invalidate cache after early updates",
                      "Refresh1Uncached":"same Adam trajectory policy as Refresh1 with cache disabled",
                      "Refresh2":"all at 1 and 6, suffix otherwise; invalidate after both",
                      "Narrow":"pair 11 and suffix at 1, suffix otherwise; qualify uncached twin",
                      "Distilled":"cached suffix restarts from teacher-trained shared factors on each scene",
                      "OfflineOriginal":"same offline 24 outer updates, original loss only; comparator"},
            gates={"versus":"Full10","accuracy_at_most":1.01,"completeness_at_most":1.03,"time_strictly_below":1.0,
                   "development":"all three point gates separately on development aggregate and stress aggregate",
                   "fresh":"all three point and one-sided upper gates with complete cohort",
                   "vs_Full40_stretch":{"time_at_most":.25,"peak_live_at_most":.764},
                   "full40_quality_separate":{"accuracy_at_most":1.01,"completeness_at_most":1.03}},
            followup={"priority":"Narrow if Refresh1 did not fit or peak ratio vs Full40 exceeds .764; otherwise Refresh2 if time ratio <1 vs Full10 and either quality point misses. At most one alternative.",
                      "selection":"First choose qualified hybrid on development, stress on Office and Courtyard; if no hybrid survives the stress gate or surviving hybrid misses full40 time/memory stretch, do one distillation pilot. Final choice must pass each development and stress aggregate. Rank by development time ratio vs Full10, peak-live ratio, then Refresh1, Refresh2, Narrow, Distilled. Never nominate an ordinary or uncached control.",
                      "distillation":"Six disjoint ARKit Training visits per fresh exposure/ordering rule. Full40 teacher each. Four deterministic passes, one outer update/window, LR .0001, teacher relative camera and valid log depth losses normalized using training only. Compare OfflineOriginal and standard init. Lock specific implementation before scoring.",
                      "matched_time":"Largest feasible ordinary k in 4,5,6,10 from development times only; test 1..3 if 4 too slow. Lock k before fresh geometry. Zero-step RGB-only and zero-step supplied-camera unprojection share forward; report postprocess time."}),
        b=dict(dev_triplets=[{"scene":r,"seeds":SEEDS,"anchor":42,
                  "ids":[f"{r.replace('/','_').replace('-','')}_four_40_{s}" for s in SEEDS]} for r in DEV],
            deployment_inputs="Fixed first seed anchor. Predicted rig aligned to supplied rig by positive scale, rotation, translation. No reference fitting. Nearest visible positive z, deterministic ties, median valid log depths for >=2, fallback to anchor. Same five anchor views, K, pixels and support; singleton holes use same fallback.",
            controls=["three native Full40 singles","three singleton reprojections","anchor native and singleton","triplicated anchor vs singleton","real three-run fusion"],
            engineering=["known Sim3 and plane, unequal K, occlusion, missing sources","fixed support","sham agreement","no reference access while sealing predictions"],
            dev_gate={"versus_native_accuracy_strict":1.,"versus_reprojected_accuracy_strict":1.,
                      "versus_native_completeness_inclusive":1.03,"rooms_improving_native_accuracy_min":2,
                      "rooms":3,"sham_must_pass":True,"coverage_complete":True},
            conditional="Only after triplet gate: run 3 scenes x 3 seeds, capture states immediately after steps 24,32,40. Compare terminal 40, three checkpoint outputs with anchor 40, and average scaled BA effective changes. Nominate 1 allowed qualifying candidate with lowest equal-room accuracy ratio vs native and completeness <=1.03, at least two rooms with improved accuracy and clean applicable reprojection control; tie shorter measured time then triplet fusion, checkpoint depths, effective BA. Baseline longer 120 for triple and closest measured ordinary 40/48 for single trajectory.",
            fresh_gate={"accuracy_strictly_below":1.,"completeness_at_most":1.03,"require_point_and_upper":True,
                        "both_superior_requires_both_upper_strict":1.}),
        fresh=dict(primary=dict(dataset="ARKitScenes Validation",visits=6,seeds=NEW_SEEDS,
                    source_revision="474aaa83d7b1c9080619d8485de2d7c885c01fc5",
                    explicit_excluded_visits=[369706,381644,381649,381658,381868,381872,381875],
                    rule="Ascending numeric visit then first eligible video/window per pinned previous loader, pose+metadata only. Exclude all previously seen scenes, training/teacher/bridge sites, known shared-site IDs, scans, duplicate frames/input hashes; do not reuse old scanner's skips. One unique window per visit. Audit full exclusions. Seal both nominees/rows/input/reference hashes before either set of scores."),
                   replication=dict(dataset="Hypersim test",scenes=6,trigger="supported primary for corresponding direction",
                    rule="Pin source revision and scene-first sorted intake and five-view rules before acquisitions/outcomes. Distinct held-out public scenes, one window each, train bridge separately; verify axes, range to Z, units, pose, image and scoring.")),
        analysis=dict(boot_draws=10000,bootstrap_seed=20261128,confidence_side=.95,
            repeated="median 3 paired seeds within scene per arm; ratio candidate/reference per scene; arithmetic mean across scenes. Bootstrap scene identities then jointly paired seed indices. Linear quantiles of 10,000 ratios.",
            triplet="one fixed triplet per scene yields ONE inference; ratio to arithmetic mean of its three native singles in that scene (and to matched reprojected singles where specified); bootstrap scenes only. Do not recycle anchor permutations.",
            grades="all gates on unrounded estimate and upper bound; lower bound on losing side supports violation; observed fail with overlapping bounds uncertain; missing registered endpoints preclude positive verdict. Show mm, secs, alloc/reserved MiB, LOO, failures and denominators separately per dataset."),
        evidence="immutable attempts/registrations and preregistered cohorts, independently rescore one prespecified saved prediction per arm/cohort (lexicographically first case and triplicate control), independent CPU decisions, C and E verified private full archive, public compact Git evidence, local $0")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","verify"))
    args=parser.parse_args()
    data=registration()
    if args.action=="register":
        seal(MANIFEST,data)
    elif not MANIFEST.is_file() or read(MANIFEST)!=data:
        raise ValueError("Registration differs from source design or pinned evidence")
    print(MANIFEST.name,digest(MANIFEST))
