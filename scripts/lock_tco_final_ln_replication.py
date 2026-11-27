"""Replay the scientific gate and, only on promotion, prepare a locked six-visit ARKitScenes replication."""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request

from lock_tco_final_ln import room_packet
from prepare_tco_final_ln_arkit import DATA_ROOT, METADATA_SHA, PUBLISHER, REVISION, SPLIT_URL, TRAINING_PAIR, build_record
from prepare_tco_final_ln_data import ARCHIVE, EXPOSURE_SUFFIXES
from replay_tco_final_ln import analyze
from tco_final_ln_protocol import LOCK, MANIFEST, PREP, PUBLIC, ROOT, SEEDS, STUDY, matrix, read, seal, sha


FOLLOW_LOCK = ROOT / "manifests" / (STUDY + "_followup.json")
REPL_LOCK = ROOT / "manifests" / (STUDY + "_replication.json")
CODE = Path(__file__).resolve()
SITE_KEYS = ("physical_location_id", "venue_id", "property_id", "site_id", "building_id")
MISSING = {"", "none", "unknown", "nan", "n/a", "na", "null", "-1"}
SKIP_REASONS = {"No five-view window from the registered time/pose rules", "Missing registered public asset"}


class InsufficientEligibleVisits(Exception):
    """The published validation roster was exhausted after all documented exclusions."""


def gate_branch(primary, follow_lock, follow):
    """The all-pass branch preserves the original candidate regardless of its component ranking."""
    if primary["dataset"] != "stanford12" or follow["dataset"] != "development":
        raise ValueError("Decision cohort mismatch")
    g = primary.get("decisions_on_complete_rooms", {})
    equal = primary["formal_coverage"] and all(g.get(k) == "pass" for k in
                                                ("accuracy_superiority_vs_short", "completeness_vs_short"))
    branch = "all_pass" if primary["joint_verdict"] == "pass" else ("equal_update_pass_only" if equal else "other")
    if follow_lock["branch"] != branch:
        raise ValueError("Follow-up branch contradicts Stanford results")
    if branch == "all_pass":
        if follow.get("branch") != branch or set(follow["analyses"]) != {"LN10", "LN10_scale", "LN10_bias"}:
            raise ValueError("Missing released scale/bias mechanism experiment")
        return "LN10", "L10"
    candidate, short = ("LN20", "L20") if equal else ("LN10_low", "L10")
    g = follow["independent_replication_gate"]
    if g["candidate"] != candidate:
        raise ValueError("Follow-up candidate mismatch")
    if not g["eligible"]:
        return None
    if (not g["coverage_complete"] or not g["preregistered_development_rooms"]
            or not all(follow["point_gates"][k] for k in ("better_accuracy_vs_short", "accuracy_vs_full",
                                                           "completeness_vs_full", "faster_vs_full"))):
        raise ValueError("Positive label contradicts the registered equal-budget/quality/time checks")
    return candidate, short


def gate():
    pp, pr, pd = [PUBLIC / ("primary_" + x + ".json") for x in ("analysis_plan", "rows", "decision")]
    fp, fr, fd = [PUBLIC / ("followup_" + x + ".json") for x in ("analysis_plan", "rows", "decision")]
    paths = (MANIFEST, LOCK, FOLLOW_LOCK, pp, pr, pd, fp, fr, fd, PUBLIC/"arkit_training_input.json")
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        return dict(status="pending", missing=missing)
    lock, follow_lock = read(LOCK), read(FOLLOW_LOCK)
    primary, follow = read(pd), read(fd)
    p_rows, f_rows = read(pr)["rows"], read(fr)["rows"]
    if (follow_lock["primary_decision_sha256"] != sha(pd)
            or lock["pinned_files"][MANIFEST.relative_to(ROOT).as_posix()] != sha(MANIFEST)):
        raise ValueError("Primary or follow-up decision no longer matches its lock")
    for jobs, rows, roles in ((lock["primary"] + lock["anchor"], p_rows, {"primary", "anchor"}),
                              (follow_lock["jobs"], f_rows, {"followup"})):
        expected = {(j["id"], j["room"], j["seed"], j["arm"]) for j in jobs}
        observed = [(r["id"], r["room"], r["seed"], r["arm"]) for r in rows if r["role"] in roles]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("Cannot verify full attempted follow-up/primary roster")
    if analyze(read(pp), p_rows, "LN10", "L10", "L40") != primary:
        raise ValueError("Primary decision did not replay")
    choice = gate_branch(primary, follow_lock, follow)
    if follow_lock["branch"] == "all_pass":
        for arm in ("LN10", "LN10_scale", "LN10_bias"):
            filtered = [r for r in f_rows if r["arm"] in (arm, "L10", "L40")]
            if analyze(read(fp), filtered, arm, "L10", "L40") != follow["analyses"][arm]:
                raise ValueError("Mechanism comparison did not replay")
    else:
        arm, ref = ("LN20", "L20") if follow_lock["branch"] == "equal_update_pass_only" else ("LN10_low", "L10")
        if analyze(read(fp), f_rows, arm, ref, "L40") != follow:
            raise ValueError("Development advancement decision did not replay")
    digests = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    if choice is None:
        return dict(status="not_promoted", branch=follow_lock["branch"], source_sha256=digests)
    return dict(status="eligible", branch=follow_lock["branch"], candidate=choice[0], short=choice[1],
                source_sha256=digests)


def cached_public(directory, name, url, expected=None):
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=40) as response:
            data = response.read()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Publisher metadata digest changed: " + name)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(data)
        partial.replace(path)
    if expected and sha(path) != expected:
        raise ValueError("Cached metadata mismatch: " + name)
    return path


def number(value):
    value = str(value).strip()
    return int(value.split(".")[0]) if re.fullmatch(r"\d+(?:\.0+)?", value) else None


def discover():
    reg = read(MANIFEST)["arkit"]
    if reg["commit"] != REVISION or reg["metadata_sha256"] != METADATA_SHA:
        raise ValueError("Training and prospective metadata revisions disagree")
    cache = DATA_ROOT.parent / (STUDY + "_eligibility")
    sources = {
        "splits": cached_public(cache, "raw_train_val_splits.csv", SPLIT_URL, METADATA_SHA),
        "assets": cached_public(cache, "metadata.csv", PUBLISHER + "/metadata.csv"),
        "laser": cached_public(cache, "laser_scanner_point_clouds_mapping.csv",
                              PUBLISHER + "/laser_scanner_point_clouds/laser_scanner_point_clouds_mapping.csv"),
        "publisher_downloader": cached_public(cache, "download_data.py",
            f"https://raw.githubusercontent.com/apple-aiml-research/ARKitScenes/{REVISION}/download_data.py",
            reg["downloader_sha256"]),
    }
    def csv_rows(key):
        return list(csv.DictReader(io.StringIO(sources[key].read_text(encoding="utf-8-sig"))))
    module = ast.parse(sources["publisher_downloader"].read_text(encoding="utf8"))
    missing = next(ast.literal_eval(s.value) for s in module.body if isinstance(s, ast.Assign)
                   and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "missing_3dod_assets_video_ids")
    hashes = {key: sha(path) for key,path in sources.items()}
    return csv_rows("splits"), csv_rows("assets"), csv_rows("laser"), set(map(int, missing)), hashes


def site_groups(metadata, laser):
    """Map visits sharing an explicit site ID or the same registered laser scan into one component."""
    parent = {}
    def root(v):
        parent.setdefault(v, v)
        while parent[v] != v:
            v, parent[v] = parent[v], parent[parent[v]]
        return v
    owner = {}
    def connect(visit, key, value):
        if visit is None or value is None or str(value).strip().casefold() in MISSING:
            return
        label = (key, str(value).strip().casefold().removesuffix(".0"))
        other = owner.setdefault(label, visit)
        a,b = root(visit),root(other)
        parent[max(a,b)] = min(a,b)
    for row in metadata:
        if "visit_id" not in row or "video_id" not in row:
            raise ValueError("Missing public visit/video identifiers")
        for key in SITE_KEYS:
            connect(number(row["visit_id"]), key, row.get(key))
    for row in laser:
        connect(number(row["visit_id"]), "laser_scan", row["laser_scanner_point_clouds_id"])
    return {v:root(v) for v in parent}


def previous_exposure(splits):
    """Find previous ARKit scientific records by typed visit/video IDs, scene labels, and case paths."""
    video_visits = {number(r["video_id"]):number(r["visit_id"]) for r in splits}
    visits = set(video_visits.values()) - {None}
    rg_terms = r"(?i)arkit(?:scenes)?|(?:video|visit)[_ -]?id"
    roots = {"manifests":ROOT/"manifests", "results":ROOT/"results", "reports":ROOT/"reports",
             "docs":ROOT/"docs", "runs":ROOT/"runs", "archive":ARCHIVE}
    ids = {TRAINING_PAIR[0]}
    sources = []
    for tag, base in roots.items():
        ignore = ["-g", "!**/tco_final_global_layernorm_v1/**", "-g", "!tco_final*", "-g", "!TCO_FINAL_GLOBAL_LAYERNORM_HANDOFF.md"]
        names = subprocess.run(["rg", "--files", "--no-ignore", "--null", *ignore, str(base)], capture_output=True)
        cmd = ["rg", "-il", "--no-ignore", "--null", *ignore]
        for suffix in EXPOSURE_SUFFIXES:
            cmd += ["-g", "*."+suffix]
        docs = subprocess.run([*cmd, "--", rg_terms, str(base)], capture_output=True)
        if any(result.returncode not in (0,1) for result in (names, docs)):
            raise RuntimeError("Prior exposure search incomplete: " + tag)
        for listing, content in ((names, False), (docs, True)):
            for filename in filter(None,listing.stdout.split(b"\0")):
                path = Path(filename.decode("utf-8"))
                rel = path.relative_to(base).as_posix()
                if not content and "arkit" not in rel.casefold():
                    continue
                text = rel + ("\n" + path.read_text(encoding="utf-8-sig",errors="replace") if content else "")
                hit = set()
                for v in re.findall(r"(?i)(?:arkit(?:scenes)?[ /_:.-]+|visit[_ -]?id[\s\"'=:\[\]]+)(\d{5,9})", text):
                    if int(v) in visits:
                        hit.add(int(v))
                for v in re.findall(r"(?i)(?:video[_ -]?id[\s\"'=:\[\]]+|arkit(?:scenes)?[ /_:.-]+\d+[ /_:.-]+)(\d{7,9})",text):
                    if video_visits.get(int(v)):
                        hit.add(video_visits[int(v)])
                if hit:
                    ids |= hit
                    sources.append(dict(scope=tag,path=rel,visits=sorted(hit)))
    return ids, sources


def ranked_visits(splits):
    members = defaultdict(set)
    assigned = {}
    for row in splits:
        if row["fold"] != "Validation":
            continue
        visit, video = number(row["visit_id"]), number(row["video_id"])
        if visit is None or video is None:
            continue
        if video in assigned and assigned[video] != visit:
            raise ValueError("Validation video assigned to multiple visits")
        assigned[video] = visit
        members[visit].add(video)
    return [(visit, sorted(members[visit])) for visit in sorted(members)]


def prepare(decision):
    if decision["status"] != "eligible":
        raise ValueError("No model qualifies for Validation acquisition")
    if REPL_LOCK.exists():
        prior = read(REPL_LOCK)
        if prior["scientific_gate"] != decision or prior["dataset"] != "arkitscenes":
            raise ValueError("Existing replication lock belongs to another scientific outcome")
        for file, expected in prior["pinned_files"].items():
            if sha(ROOT/file) != expected:
                raise ValueError("Locked replication software or evidence changed: " + file)
        for room in prior["rooms"]:
            for role,field in (("model","input_file_sha256"),("reference","reference_file_sha256")):
                if sha(PREP/role/(room["slug"]+".npz")) != room[field]:
                    raise ValueError("Locked stimulus or reference changed")
        print("Verified existing replication lock", sha(REPL_LOCK))
        return
    splits, assets, laser, missing, hashes = discover()
    groups = site_groups(assets, laser)
    exposed, exposure_sources = previous_exposure(splits)
    unavailable = {groups.get(v,v) for v in exposed}
    lookup = {number(r["video_id"]): number(r["visit_id"]) for r in assets}
    rooms, exclusions, reached = [], [], set()
    visits = ranked_visits(splits)
    if not visits:
        raise ValueError("No Validation visits in the downloaded split metadata")
    for visit, videos in visits:
        group = groups.get(visit, visit)
        if group in unavailable or group in reached:
            exclusions.append(dict(visit=visit, reason="prior exposure or duplicate published physical site/scan", site_group=group))
            continue
        for video in videos:
            if video in missing or lookup.get(video) != visit:
                exclusions.append(dict(visit=visit, video=video, reason="publisher missing trajectory or inconsistent metadata"))
                continue
            try:
                row = build_record(DATA_ROOT, "Validation", visit, video, PREP)
            except urllib.error.HTTPError as exc:
                if exc.code not in (404, 410):
                    raise
                exclusions.append(dict(visit=visit, video=video, reason=f"publisher required asset HTTP {exc.code}"))
                continue
            except ValueError as exc:
                if str(exc) not in SKIP_REASONS:
                    raise
                exclusions.append(dict(visit=visit, video=video, reason=str(exc)))
                continue
            rooms.append(room_packet(dict(row,environmental_site=str(visit)),"arkitscenes"))
            reached.add(group)
            break
        if len(rooms) == 6:
            break
    evidence = dict(public_metadata_sha256=hashes, ranked_visit_count=len(visits),
                    roster_exhausted=len(rooms) < 6, missing_or_duplicate=exclusions, previously_exposed=sorted(exposed),
                    previous_exposure_records=exposure_sources,selected=rooms,
                    identity_evidence="Distinct registered visits, explicit shared site identifiers and laser scan IDs where available")
    seal(PUBLIC/"arkit_roster_intake.json", evidence)
    if len(rooms) != 6:
        raise InsufficientEligibleVisits(f"The public roster supplied {len(rooms)} eligible visits of six required")
    arms = [decision["short"],decision["candidate"],"L40"]
    jobs = matrix(rooms, arms, "replication")
    files = [CODE, ROOT/"scripts/prepare_tco_final_ln_arkit.py", PUBLIC/"arkit_roster_intake.json",
             *[PUBLIC/f"arkit_validation_{r['environment']}.json" for r in rooms]]
    pinned = dict(read(LOCK)["pinned_files"], **decision["source_sha256"])
    pinned.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
    seal(REPL_LOCK, dict(study=STUDY, dataset="arkitscenes", seeds=SEEDS, arms=arms, rooms=rooms,
                         scientific_gate=decision, pinned_files=pinned, jobs=jobs))
    print("Locked",len(rooms),"visits and",len(jobs),"matched trials",sha(REPL_LOCK))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("status","prepare"))
    args = p.parse_args()
    result = gate()
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.action == "prepare" and result["status"] == "eligible":
        prepare(result)


if __name__ == "__main__":
    main()
