"""Load sealed, pretrained MASt3R pixel matches for profiled LoRA. CPU audit.

Only the historical zero-adapter correspondence caches are deserialized. No
adapted predictions, target depth, camera poses, forward pointmaps or confidence
maps are opened. Coordinates are native zero-based (x, y), with W=512, H=384.
The archive includes both directions; this API returns each unordered edge once.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/mast3r_neural_transfer"
CACHE = ROOT / "reports/mast3r_neural_transfer/cache"
REPORT = ROOT / "results/lora_profiled_v1/baseline_graph.json"
FOLDER = "corres_conf=desc_conf_subsample=8"
SCENES = ("heads", "chess", "fire")
FRAMES = (25, 225, 425, 625, 825)
PARENT = "afef4e998ac0ef3786e7a0f04844c52187a8fa07"
CHECKPOINT = "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb"
MAST3R = "f5209afc300cec36239a7ac992263f36847bbba0"
# Public historical artifacts, also recorded by the geometry/matching manifest.
PINS = {
    "plan.json": "a1123663ea6b0421046fd9fbc5a06bfd72b1c4c16f8c247efa7fd1e3d2c65d2c",
    "archive.json": "e83b3508863a2e6f91713f6cb72cb0cd0f38cc7214be003e98a4c56a39b27777",
    "session.json": "4b9b06e2ff79d9a8d8f92cada9117baf5c9a6e76e9c785577dd4eec884714669",
    "zero_adapter_bridge.json": "f42b63d8390c7d2342ea624fab5e3cfe25d899cef16a905dbd74040e52fabbf4",
}
CASE_PINS = {
    "heads": "0855fb558bd3ff6bfce95e8ecbe9b34e88496b555ad70a04ff48222e101b57bc",
    "chess": "8a6a33fd4d4896806fadef20f8688cc9f215a841fba6cc9313e1bbedb22024f2",
    "fire": "f3deec56059d2f9cea353c29ef0dd847ce4e589f063ef13cd1081db6c5a5e86d",
}
UPSTREAM_PINS = {
    "third_party/mast3r/mast3r/cloud_opt/sparse_ga.py":
        "d811614a10fe7920fb41285c3b8b9162b38a5e1ea68823f76383a2f209d3a49f",
    "third_party/mast3r/mast3r/fast_nn.py":
        "2997b5041853b9ff8d072ef2c0f533a30630cd1ef960c1a3f5b293e6f6870d90",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_record(path, expected=None, text=False):
    digest = sha(path, text=text)
    if expected is not None:
        require(digest == expected, f"Changed sealed source: {path}")
    return dict(file=path.relative_to(ROOT).as_posix(), sha256=digest,
                hash_convention="LF-normalized UTF-8" if text else "file bytes")


def provenance(scene):
    """Validate the historical checkpoint/inference chain, without model loading."""
    records = [source_record(PUBLIC / name, value) for name, value in PINS.items()]
    plan, archive, session, bridge = [read(PUBLIC / name) for name in PINS]
    require(plan["checkpoint_sha256"] == bridge["checkpoint_sha256"] == CHECKPOINT,
            "Historical checkpoint identity differs")
    require(bridge["pass_all"] and all(bridge["training_pair_exact"].values()) and
            all(item["exact"] for item in bridge["final_inference_difference"].values()),
            "Zero-adapter equivalence was not established")
    require(bridge["original_state_hash"] == bridge["final_state_hash"],
            "Historical frozen weights changed")
    require(session["upstream_commits"]["mast3r"] == MAST3R and
            session["upstream_worktrees_checked_clean_at_completion"] and
            session["exit_code"] == 0 and not session["oom_killed"],
            "Historical runtime provenance differs")
    for name in ("scripts/run_mast3r_neural_transfer.py", "scripts/mast3r_network_capture.py",
                 "scripts/mast3r_network_priors.py", "third_party/TCO/peft/lora.py"):
        records.append(source_record(ROOT / name, plan["sources"][name], text=True))
    records.extend(source_record(ROOT / name, value, text=True)
                   for name, value in UPSTREAM_PINS.items())
    case = PUBLIC / "cases" / f"{scene}_zero_none"
    records.append(source_record(case / "complete.json", CASE_PINS[scene]))
    complete = read(case / "complete.json")
    require(complete["job"] == dict(scene=scene, grid="none", arm="zero",
                                    case=f"{scene}_zero_none", post=["ordinary"]),
            "Selected cache is not the original zero-adapter inference")
    require(complete["plan_sha256"] == PINS["plan.json"] and
            complete["original_state_sha256"] == bridge["original_state_hash"],
            "Zero-case identity differs from the validated backbone")
    records.append(source_record(case / "ordinary.json", complete["metadata_sha256"]["ordinary.json"]))
    ordinary = read(case / "ordinary.json")
    require(ordinary["input_sha256"] == plan["rgb_sha256"][scene] and
            ordinary["prior_cache_files"] == 0 and ordinary["pairs"] == 20 and
            ordinary["native_size"] == [384, 512] and ordinary["inference_precision"] == "float32",
            "Fresh full-resolution pretrained inference is not established")
    require(ordinary["output_sha256"] == complete["files"]["ordinary.npz"],
            "Historical inference record differs from its completion seal")
    registry_path = ROOT / "manifests/dense_focal_inputs.json"
    records.append(source_record(registry_path,
        "20bbc76fabc3e903b0ba904beb04bf42931e141d074291e2748a185a40befd57"))
    rgb = read(registry_path)["inputs"][scene + "_base"]
    require(rgb["sha256"] == plan["rgb_sha256"][scene], "RGB registry differs")
    rgb_path = ROOT / "reports/dense_focal_pilot/inference" / rgb["file"]
    records.append(source_record(rgb_path, rgb["sha256"]))
    return plan, archive, records, complete["execution_commit"]


def match_digest(xy_i, xy_j):
    """Canonical numeric digest, independent of torch archive serialization."""
    values = torch.cat((xy_i, xy_j), dim=1).numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def load_baseline_graph(scene, device="cpu"):
    """Return ten fixed unordered pixel edges and their deterministic audit.

    Every file is verified against the sealed historical archive before loading.
    The returned list contains only i, j, xy_i, xy_j; no matching scores or
    predicted geometry are passed to training. No confidence filtering is done.
    """
    require(scene in SCENES, f"Unknown scene {scene!r}; expected one of {SCENES}")
    plan, archive, sources, execution_commit = provenance(scene)
    names = [f'{scene}_base_{plan["rgb_sha256"][scene][:12]}_{i}' for i in range(5)]
    ids = [hashlib.md5(name.encode()).hexdigest() for name in names]
    cache = CACHE / f"{scene}_zero"
    expected = {f"{FOLDER}/{a}-{b}.pth" for a, b in itertools.permutations(ids, 2)}
    records = [row for row in archive["matching_caches"][f"{scene}_zero"]
               if row["file"].startswith(FOLDER + "/")]
    require(len(records) == 20 and {row["file"] for row in records} == expected,
            "Archived correspondence inventory is incomplete")
    require({path.relative_to(cache).as_posix() for path in (cache / FOLDER).glob("*.pth")} == expected,
            "Local correspondence inventory differs")
    inventory = {row["file"]: row for row in records}
    loaded = {}
    for relative in sorted(expected):
        path, entry = cache / relative, inventory[relative]
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                f"Changed pretrained correspondence cache: {path}")
        # These locally generated archives contain only tuples and tensors.
        score, (xy_i, xy_j, confidence) = torch.load(path, map_location="cpu", weights_only=True)
        require(xy_i.dtype == xy_j.dtype == torch.int64 and xy_i.ndim == xy_j.ndim == 2 and
                xy_i.shape == xy_j.shape and xy_i.shape[1] == 2,
                f"Unexpected coordinate format: {path}")
        require(confidence.ndim == 1 and len(confidence) == len(xy_i) and
                bool(torch.isfinite(confidence).all()) and bool((confidence > 0).all()),
                f"Invalid archived matching confidence: {path}")
        require(int(score[2]) == len(xy_i), f"Archived match count differs: {path}")
        for xy in (xy_i, xy_j):
            require(bool(((xy >= 0) & (xy < torch.tensor([512, 384]))).all()),
                    f"Coordinate outside the native raster: {path}")
        paired = torch.cat((xy_i, xy_j), dim=1).numpy()
        require(len(np.unique(paired, axis=0)) == len(paired), f"Duplicate pixel pair: {path}")
        loaded[relative] = (score, xy_i, xy_j, confidence)
    matches, edges = [], []
    neighbors = {i: set() for i in range(5)}
    for i, j in itertools.combinations(range(5), 2):
        first = f"{FOLDER}/{ids[i]}-{ids[j]}.pth"
        reverse = f"{FOLDER}/{ids[j]}-{ids[i]}.pth"
        score, xy_i, xy_j, confidence = loaded[first]
        rscore, rxy_j, rxy_i, rconfidence = loaded[reverse]
        require(score == rscore and torch.equal(xy_i, rxy_i) and torch.equal(xy_j, rxy_j) and
                torch.equal(confidence, rconfidence), f"Reverse edge differs: {scene}/{i}-{j}")
        if len(xy_i):
            neighbors[i].add(j)
            neighbors[j].add(i)
        matches.append(dict(i=i, j=j, xy_i=xy_i.to(device), xy_j=xy_j.to(device)))
        edges.append(dict(i=i, j=j, frames=[FRAMES[i], FRAMES[j]], count=len(xy_i),
                          pixel_pairs_sha256_int64_le=match_digest(xy_i, xy_j),
                          forward=dict(file=(cache / first).relative_to(ROOT).as_posix(),
                                       **{k: inventory[first][k] for k in ("sha256", "bytes")}),
                          reverse=dict(file=(cache / reverse).relative_to(ROOT).as_posix(),
                                       **{k: inventory[reverse][k] for k in ("sha256", "bytes")})))
    unseen, components = set(range(5)), []
    while unseen:
        todo, component = [min(unseen)], set()
        while todo:
            node = todo.pop()
            if node not in component:
                component.add(node)
                todo.extend(neighbors[node] - component)
        unseen -= component
        components.append(sorted(component))
    require(len(components) == 1, f"Disconnected pretrained match graph for {scene}")
    audit = dict(scene=scene, cache=cache.relative_to(ROOT).as_posix(), names=names,
                 target_frames=list(FRAMES), native_height_width=[384, 512],
                 coordinate_convention="zero-based integer (x,y), flatten=y*512+x",
                 sources=sources, parent_evidence_commit=PARENT,
                 historical_inference_commit=execution_commit, checkpoint_sha256=CHECKPOINT,
                 pretrained_zero_equivalence_verified=True, fresh_float32_inference_verified=True,
                 checkpoint_provenance="Pinned historical checkpoint and zero-equivalence records; model bytes not loaded by this audit",
                 edges=edges, unordered_edges=len(edges), verified_directed_files=len(loaded),
                 total_unordered_matches=sum(edge["count"] for edge in edges),
                 connected_components=components, exact_duplicate_pairs=0,
                 reverse_copies_exact=True, native_pixel_bounds_valid=True,
                 confidence_filter_applied=False, scores_or_confidences_returned=False,
                 adapted_predictions_loaded=False, forward_pointmaps_loaded=False,
                 target_depth_or_camera_arrays_loaded=False, pass_all=True)
    return matches, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Rerun CPU audit and compare with the sealed report")
    args = parser.parse_args()
    report = dict(schema="profiled_baseline_graph_v1", source_sha256_lf=sha(Path(__file__), text=True),
                  scope="Pretrained fixed pixel graph and provenance audit; no accuracy result",
                  scenes={scene: load_baseline_graph(scene)[1] for scene in SCENES},
                  inference_or_gpu_work_performed=False, rental_compute_usd=0, pass_all=True)
    if args.verify:
        require(read(REPORT) == report, "Baseline graph semantic report differs; preserve evidence and investigate")
        print("Verified sealed pretrained graphs: 3 scenes, 30 unordered edges, 60 directed cache files.")
    else:
        require(not REPORT.exists(), "Preserve the existing graph report; use --verify")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Sealed pretrained baseline graph audit: {REPORT}")


if __name__ == "__main__":
    main()
