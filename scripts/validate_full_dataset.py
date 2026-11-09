"""Validate full public test-sequence preparation without computing model metrics."""
import argparse
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, default=ROOT/"results/data_readiness")
args = parser.parse_args()
manifest = ROOT/"manifests/sevenscenes_full_kf200.json"
selection = json.loads(manifest.read_text())
preparation = json.loads((args.data_root/"preparation.json").read_text())
prepared = {r["sequence"]: r for r in preparation["sequences"]}
assert set(selection) == set(prepared) and len(selection) == 18
rows = []
for sequence, frames in selection.items():
    record = prepared[sequence]
    assert record["registration_matches_reference"]
    counts = {}
    for suffix in [".color.png", ".depth.proj.png", ".pose.txt"]:
        files = list((args.data_root/sequence).glob("*"+suffix))
        indices = {int(p.name.split("-")[1].split(".")[0]) for p in files}
        assert indices == set(range(len(files))), (sequence, suffix, "noncontiguous original frame IDs")
        counts[suffix] = len(files)
    assert len(set(counts.values())) == 1 and counts == record["counts"]
    assert max(frames) < counts[".color.png"]
    hashes = {}
    for frame in frames:
        for suffix in counts:
            name = f"frame-{frame:06d}{suffix}"
            digest = hashlib.sha256((args.data_root/sequence/name).read_bytes()).hexdigest()
            assert digest == record["sample_sha256"][name]
            hashes[name] = digest
    rows.append(dict(sequence=sequence, full_sequence_frames=counts[".color.png"],
        selected_frames=frames, first_frame_registration_equivalence_recorded=True, selected_sha256=hashes))
result = dict(status="prepared_and_validated", sequences=18,
    total_frames=sum(r["full_sequence_frames"] for r in rows),
    selected_frames=sum(len(r["selected_frames"]) for r in rows),
    sampling_source="yyfz/Pi3 commit 97ce97b8e067e7bff5b5c8e507776e4e95189fac",
    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    registration_source=preparation["registration_source"],
    depth_source_focal=preparation["depth_source_focal"],
    registration_rgb_focal=preparation["registration_rgb_focal"],
    upstream_tco_loader_focal=preparation["upstream_tco_loader_focal"],
    holdout_policy="manifests/holdout_policy_v1.json",
    this_validation_computes_model_metrics=False,
    scope="File preparation and integrity only; registration equivalence is recorded by the separate preprocessing step",
    rows=rows)
args.output.mkdir(parents=True, exist_ok=True)
(args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))
