"""Package the fixed 25 evaluation frames, preserving upstream metadata.

The cache records the actual full-sequence frame count from the prepared local
data. It is a transport optimization, not a change to frame selection or inputs.
"""
import argparse
import hashlib
import io
import json
import pathlib
import tarfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=pathlib.Path, required=True)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((root / "manifests/midterm_3scene_kf200.json").read_text())
outdir = root / "cloud"
outdir.mkdir(exist_ok=True)
metadata = {}
hashes = {}
archive = outdir / "midterm_inputs.tar.gz"
with tarfile.open(archive, "w:gz") as output:
    for seq, frames in manifest.items():
        sequence = args.data_root / seq
        actual_count = len(list(sequence.glob("*.color.png")))
        assert actual_count > max(frames)
        metadata[seq] = actual_count
        for frame in frames:
            for suffix in (".color.png", ".depth.proj.png", ".pose.txt"):
                path = sequence / f"frame-{frame:06d}{suffix}"
                relative = f"data/7scenes/{seq}/{path.name}"
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                output.add(path, arcname=relative)
    for scene in sorted({seq.split('/')[0] for seq in manifest}):
        output.add(args.data_root / scene / "TestSplit.txt", arcname=f"data/7scenes/{scene}/TestSplit.txt")
    cache = io.BytesIO()
    np.save(cache, metadata)
    value = cache.getvalue()
    info = tarfile.TarInfo("data/7scenes/full_sequence_metadata.npy")
    info.size = len(value)
    output.addfile(info, io.BytesIO(value))
    content = json.dumps(hashes, indent=2).encode()
    info = tarfile.TarInfo("selected_frame_sha256.json")
    info.size = len(content)
    output.addfile(info, io.BytesIO(content))
record = {"archive": archive.name, "bytes": archive.stat().st_size,
          "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
          "sequences": metadata, "selected_frames": sum(map(len, manifest.values())),
          "selection_source": "Pi3 evaluation's published 7scenes kf200 manifest"}
(outdir / "inputs_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
