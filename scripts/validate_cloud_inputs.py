"""CPU check: transported frames produce identical upstream loader outputs."""
import json
import pathlib
import sys
import tarfile
import tempfile

import numpy as np
import torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "third_party/TCO"))
from datasets.sevenscenes import SevenScenes

manifest = json.loads((root / "manifests/midterm_3scene_kf200.json").read_text())
records = []
with tempfile.TemporaryDirectory() as temporary:
    temp = pathlib.Path(temporary)
    with tarfile.open(root / "cloud/midterm_inputs.tar.gz") as archive:
        archive.extractall(temp, filter="data")
    full = SevenScenes(SEVENSCENES_DIR="/data/data/7scenes", split="test",
                       load_img_size=518, cache_file=str(temp / "full_cache.npy"))
    transported = SevenScenes(SEVENSCENES_DIR=str(temp / "data/7scenes"), split="test",
                              load_img_size=518,
                              cache_file=str(temp / "data/7scenes/full_sequence_metadata.npy"))
    assert full.metadata == transported.metadata
    for seq, frames in manifest.items():
        original = full.get_data(sequence_name=seq, ids=frames)
        packaged = transported.get_data(sequence_name=seq, ids=frames)
        for key in ["images", "pointclouds", "valid_mask", "intrs", "extrs"]:
            a, b = original[key], packaged[key]
            assert torch.equal(a, b) if torch.is_tensor(a) else np.array_equal(a, b), (seq, key)
        records.append({"sequence": seq, "frames": frames,
                        "image_shape": list(original["images"].shape),
                        "all_model_and_evaluation_inputs_exactly_equal": True})
result = {"status": "passed", "sequences": records}
(root / "environment/cloud_inputs_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
