"""Export the historical five-view development inputs and separate evaluation targets."""
import json
import sys
from pathlib import Path

import numpy as np

from run_profiled_d0 import sha
from run_tco_frozen_prefix import signature
from run_tco_one_percent import load_case


def main():
    root = Path("/prepared")
    for name in ("model", "reference"):
        (root / name).mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path("results/tco_one_percent_v1/evidence.json").read_text())["rows"]
    result = []
    small = "--small" in sys.argv
    width = 294 if small else 518
    for room in (("heads/seq-01",) if small else ("heads/seq-01", "chess/seq-03")):
        job = dict(dataset="7scenes", scene=room, frames=[0, 200, 400, 600, 800], phase="development")
        supplied, reference = load_case(job, root, width)
        model = dict(images=supplied["images"].numpy(), extrs=supplied["extrs"], intrs=supplied["intrs"])
        ground = dict(pointclouds=reference["pointclouds"], valid_mask=reference["valid_mask"])
        inputs_sha = signature(dict(rgb=model["images"], poses=model["extrs"], intrinsics=model["intrs"]))
        reference_sha = signature(dict(mask=ground["valid_mask"], surface=ground["pointclouds"]))
        prior = next(row for row in rows if row["scene"] == room and row["frames"] == job["frames"])
        if not small and (inputs_sha, reference_sha) != (prior["input_sha256"], prior["reference_sha256"]):
            raise ValueError("Historical decode mismatch for " + room)
        item = dict(room=room, environment=room.split("/")[0], dataset="7scenes", frames=job["frames"],
                    slug=room.replace("/", "__") + ("_294" if small else ""), input_sha256=inputs_sha, reference_sha256=reference_sha)
        for name, arrays, field in (("model", model, "input_file_sha256"), ("reference", ground, "reference_file_sha256")):
            file = root / name / (item["slug"] + ".npz")
            if file.exists():
                with np.load(file, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k,v in arrays.items()):
                        raise ValueError("Different portable packet " + str(file))
            else:
                temporary = file.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(file)
            item[field] = sha(file)
        result.append(item)
    receipt = Path("/receipts") / ("qualification_inputs.json" if small else "development_inputs.json")
    encoded = json.dumps({"rooms": result}, indent=2, sort_keys=True) + "\n"
    if receipt.exists() and receipt.read_text() != encoded:
        raise ValueError("Existing development intake receipt changed")
    receipt.write_text(encoded)
    print("Verified native development RGB, camera priors and evaluation references in both rooms.")


if __name__ == "__main__":
    main()
