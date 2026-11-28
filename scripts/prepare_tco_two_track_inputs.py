"""Decode exposed five-view windows with the original loader and separate RGB/camera inputs from reference surfaces."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run_tco_frozen_prefix import signature
from run_tco_one_percent import load_case
from tco_final_ln_protocol import ROOT, seal, sha


WINDOWS = {
    "development": ("heads/seq-01", "chess/seq-03", "fire/seq-03"),
    "stress": ("office/seq-02", "courtyard"),
}


def save_packet(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v, equal_nan=True) for k, v in arrays.items()):
                raise ValueError("Archived input changed: " + str(path))
    else:
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    return sha(path)


def prepare(stage):
    source = ROOT / "results/tco_one_percent_v1" / ("evidence.json" if stage == "development" else "confirmation_evidence.json")
    rows = json.loads(source.read_text())["rows"]
    records = []
    for room in WINDOWS[stage]:
        candidates = [r for r in rows if r["scene"] == room and r["arm"] == "four" and r["steps"] == 40]
        assert len(candidates) == 3 and len({(r["input_sha256"], r["reference_sha256"]) for r in candidates}) == 1
        prior = candidates[0]
        job = dict(scene=room, dataset=prior["dataset"], frames=prior["frames"],
                   phase="development" if stage == "development" else "confirmation")
        with TemporaryDirectory() as tmp:
            supplied, gt = load_case(job, Path(tmp), 518)
        model = dict(images=supplied["images"].numpy(), extrs=supplied["extrs"], intrs=supplied["intrs"])
        reference = dict(pointclouds=gt["pointclouds"], valid_mask=gt["valid_mask"])
        ih = signature(dict(rgb=model["images"], poses=model["extrs"], intrinsics=model["intrs"]))
        rh = signature(dict(mask=reference["valid_mask"], surface=reference["pointclouds"]))
        if (ih, rh) != (prior["input_sha256"], prior["reference_sha256"]):
            raise ValueError("Current loader differs from the historical five-view case: " + room)
        slug = room.replace("/", "__")
        packet = dict(room=room, slug=slug, dataset=job["dataset"], frames=job["frames"], input_sha256=ih,
                      reference_sha256=rh, raster=list(model["images"].shape[-2:]))
        for name, data, key in (("model", model, "input_file_sha256"), ("reference", reference, "reference_file_sha256")):
            packet[key] = save_packet(Path("/prepared") / name / (slug + ".npz"), data)
        records.append(packet)
        print("Verified", room, packet["raster"], flush=True)
    seal(Path("/receipts") / (stage + "_inputs.json"), dict(study="tco_two_track_v1", source_sha256=sha(source), rooms=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=WINDOWS)
    prepare(parser.parse_args().stage)
