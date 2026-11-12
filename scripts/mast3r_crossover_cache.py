"""Materialize explicit geometry/confidence/matching operands for cached alignment."""
import hashlib
import itertools
from pathlib import Path
import shutil

import torch

from plan_mast3r_crossover import MATCH_FOLDER, require, sha256


def directed_paths(names):
    identifiers = [hashlib.md5(name.encode()).hexdigest() for name in names]
    require(len(names) == len(set(names)), "Duplicate view names")
    return [(f"forward/{a}/{b}.pth", f"{MATCH_FOLDER}/{a}-{b}.pth")
            for a, b in itertools.permutations(identifiers, 2)]


def validate_pairs(cache, names):
    expected = {path for pair in directed_paths(names) for path in pair}
    actual = {p.relative_to(cache).as_posix() for folder in ("forward", MATCH_FOLDER)
              for p in (cache / folder).rglob("*.pth")}
    require(actual == expected, f"Cache ordered-pair inventory differs in {cache}")
    return len(expected) // 2


def operand_digest(values):
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def materialize(geometry, bundle, target, names, inventory):
    """Create a fresh cache, with provenance for every consumed file."""
    require(not target.exists(), f"Output cache already exists: {target}")
    validate_pairs(geometry, names)
    validate_pairs(bundle, names)
    target.mkdir(parents=True)
    records = []
    for forward, matching in directed_paths(names):
        for folder, path in ((geometry, forward), (bundle, forward), (bundle, matching)):
            expected = inventory[folder.name][path]
            require(sha256(folder / path) == expected, f"Changed source operand: {folder / path}")
        xyz = torch.load(geometry / forward, map_location="cpu", weights_only=True)
        observations = torch.load(bundle / forward, map_location="cpu", weights_only=True)
        require(len(xyz) == len(observations) == 4, "Unexpected forward schema")
        mixed = (xyz[0], observations[1], xyz[2], observations[3])
        require(all(torch.isfinite(v).all() for v in mixed), "Nonfinite forward operand")
        destination = target / forward
        destination.parent.mkdir(parents=True, exist_ok=True)
        if geometry == bundle:
            shutil.copyfile(geometry / forward, destination)
        else:
            torch.save(mixed, destination)
        output = torch.load(destination, map_location="cpu", weights_only=True)
        require(all(torch.equal(a, b) for a, b in zip(output, mixed)), "Mixed cache changed values")
        destination_match = target / matching
        destination_match.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / matching, destination_match)
        require(sha256(destination_match) == inventory[bundle.name][matching], "Matching copy changed")
        records.append(dict(forward=forward, matching=matching,
                            xyz_sha256=operand_digest([output[0], output[2]]),
                            confidence_sha256=operand_digest([output[1], output[3]]),
                            forward_sha256=sha256(destination), matching_sha256=sha256(destination_match)))
    require(validate_pairs(target, names) == len(names) * (len(names) - 1), "Missing directions")
    return records
