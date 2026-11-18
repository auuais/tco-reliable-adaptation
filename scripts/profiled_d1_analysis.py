"""Replay the four registered contrasts from sealed, model-free data."""
from itertools import permutations

STATES = ("B0", "B1", "F0", "F1")
CONTRASTS = (("B0", "B1"), ("F0", "F1"), ("B0", "F0"), ("B1", "F1"))
PAIRS = tuple(permutations(range(5), 2))
FIELDS = ("A", "b", "Q", "d", "ridge")


def key(fold, state, field):
    return f"fold{fold}_{state}_{field}"


def analyze_matrices(matrices):
    import numpy as np
    from profiled_d1_math import analyze_response
    expected = {key(fold, state, field) for fold in (0, 1)
                for state in STATES for field in FIELDS}
    if set(matrices) != expected:
        raise ValueError("Matrix inventory must contain every registered fold and state")
    rows = []
    for fold in (0, 1):
        fixed = {field: matrices[key(fold, "B0", field)] for field in ("b", "d", "ridge")}
        for state in STATES:
            for field in FIELDS:
                value = matrices[key(fold, state, field)]
                if value.dtype != np.float64:
                    raise ValueError("Sealed matrix fields must retain float64")
                if field in fixed and not np.array_equal(value, fixed[field]):
                    raise ValueError("Targets and ridge must remain fixed across all states")
        for base, final in CONTRASTS:
            response = analyze_response(
                matrices[key(fold, base, "A")], matrices[key(fold, base, "Q")],
                matrices[key(fold, final, "A")], matrices[key(fold, final, "Q")],
                fixed["b"], fixed["d"], float(fixed["ridge"]),
                with_derivative=(base, final) == ("B0", "B1"))
            rows.append(dict(fold=fold, contrast=f"{base}_to_{final}", response=response))
    return rows


def depth_statistics(depths):
    import numpy as np
    from profiled_d0_math import compare_depths, grid_coordinates, aggregate_equal_pairs
    expected = {f"xy_{i}" for i in range(5)}
    expected.update(f"{state}_{i}_{j}" for state in STATES for i, j in PAIRS)
    expected.update(f"{state}_fused" for state in STATES)
    if set(depths) != expected:
        raise ValueError("Depth inventory differs from the registered four states")
    for name in expected:
        if name.startswith("xy_"):
            continue
        expected_shape = (5, 384, 512) if name.endswith("_fused") else (384, 512)
        if depths[name].shape != expected_shape or depths[name].dtype != np.float32:
            raise ValueError("Saved depth raster violates registered shape or dtype")

    grid = grid_coordinates()
    rows = []
    for base, final in CONTRASTS:
        for level, indices in (("partner", PAIRS), ("fused", tuple((i, None) for i in range(5)))):
            records = []
            for i, j in indices:
                first, second = (depths[f"{s}_fused"][i] if j is None else depths[f"{s}_{i}_{j}"]
                                 for s in (base, final))
                records.append(dict(view=i, partner=j,
                    grid=compare_depths(first, second, grid),
                    adaptation=compare_depths(first, second, depths[f"xy_{i}"])))
            rows.append(dict(contrast=f"{base}_to_{final}", level=level, records=records,
                grid_mean=aggregate_equal_pairs(r["grid"] for r in records),
                adaptation_mean=aggregate_equal_pairs(r["adaptation"] for r in records)))
    return rows
