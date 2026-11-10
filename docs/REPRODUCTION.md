# Reproduction boundaries and evidence

**Comparator precision:** Our stored zero-step TCO maps were unprojected in a different arithmetic context from the adapted endpoints. A [three-scene control](../results/pointmap_precision_v9/RESULTS.md) isolates this effect with identical model heads. Do not attribute the entire normal-consistency change to adaptation or retroactively replace the published historical score table. This control does not establish a defect in the authors' published benchmark.

## Immutable reference

- TCO commit: `65387333877723cf99086f7f29089379dd50c59a`.
- VGGT checkpoint revision: `860abec7937da0a4c03c41d3c269c366e82abdf9`.
- Checkpoint SHA-256: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
- Historical baseline runner SHA-256: `32b8ad5002ade2e430496fb3d2c7bee66f46e116413b991ddfc156d05f11ae04`.
- Historical diagnostic runner SHA-256: `39db9c2226af38ecf954a64fd75e341900b7526fe5be160a603b40a6fba4f615`.
- Only upstream compatibility patch: cast CPU bfloat16 to float32 before NumPy export. This represents the bfloat16 values exactly; it does not alter inference or optimization.

The initial subset contains Heads 01, Chess 03/05 and Fire 03/04, each using original frame IDs 0, 200, 400, 600, 800. Images are loaded at 518, producing 392 by 518 pixels. The 40-step run uses fixed Adam learning rate 0.001, rank-4 LoRA and the released loss weights. `use_cos_weighting` is a Gaussian-rendering weighting option, not a cosine learning-rate schedule.

Reference camera poses and nominal intrinsics are supplied as priors. Depth and reference points are evaluation-only. Evaluation performs Umeyama Sim(3) alignment followed by rigid ICP at 0.1 m and nearest-neighbor error. These scores test aligned shape, not absolute scale, odometry drift, or SLAM performance.

## What was reproduced

All five original runs improved the three reported mean metrics. Their equal-sequence mean accuracy and completeness error reductions were 20.0% and 20.8%. All 200 adaptation updates were finite, and tracked base-parameter version counters were unchanged. This check is not a full cryptographic comparison of base weights.

The paper's complete benchmark and our selected subset have different sample populations. Their means cannot be compared as an exact reproduction check. The full pinned 7-Scenes sampling map contains 18 sequences; only five were initially executed.

## Numerical variation and adverse evidence

A subsequent clean bridge failed a predeclared 1% tolerance. The `diagnostics_v2` batch was aborted rather than relaxing that tolerance: 8 jobs completed, 1 aborted, 52 were unstarted. Its manifest remains available and its evidence remains archived locally.

Original-runner repeats also varied. Numerical probes found identical initial adapter tensors, sampled RNG states, and first scalar loss components but differing first clipped gradients. Disabling the multiview objective or enabling strict Torch determinism with the prescribed cuBLAS workspace did not remove the variation. The responsible operation has not been localized. Timing under additional instrumentation is not used for performance claims.

The completed `diagnostics_v3_repeats` study contains 24 fresh-process jobs: 2 scenes, 4 conditions and 3 same-seed execution repeats. All 780 updates were finite and all evidence checks passed. Severe corruption means random-axis 10-degree pose rotation, camera-center displacement of 20% of trajectory diameter, and a common +20% focal change; the first pose is fixed. This is our stress protocol, not a reproduction of the paper's noise table.

All individual metrics, condition means and observed ranges are retained. The descriptive harm threshold is accuracy OR completeness error exceeding the same-scene zero-step error by more than 1%. These counts are not estimated population failure probabilities.

## Archive provenance

Raw archives contain dataset-derived images and predictions and are not distributed here. Local verified SHA-256 values:

| Evidence | SHA-256 |
|---|---|
| Initial reproduction | `d29bf3d49ac4df9385d63f5fc560e4ce1407bbfbdd95937655d9b15c9c7bb058` |
| Aborted v2 | `652ad88b9f46dd1ef6852aaff1ad6d18dc51175184fee4fc706db8b4b25b2a2a` |
| Numerical investigation | `c4ce07f2f3b2fd289818f07505d2b28dfbc8cba1c75de13163fe994f63a06fa4` |
| Repeated v3 screen | `97c70db7c29735c1d74746241efcce0a689f6b38c5ec41877d2bdbb421292dd6` |

The first Git commit imports pre-existing records. It does not claim those historical experiments ran from that commit. New studies record an actual pre-execution code commit and frozen manifest hash.
