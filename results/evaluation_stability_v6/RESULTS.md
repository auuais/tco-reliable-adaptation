# CPU evaluation stability

All **24 frozen evaluations completed**: four saved prediction sets, three fresh processes each at one and four numerical-library threads. The original prediction/reference file hashes were checked before and after evaluation. The v4 and v5 evaluation functions have identical parsed code. Original published metrics remain unchanged.

| Saved prediction | Maximum relative range of either distance metric | Normal-consistency absolute range |
|---|---:|---:|
| v5 Heads, visual-only, seed 123 | 1.07e-13 | 2.90e-12 |
| v5 Fire, visual-only, seed 42 | 4.29e-13 | 1.45e-13 |
| v5 Fire, both priors, seed 42 | 1.32e-12 | 4.86e-13 |
| v4 Fire, visual-only, seed 42 | 1.74e-13 | 1.97e-13 |

Every comparison is below the predeclared 1e-6 relative distance / absolute normal-consistency thresholds. The largest absolute distance difference from an original published metric is 2.46e-14 metres. Numerical variation from this evaluator and these thread choices cannot explain the previously observed centimetre-scale changes.

For example, the stored Fire visual-only seed-42 predictions reproduce their respective original accuracy errors: approximately 1.2135 cm for v4 and 1.9797 cm for v5, despite an identical reference tensor digest. The next diagnostic should therefore inspect prediction generation, adapter initialization, random states and adaptation trajectories. This result does not isolate the cause to GPU hardware or certify every possible platform/input.

This was a CPU-only local study with zero RunPod compute expenditure. It used NumPy 2.1.2, Open3D 0.19.0 and PyTorch 2.7.1 in the existing Linux container. All individual metrics, alignment transforms, library/thread metadata and hashes are in [summary.json](summary.json); the precommitted design is [evaluation_stability_v6.json](../../manifests/evaluation_stability_v6.json).

Decision: proceed with the separately frozen same-device repeatability and common-reference calibration batch within the authorized USD 3 stage cap. The later MASt3R/held-out/verification allocations remain gated on a redesigned method with reliable evidence.
