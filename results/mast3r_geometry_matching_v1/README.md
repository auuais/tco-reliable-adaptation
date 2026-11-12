# MASt3R geometry and matching crossover

Neither hybrid meets the registered nomination checks. The decision is a local no-go for the uniform freezing strategies tested here.

Coordinates give a larger isolated gain than the observation bundle in every aggregate setting. Full adaptation gives the lowest aggregate error in all eight grid/pose/reference comparisons. On Fire, G1B0 improves on full adaptation in all four registered-RGB settings. Heads and Chess favor full adaptation. The direction of the observation effect therefore varies by scene.

All 48 alignments completed on the local RTX 3080. All 24 archived endpoint controls pass the registered tolerance, with 24 reproducing every depth, intrinsic and camera value exactly. The 24 crossed combinations were scored after all outputs were sealed. There are 96 reconstruction/reference associations across registered RGB and released references.

G0B0 uses pretrained coordinates and observations. G1B1 uses both from the sparse-depth adapter. G1B0 combines adapted coordinates with pretrained observations. G0B1 combines pretrained coordinates with adapted observations. Observations include point confidences, pixel matches, matching weights and their effects on the view tree.

Scores average accuracy and completeness after normalizing each component to ordinary MASt3R for the same scene and reference. Each scene receives equal weight. Lower scores indicate lower surface error.

| Grid and poses | Reference | G0B0 | G1B1 | G1B0 | G0B1 | Geometry gain | Observation gain | Interaction | Resolution |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centered_clean | registered_rgb | 0.918545 | 0.896762 | 0.900548 | 0.911924 | +1.959% | +0.721% | +0.002835 | 1e-06 |
| centered_clean | released | 0.931087 | 0.911344 | 0.914341 | 0.926272 | +1.799% | +0.517% | +0.001818 | 1e-06 |
| centered_noisy | registered_rgb | 0.920157 | 0.897724 | 0.902168 | 0.914031 | +1.955% | +0.666% | +0.001681 | 1e-06 |
| centered_noisy | released | 0.930560 | 0.910494 | 0.913271 | 0.926336 | +1.858% | +0.454% | +0.001448 | 1e-06 |
| shifted_clean | registered_rgb | 0.905832 | 0.884677 | 0.888851 | 0.902961 | +1.875% | +0.317% | -0.001303 | 1e-06 |
| shifted_clean | released | 0.921131 | 0.900785 | 0.902895 | 0.919184 | +1.980% | +0.211% | -0.000164 | 1e-06 |
| shifted_noisy | registered_rgb | 0.910043 | 0.887977 | 0.889534 | 0.904538 | +2.254% | +0.605% | +0.003948 | 1e-06 |
| shifted_noisy | released | 0.922853 | 0.901361 | 0.902076 | 0.919017 | +2.251% | +0.416% | +0.003121 | 1e-06 |

Interaction is S11 − S10 − S01 + S00. A positive value means the joint error exceeds the additive prediction. Geometry and observation gains in the table are relative to G0B0. The resolution column is the registered margin derived from archived endpoint replay errors.

| Branch | Condition and reference | Nomination check |
|---|---|---|
| geometry | centered_clean/registered_rgb | fail |
| geometry | centered_clean/released | fail |
| geometry | centered_noisy/registered_rgb | fail |
| geometry | centered_noisy/released | fail |
| geometry | shifted_clean/registered_rgb | fail |
| geometry | shifted_clean/released | fail |
| geometry | shifted_noisy/registered_rgb | fail |
| geometry | shifted_noisy/released | fail |
| observations | centered_clean/registered_rgb | fail |
| observations | centered_clean/released | fail |
| observations | centered_noisy/registered_rgb | fail |
| observations | centered_noisy/released | fail |
| observations | shifted_clean/registered_rgb | fail |
| observations | shifted_clean/released | fail |
| observations | shifted_noisy/registered_rgb | fail |
| observations | shifted_noisy/released | fail |

The three scenes are development observations. Both grids and repeated pose conditions reuse their images. The match audit reports shared, gained and lost pairs using fixed benchmark geometry after sealing, with visibility checks and equal weighting by unordered image pair.

The first bridge attempt used PyTorch's default TF32 setting. Its native arrays failed tolerance. A direct run against the original unmodified cache reproduced that mismatch exactly. All five recomputed canonical views matched their archived tensors exactly. CroCo's model import enables TF32, and explicitly restoring this parent setting repaired the bridge. The failed attempt, its traces and diagnosis are retained under `attempts/tf32-disabled01/` and `diagnosis/`.

The first match audit failed its projection check because the supplied pose packet uses matrix inversion while the reference builder uses the loader's transpose-based inversion. The source rotations have small orthogonality errors. Reproducing the reference builder's camera matrices brought the same-view maximum error below 0.00006 native pixels. The diagnostic uses these archived-reference cameras. Both surface score definitions use their original frozen reference archives.

Additional RunPod compute for this phase is USD 0. The reserved USD 25 remains available for a subsequent decision. Total alignment, materialization time and peak memory are recorded per case. These cached diagnostics cover the alignment stage. A deployable method must also account for model inference, training and matching.

Run `python scripts/replay_mast3r_crossover.py` for the portable evidence checks. The [registered protocol](../../docs/MAST3R_GEOMETRY_MATCHING_GOAL.md), [96 scene rows](scene_metrics.csv), [metric identities](associations.json), [contrasts and decisions](scores.json), [match diagnostics](match_audit.json) and [two-drive archive](archive.json) provide the supporting records.
