# Can image matches identify which camera components to trust?

The track-selected camera rule failed the predeclared development comparison. It used five RGB images per scene, predicted depth and cameras, candidate camera priors, and previously frozen DISK/LightGlue image matches. On clean inputs the candidate produced 12.79% higher normalized error than the fixed rule that retained half of each TCO depth change. On perturbed inputs its normalized error was 46.73% higher than the initial model.

| Method | Clean normalized error | Clean harmful / 3 | Perturbed normalized error | Perturbed harmful / 45 |
|---|---:|---:|---:|---:|
| Initial model | 1.0000 | 0 | 1.0000 | 0 |
| Model cameras with fitted depth scales | 1.0020 | 2 | 1.0020 | 30 |
| Direct supplied cameras | 0.9615 | 1 | 2.1648 | 45 |
| Supplied cameras with one fitted scale | 0.9507 | 1 | 2.2060 | 45 |
| Supplied cameras with fitted per-view scales | 1.0571 | 3 | 2.9729 | 45 |
| Track-selected camera components | 1.0302 | 2 | 1.4673 | 27 |
| Half of TCO depth change | 0.9134 | 0 | — | — |
| Unchanged TCO | 0.9344 | 1 | — | — |

The clean group has three deterministic scene cases. For each of the five corruption profiles, three perturbation draws on each scene give nine cases. The 45 noisy cases share three scenes. TCO and half-depth comparison numbers average three adaptation seeds per scene before the clean comparison. A harmful case exceeds its scene's matched initial error by over 1% in either accuracy or completeness. Lower normalized error is better. The dashes indicate missing matched noisy TCO runs.

| Supplied-prior condition | Selection score | Accuracy error (cm) | Completeness error (cm) | Normals |
|---|---:|---:|---:|---:|
| Released clean convention | 1.0302 | 1.7158 | 1.9839 | 0.7460 |
| 5° rotation | 1.0644 | 1.7747 | 2.0180 | 0.7451 |
| 10% translation | 1.5917 | 2.6750 | 2.5507 | 0.7328 |
| 10% increased focal length | 1.0718 | 1.7920 | 2.0206 | 0.7469 |
| Moderate mixed errors | 1.6270 | 2.6999 | 2.6232 | 0.7318 |
| Severe mixed errors | 1.9819 | 3.1327 | 3.1619 | 0.7156 |

For each of eight camera combinations, the method triangulates axial depth from matches, estimates per-view depth scales on one track fold, and measures bidirectional pixel error on the other. It swaps the folds and selects by their mean validation error. Rotation, camera centers and intrinsics may each come from either the model or the external prior. Unit-scale and shared-scale controls for both full rigs also compete. The winning scales are refit on all tracks. The registered pose priors and camera estimates are expressed in the same first-camera coordinate frame. Every one of the 288 method-case outputs was sealed before dense evaluation. Identical maps within a scene share a score. There were 140 distinct maps.

With the prespecified illustrative 50/50 weighting of clean and perturbed inputs, the candidate's relative gain was -24.88% against the best single fixed baseline, initial model. This weighting describes an assumed deployment mix. The separate primary clean and perturbed comparisons are reported above. [Deployment-mixture sensitivity](mixture.json)

The dense evaluation used the released benchmark conventions. An offline sensor-frame audit found that composing RGB registration into the depth-sensor poses changes pixel agreement. Correcting the camera convention could affect both selection and dense reference scores. The current observations concern the three development scenes, with synthetic perturbations. A timing comparison and a noisy-input TCO comparison would be needed to establish speed or superiority over adapted TCO in those settings. [Sensor-frame comparison](../sensor_frame_v14/RESULTS.md)

[Frozen protocol](../../manifests/track_camera_selection_v13.json) · [All scores and subgroup decisions](summary.json) · [Prediction seal and fold errors](seal.json) · [RGB-track diagnostic](track_diagnostic.json) · [Archive record](archive.json)
