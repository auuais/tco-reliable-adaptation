# Fixed RGB calibration improves pixel agreement on nine new development windows

Changing the camera frame and intrinsics together lowered the median registered-depth reprojection error in all nine five-frame windows. Averaging the nine medians gave **13.03 px with the released convention and 5.29 px with the fixed RGB calibration**, a 59.4% reduction. Depth-independent Sampson distance also fell in every window. The 45 new images came from the same three development sequences previously studied. [Window-level results](summary.json)

| Camera convention | Mean window median Sampson distance (px) | Mean window median depth reprojection error (px) |
|---|---:|---:|
| Released poses and intrinsics | 3.95 | 13.03 |
| Change intrinsics to registration calibration | 1.65 | 5.96 |
| Also compose physical depth-to-RGB transform | **1.55** | **5.29** |
| Use transform fitted on two other scenes | 1.65 | 5.66 |
| Compose the inverse physical transform as a control | 1.99 | 7.67 |

The intrinsics change accounts for most of the overall decrease. Adding the physical transform at the same intrinsics lowered the depth reprojection median in six windows and increased it in three. Fitting a transform improved four windows over the supplied physical transform and worsened five. For example, on Fire the fitted transform increased mean window-median reprojection error from 6.06 to 7.51 px. A single shared rigid correction therefore explains only part of the disagreement on these sequences. [Scene averages and paired window counts](aggregates.json)

Each sequence contributed three windows beginning at frame 50, 100 or 150, with five images spaced 200 frames apart. Each image was new relative to the transform-fitting data. DISK and LightGlue retained 62 of 90 candidate image pairs after geometric filtering, yielding 10,211 correspondences. Every method used the same matches. Depth reprojection uses directed observations with valid registered depth and positive depth in the target view. The raw table includes all attempted pairs, depth-valid and positive-depth counts, camera-file hashes and RGB hashes. The scene means give each window equal weight. [Frozen protocol](../../manifests/sensor_windows_v16.json) · [Image-matching protocol](../../manifests/learned_support_v8.json)

These metrics measure agreement between camera geometry and RGB observations. Dense reconstruction accuracy, effects of applying the correction during TCO optimization, and generalization to a different camera rig require separate experiments. The computation used the local CPU and cost **$0 in RunPod charges**. [Source and evaluation](../../scripts/check_sensor_windows.py) · [Aggregate verifier](../../scripts/summarize_sensor_windows.py)
