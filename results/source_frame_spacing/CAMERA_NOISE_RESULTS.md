# Camera-noise direction repeats

The original supplied pose arrays, fixed-depth maps and tangent-corrected maps reproduced exactly for seed 137 in all three scenes. We then changed only the random directions of the declared one-degree rotations and 2%-of-rig-diameter translations. Each draw reused the same RGB images, frozen model heads, first unperturbed camera and evaluation surfaces. Scores below are the equally scene-weighted percentage decrease in combined surface error for the coupled update. [Frozen protocol](../../docs/CAMERA_PRIOR_NOISE_REPEATS.md), [input priors and sealed maps](prior_noise_predictions.json) and [36 new canonical evaluations](prior_noise_scores.json).

| Noise seed | Registered vs no prior | Released vs no prior | Registered vs same pose, fixed depth | Released vs same pose, fixed depth |
|---|---:|---:|---:|---:|
| 137 (original) | 1.90% | 2.63% | 8.37% | 3.62% |
| 211 | 3.96% | 3.67% | 9.24% | 5.88% |
| 307 | 2.04% | 2.54% | 7.83% | 4.48% |
| 401 | 3.72% | 4.03% | 6.14% | 3.81% |

All nine new scene-by-draw comparisons improved the combined surface score over the no-prior ensemble on both reference conventions. Across scenes, registered accuracy distance worsened by 0.19 to 0.51 mm per new draw, completeness distance improved by 1.20 to 1.61 mm and normal consistency fell by 0.022 to 0.029. The released-reference averages had the same directions. Relative to the historical frozen comparator the registered score improved by 1.22% to 3.16% across these draws. None met the original advancement criteria.

Corresponding depth gains versus the no-prior ensemble were -0.55%, +0.85% and -0.01% for seeds 211, 307 and 401. Camera-frame normal error increased by 3.89, 3.46 and 3.14 degrees, respectively. These repeats support a persistent completeness-versus-accuracy tradeoff under this noise model. They add no independent scenes or sensors. [Per-target depth, normal and pose errors](prior_noise_paired.json).

## Direction specificity

Reversing the same per-camera rotation and translation moves worsened corresponding depth MSE relative to the supplied direction on all four draws. Moving toward the supplied pose improved this measure by 1.95% to 4.69% against reversal. Yet only seeds 211 and 401 favored the supplied direction on the scene-averaged whole-surface score on both references. Seeds 137 and 307 favored reversal on both. Thus the repeated composite gain over the no-prior model does not consistently distinguish the direction supported by the synthetic measurement. [Control protocol](../../docs/CAMERA_PRIOR_DIRECTION_CONTROL.md), [canonical comparisons](prior_direction_scores.json) and [paired pixel comparisons](prior_direction_paired.json).

Across draws, 83.4% to 86.1% of the increase in corresponding normal error came from stencils entirely inside the updated region, which comprised roughly 83% of valid stencils. Boundary stencils contributed the remainder. The next local test should check whether estimating a plane from a neighborhood reduces the interior error while retaining depth accuracy. [Per-target region decomposition](prior_noise_normal_regions.json).
