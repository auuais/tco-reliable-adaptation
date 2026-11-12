# Controlled multi-view MASt3R neural adaptation

## Question and scope

Does decoder adaptation add reconstruction value beyond calibrated geometric alignment, and does the contribution survive an independently corrupted camera-prior channel? This is a fixed development screen on Heads, Chess and Fire, both existing sparse grids, the five frozen RGB targets per scene and the pinned metric checkpoint. It is not a novel-method claim or independent confirmation. All work is local; the USD 25 rental allocation remains reserved.

The zero-adapter bridge must reproduce ordinary MASt3R exactly after fresh float32 inference, descriptor matching and 300-coarse/300-fine alignment. Training uses cached frozen encoder features and bfloat16 decoder autocast, with the original float32 heads. Final inference and matching are recomputed in float32 for every adapted state. Do not substitute pretrained descriptor matches or confidences for adapted predictions.

## Fixed objectives and controls

All neural arms use rank-4, alpha-1 adapters in the 144 decoder attention projections, a frozen encoder and heads, non-reentrant decoder checkpointing and fresh Adam state. Reset all initial adapter tensors and RNG to seed 42 for each independent case. Take 40 updates at LR 0.0001 and clip the joint adapter gradient norm at 1. Record the unclipped norm and every component. Each update uses two ordered pairs sharing the first image. Cycle through reference i=0..4 and offsets 1..4 twice; j=(i+offset) mod 5, k=(j+1) mod 5, advancing k once if it equals i.

For each reference image retain native-grid sparse pixels whose initial self-depth is finite and positive for all four partners. Fix a per-pair metric scale from the median measured/predicted sparse ratio and a per-view unit from median measured depth. No scale, support or masking decision changes during training. All arms use sparse axial-depth Huber fitting plus 0.1 times cross-pair metric-point consistency. Evaluate consistency on the fixed image grid starting at pixel 8 with stride 16. Normalize both by 0.02 times the metric unit, with Huber transition 1.

For supplied C2W poses, T21=inverse(C2) C1. The second-camera point is R21(s P21)+t21; the first-camera point is s P11. Compare normalized point directions with normalized known-K rays, with residual divisor 0.02. All fixed grid points participate, including behind-camera predictions. The intrinsic-only penalty uses first-camera points. The cross-camera penalty uses second-camera points and therefore the supplied relative pose. This makes intrinsic-only adaptation an explicit control for the joint camera/calibration penalty.

| Neural arm | Depth + consistency | Intrinsic direction weight | Cross-camera direction weight | Pose channel |
|---|---:|---:|---:|---|
| D | yes | 0 | 0 | unused for supervision |
| K | yes | 0.01 | 0 | unused for supervision |
| J | yes | 0.01 | 0.01 | supplied clean |
| N | yes | 0.01 | 0.01 | controlled error |

The calibration is the established 525-pixel registration model, the existing 1.12 image zoom, and the center-preserving resize to 512 by 384. The same K and measurements are available to every comparison. Noise keeps view zero fixed, right-multiplies each other C2W rotation by a random-axis 5-degree rotation, and shifts its center in a random direction by 2% of the clean trajectory diameter. Use NumPy default_rng seeds 20261112, 20261113 and 20261114 by scene, and publish both pose packets. Depth packets and intrinsics stay fixed. This is controlled inconsistency in the pose channel, not a fully simulated noisy RGB-D/IMU system.

## Alignment and complete denominator

Define KD as fine alignment with the earlier 0.01 sparse-depth penalty plus a 0.01 known-intrinsics penalty. Normalize fx/fy errors by 2% of their known values and cx/cy errors by 2% of image width/height. KDP additionally uses the earlier relative-pose penalty at weight 0.01, with clean or corrupted poses. The ordinary alignment objective and all optimizer settings remain unchanged otherwise.

For each scene compute one zero-adaptation ordinary reconstruction. On each sparse grid compute zero-adaptation KD, KDP-clean and KDP-noisy. This gives 21 zero-adaptation reconstructions. For each scene/grid train D, K, J and N independently, giving 24 adapted states and 960 updates. D and K each receive ordinary, KD, KDP-clean and KDP-noisy alignment. J receives ordinary, KD and KDP-clean; N receives ordinary, KD and KDP-noisy. These give 84 adapted reconstructions, for **105 total reconstructions and 210 reference evaluations**. Include the earlier exact-input ordinary/depth/joint alignment results as historical controls, labeling any clean-pose oracle as ineligible in noisy-channel comparisons.

## Execution and interpretation

Require the real two-pair smoke backward pass to succeed before the batch. Check exact zero-update feature-path fidelity and unchanged original-state hashes. Preserve adapter states, full traces, gauges, sparse support, pose packets, native outputs, source hashes, execution commits, timings and failures. Stop on an unexpected failure, preserve the failed attempt, and do not retry automatically. Resuming a completed case requires matching all completion hashes. Return each incomplete attempt for explicit engineering review.

Complete all 105 native outputs and seal all point-map hashes before opening dense references. Use the existing registered and released references, masks and distance evaluator. The primary candidates are J+KDP-clean and N+KDP-noisy. Compare each with the strongest eligible zero-adaptation and D/K control under that same pose condition. Report every other alignment ablation, accuracy/completeness separately, scene gains, and the second reference. A descriptive positive screen requires at least 5% gain, no scene regression, each distance component no more than 2% worse, and no aggregate regression under the second reference, on both grids and both conditions. This is not a significance test or automatic spending authorization. Independent execution repeats, more noise draws and later untouched evaluation are still required before claiming reliability.

Publish failures and null results. Preserve all previously failed advancement decisions. Sparse fitted loss, cross-pair agreement and a changed network output are not substitutes for measured reconstruction quality. Full learned adaptation here differs from the released TCO rendering objective and from official Test3R visual prompts; those remain explicit comparison and scope limitations.
