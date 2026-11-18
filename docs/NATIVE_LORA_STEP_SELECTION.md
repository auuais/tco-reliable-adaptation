# Choosing a MASt3R LoRA step from adaptation depth

The earlier directional study found full-step gains and regressions, exact repeats, and two cases with zero held coverage in an Office view. This study compares a step chosen from independent adaptation observations against the same frozen-backbone MASt3R P3 direction at its fixed full step. It uses the local RTX 3080 and USD 0 in RunPod compute. [Prior directional study](../results/native_lora_direction_v1/README.md)

## Rooms and coverage

Use Stanford 12-Scenes `apt1/living`, `apt2/kitchen` and `office2/5b`. These room IDs were absent from the project's earlier evaluations. Office 1 remains reserved. Use the first metadata-eligible contiguous five-frame window under the original capture, stride, baseline and tracking rules. Build the earlier and later packets from the immediately neighboring source frames. Use the original axial-depth and camera projection conventions. Each source samples an 8 × 8 lattice. Earlier rays use the quarter-cell center on each axis, and later rays use the three-quarter-cell center. This gives the two packets different coordinates even for nearly stationary cameras.

Before GPU work, an input checker can read measurement packets to count coverage. It verifies at least 16 unique held pixels per view after removing any overlap with the entire adaptation packet, in both temporal directions. It also checks at least eight points per view in every adaptation partition. Freeze the room IDs, frame IDs, source bytes, packet hashes, coordinate mask hashes and counts before inference. This preflight has no model predictions or depth-error comparisons. Any candidate-dependent sample removal invalidates the result. [Dataset and calibration](https://graphics.stanford.edu/projects/reloc/)

## Step selection

For each temporal direction, use the existing provenance grouping by source ray and native 8 × 8 cell, with the original fold seed and fixed role suffixes. Split the adaptation packet into halves. Split the second half again and reserve its second group for step selection. Split the remaining observations into gradient support and query folds. Groups remain intact. The blind scoring packet is the opposite temporal source. The adaptation selector can see its own reserved coordinates and depths and the rendered candidates.

Compute the original, sensor-only P3 first Adam update with the same frozen model, 20 directed image pairs, rank 4, alpha 1, seed 42, gradient clipping at 1 and learning rate 0.0001. Give every candidate the same direction, initialization and model. Compare signed multipliers 0, +0.25, +0.5 and +1. Preserve the exact saved adapter factors at 0 and +1. The 24 endpoints each use fresh FP32 inference, matching and native 300/300 alignment with the existing camera/depth/pose prior settings. Alignment sees only support and query depth. Compute the state-specific global metric gauge from support and query alone.

Choose the smallest multiplier within 0.05 mm of the minimum equal-view selector MAE after native alignment. This allows skipping the update. Write choices with their inputs and prediction digests, and seal all choices and predictions before held scoring. Use the same predictions to evaluate fixed 0, 0.25, 0.5 and 1 on the same held pixels. Fixed 1 is the primary comparator. The other fixed multipliers help distinguish the value of selection from simply using a smaller step.

## Verdict and replay

Compute mean absolute axial depth error in millimeters per view, average views equally within each case, then average the three rooms equally within each temporal direction. Report all six paired changes. Register consistent evidence for the selection rule only if its mean error falls by at least 1% against fixed 1 in both directions, at least two of the same rooms improve by more than 0.05 mm in both directions, and no valid case regresses by more than 1%. Use all six cases for this decision. This is a fresh-room diagnostic with three paired rooms. Report all fixed baselines and actual effects even if the screen misses the criterion.

The two directions reuse the same RGB views and physical room. The reserved selection depth is distinct from the observations that determine the direction, native priors and scale. The outcome covers held axial depth in small viewpoint windows. It leaves dense surfaces, later optimizer steps and untouched environments for future evaluation.
