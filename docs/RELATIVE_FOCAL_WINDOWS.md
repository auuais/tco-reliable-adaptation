# New frame windows for the relative focal experiment

Repeat the registered camera-relative rule on frame IDs 25, 225, 425, 625 and 825 from each of Heads/seq-01, Chess/seq-03 and Fire/seq-03. The 15 images are distinct from the offset-0 redesign windows and the offset-50/100/150 earlier metadata tests. The three sequences are already known development data. Keep all held-out scenes sealed. Hash the original RGB and pose files and complete resized RGB tensors before looking at any new predictions.

Use the parent study's resampling grid, fresh two-fold DISK/LightGlue matching, original CPU FP32 VGGT camera head, seed, camera reference, pose perturbations, 465-condition grid and error definitions. Run the 93-config PoseLib comparator on the newly detected tracks. Evaluate both relative VGGT plus two geometric folds (the registered candidate) and relative VGGT alone (the stronger simple comparator on the design images). Freeze all directions, margins, shared-bias estimator and candidate targets. Keep abnormal cases, abstentions, hidden changes, per-scene errors and both pose seeds in the record. Report any loss of performance from the design windows.

For the registered method repeat the exact advancement checks in the previous design. For the simpler model-only comparison apply the same 3/60 valid-edit ceiling, 20% reduction against the better unconditional gradient, no scene worse than its better unconditional baseline, and primary gradient no worse than the raw model plus both folds. If both pass, prioritize a method requiring fewer inputs and less compute only if its primary gradient is also no worse. If a baseline already explains the effect, describe that explicitly. A pass supports an end-to-end dense development pilot. A held-out, cross-dataset or MASt3R result requires further evidence.

## Replay the collection

Use the original environment, pinned checkpoint and matching assets in the [parent replay guide](REPLAY_FOCAL_ZOOM.md). Run from the repository root.

```sh
python scripts/collect_relative_focal_windows.py camera --archives /data/archives --checkpoint /data/checkpoints/VGGT-1B/model.safetensors --threads 5
python scripts/collect_relative_focal_windows.py tracks --archives /data/archives --support /support --threads 2
```
