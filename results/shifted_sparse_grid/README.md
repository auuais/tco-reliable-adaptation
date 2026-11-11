# Disjoint sampling-grid results

The repeat passed all five registered checks. With supplied poses, depth-aware interpolation scored 0.85679 versus 0.91935 for spatial interpolation on the registered reference, a 6.80% improvement. Relative gains were 6.18% on chess, 8.45% on fire, and 4.72% on heads. The released-reference scores were 0.86807 and 0.89841. The centered-grid gain was 6.74%. Both grids used the same 15 targets, neighboring frames and supplied poses. This repeat supports transfer testing, while leaving generalization, pose-noise tolerance, and comparison with equally conditioned learned methods open.

Run `python scripts/replay_shifted_sparse_grid.py` for a public CPU replay. The full metric evaluation uses the original datasets and parent local reference archives. All computations used local hardware. The 12 full point maps are checksum-verified on two local physical disks.
