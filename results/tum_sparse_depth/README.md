# Cross-dataset inputs

Three public Freiburg 3 archives passed complete gzip integrity checks. Timestamp-only selection produced fifteen target RGB frames and thirty neighboring source depths. The two fixed grids contain 3,042 usable projected measurements in total. Per-target coverage ranges from 91 to 120 samples. Target-associated depth files are disjoint from the source depth files and remain evaluation inputs.

The prepared RGB tensors and sparse packets are ready locally. Model inference and score computation remain pending the held-out outcome. No method settings were selected from TUM depth labels. The supplied trajectories come from the benchmark, so this tests the same known-pose condition as the successful VGGT development experiment.

Run `python scripts/verify_tum_sparse_inputs.py` to validate the public timestamp, pose, calibration and measurement records. To reconstruct the complete inference inputs, run `prepare_tum_sparse_depth.py`, `stage_tum_sparse_inputs.py`, then `prepare_tum_sparse_packet.py` under scripts, using the same local destination argument. The downloader records archive checksums and copies only timestamp and trajectory metadata out of each archive. Packet preparation opens the selected RGB and source-depth images.
