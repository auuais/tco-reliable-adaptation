# Reproducing the offset-25 camera experiment

Use the same public three scene ZIPs, original VGGT checkpoint, pinned TCO checkout, DISK/LightGlue weights and PoseLib 2.0.5 environment used on the first windows. The collection manifest records every input RGB and pose hash for the new frames. [Input and environment instructions](../../docs/REPLAY_FOCAL_ZOOM.md) · [Replication protocol](../../docs/RELATIVE_FOCAL_WINDOWS.md)

To regenerate the observations and scores, start with a fresh checkout of revision `9fb0885`. The score writer refuses to overwrite a completed evidence file. Collect the two observation types independently on CPU. Run the following from the repository root.

```sh
python scripts/collect_relative_focal_windows.py camera --archives /data/archives --checkpoint /data/checkpoints/VGGT-1B/model.safetensors --threads 5
python scripts/collect_relative_focal_windows.py tracks --archives /data/archives --support /support --threads 2
python scripts/score_relative_focal_windows.py pack
python scripts/score_relative_focal_windows.py score --archives /data/archives
python scripts/verify_relative_focal_windows.py --refit-solver --archives /data/archives
```

On a checkout already containing the public results, `python scripts/verify_relative_focal_windows.py --refit-solver` verifies the paired decisions, refits the classical camera solver on the public derived coordinates, and compares the stored TCO field-of-view derivatives with the analytical expressions. Add the archives option to replay both geometric track scores from the original camera poses. The stored rows include all metadata faults, legitimate zooms, hidden changes, uniform controls and pose severities. A repeated pose seed counts as a measurement of the same intervention.
