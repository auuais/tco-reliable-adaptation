# RGB tracks can distinguish many focal metadata errors from consistent image zooms

The same supplied camera matrix can describe a faulty focal prior or a legitimate zoom. On 270 matched pairs in nine development windows, a two-fold RGB check corrected **201 of 270 metadata faults (74.4%)** and incorrectly edited **4 of 270 valid zooms (1.5%)**. It left all nine unchanged windows alone. Mean absolute log-focal error fell **77.20%** relative to each unconditional baseline. This met the criteria for a separate TCO gradient test. [Preregistered design](../../docs/FOCAL_METADATA_CONTROL.md) · [All decisions, scores and abstentions](summary.json)

| Decision rule | Metadata faults corrected / 270 | Valid zooms wrongly changed / 270 | Mean absolute log-focal error |
|---|---:|---:|---:|
| Keep every supplied focal | 0 | 0 | 0.06421 |
| Always replace the outlier with its neighbors' median | 270 | 270 | 0.06421 |
| Require RGB evidence on one track fold | 203 | 5 | **0.01455** |
| Require agreement on both track folds | 201 | **4** | 0.01464 |

Requiring both folds prevented one wrong zoom edit and missed two additional metadata faults. The one-fold rule had slightly lower average error. Both folds had sufficient matching support for 81.1% of the cases. The largest relative error reduction was on Chess (95.36%) and the smallest on Heads (54.45%). The paired data deliberately give metadata faults and valid zooms equal weight. Applying an edit to a valid zoom could carry a different cost in deployment. [Scene and severity breakdown](summary.json)

Every pair used the same five supplied cameras, the same match IDs, and the same field-of-view intersection. For a legitimate zoom, both the focal length and the observed coordinates changed around the principal point. For faulty metadata only the supplied focal changed. The proposed correction used the other four cameras' median focal lengths. RGB tracks determined whether to apply it. The focal multipliers were 1.05, 1.12 and 1.25 and their reciprocals.

The test used transformed point coordinates from fixed-lens RGB-D sequences. It assumes one focal outlier, four cameras sharing a common focal calibration, and the supplied calibrated poses. It has yet to test feature re-detection after image resampling, uncertain poses, genuinely varying optics, or reconstruction accuracy. The predeclared next step is a local gradient test with pose uncertainty and fresh image matching. Cloud cost was $0. The transfer allocation remains reserved. [Code and input hashes](../../manifests/focal_metadata_control.json)

For a full replay, run `python scripts/focal_metadata_control.py run --archives /path/to/7scenes-zips` in a scratch execution checkout with the result file absent. The portable command `python scripts/verify_focal_metadata_control.py` verifies every archived edit decision, the paired grid, baseline identity, and summary arithmetic. Rebuilding with Windows NumPy 1.25.2 and Linux NumPy 2.1.2 produced identical decisions; three summary values differed by at most 1.12 × 10⁻¹⁶ when recomputing from the published rows. [Public input instructions](../sensor_public_replay/README.md)
