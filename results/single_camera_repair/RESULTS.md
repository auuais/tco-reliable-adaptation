# Selective camera repair missed its advancement criteria

The primary repair reduced average camera projection error from **4.59 to 4.35 pixels (5.3%)**, below the prespecified 20% reduction. It worsened rotation-only and focal-only inputs. On Heads, error rose from 4.60 to 6.33 pixels. Two clean evaluations, opposite track folds of the same Heads window, triggered incorrect edits. The larger error was 8.67 pixels against a 1-pixel clean limit. The camera-repair branch stops at this CPU test. [All outcomes and decision checks](summary.json) · [Frozen criteria](../../manifests/single_camera_repair.json)

| Fault in one of five cameras | Unchanged (px) | Alert and validation (px) | Always edit (px) | Known faulty index (px) |
|---|---:|---:|---:|---:|
| Rotation, 3° | 3.96 | 4.06 | 4.41 | 3.06 |
| Center displacement, 6% of trajectory diameter | 3.98 | 3.81 | 4.06 | 3.09 |
| Focal length, ×1.12 or ÷1.12 | 3.74 | 4.11 | 4.51 | 3.34 |
| All three on one camera | 6.67 | 5.40 | 5.71 | 4.11 |

We selected and fitted one suspect camera using one connected-track fold, then evaluated the proposed adjustment on disjoint RGB matches. All 266 alerted proposals in corrupted cases also passed the held-back epipolar check. That check added no protection to the alert rule in this sample. Providing the correct faulty-camera index to the same optimizer reduced overall error to 3.40 pixels, a 25.9% reduction. This privileged comparison diagnoses the cost of selecting the wrong camera and of insufficient support. It is unavailable at inference. [Repair implementation](../../scripts/repair_one_camera.py) · [Localization results](../single_camera_fault_localization/RESULTS.md)

The offline error measures where cameras project a fixed grid at depths of 1, 2 and 4 metres, averaged over all five views. Synthetic clean camera labels enter this measurement only. We evaluated nine development windows, five possible fault locations, four fault types and two track-fold directions, giving 360 corrupted and 18 clean evaluations. These share images and scenes. No neural adaptation or dense point-cloud evaluation was performed in this test. Additional cloud cost was **$0**.

Run `python scripts/screen_single_camera_repair.py verify` to check the published arithmetic. A scratch checkout with the published result folder moved aside can regenerate the table using `python scripts/screen_single_camera_repair.py run --archives /path/to/7scenes-zips` in the project environment. [Public-data replay instructions](../sensor_public_replay/README.md)
