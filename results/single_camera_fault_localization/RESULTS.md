# Which camera supplied the bad prior?

We perturbed one camera at a time in nine five-view development windows from Heads, Chess and Fire. The detector saw the supplied poses, intrinsics and frozen RGB matches. For each camera it took the median epipolar error over its incident image pairs, with one vote per neighbor. We included all five possible fault locations, including the first camera. Each camera needed at least two matched neighbors to be eligible. Uniform guessing would identify the faulty camera 20% of the time. [Frozen design and fault magnitudes](../../manifests/single_camera_fault_localization.json)

| Injected fault | Top-one, released cameras | Top-one, calibrated RGB cameras | Alert and correct, calibrated RGB cameras |
|---|---:|---:|---:|
| Rotation, 5° | 59.3% | 75.6% | 66.7% |
| Center displacement, 10% of trajectory diameter | 69.6% | 79.3% | 72.6% |
| Focal length, ×1.10 or ÷1.10 | 39.6% | 71.1% | 45.9% |

Each percentage uses 135 injected conditions, covering all nine windows, five cameras and three deterministic direction or sign seeds. The same images and fault instances occur under both conventions. These are repeated synthetic interventions on three scenes. Under both conventions, 82.2% of fault locations had two matched neighbors. Ineligible locations stayed in the denominator. Top-one agreement under calibrated cameras ranged from 52.6% on Heads to 92.6% on Chess across fault types. [All 810 corrupted and 18 clean conditions](summary.json)

The alert threshold was the largest clean-window score observed in the other two scenes. Both conventions produced one false alarm in nine clean windows. The calibrated detector often ranked the faulty focal length first without exceeding that threshold. The experiment supports testing selective camera correction with abstention. A position ranking alone provides no evidence that editing a prior improves dense reconstruction. [Detector and alert rules](../../scripts/localize_camera_prior_faults.py)

## Reproduce on a CPU

The published score table can be checked from the checkout with `python scripts/localize_camera_prior_faults.py verify`. For a full replay, use a fresh scratch checkout, move its published `results/single_camera_fault_localization` folder aside, and run `python scripts/localize_camera_prior_faults.py run --archives /path/to/7scenes-zips` in the project environment. The script checks pose and match hashes and reproduces the clean epipolar scores. It reads the original pose text files and the published RGB correspondence archive. The original scene ZIPs and extracted sequence ZIPs produced byte-identical results with SHA-256 `56ffbf15357556a3cc24a921b3640e8c682a2024ec74fbcb1c5c556f2a6e18c5`. The additional cloud cost was **$0**. [Public RGB matches](../sensor_public_replay/README.md) · [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
