# Reproduce the RGB calibration results from the original dataset

This CPU replay reconstructs registered depth directly from the original Microsoft depth PNGs. It opens the archived camera poses, verifies the identity of all 45 RGB frames and poses, and evaluates the published image correspondences. Both the original scene archives and the extracted sequence archives reproduced all **45 window-by-camera conditions**. All integer counts matched, and the largest difference between corresponding median pixel errors was **0.000026 px**. [Replay output](replay.json) · [Study results](../sensor_windows_v16/RESULTS.md)

1. Download the Heads, Chess and Fire ZIPs into one folder and retain the names `heads.zip`, `chess.zip` and `fire.zip`. The script reads the nested sequence ZIPs directly. [Microsoft 7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
2. From a checkout of this repository, install NumPy, Pillow and `opencv-python-headless` in a Python environment. The replay uses the frozen 2D correspondences included in the repository. The original matcher and a GPU are unnecessary for this calculation. [113 KB of image correspondences](rgb_matches.zip)
3. Run the command below and choose a new output filename. The script fails if a file hash, any integer count, or any median beyond a tolerance of 0.0002 px differs.

```sh
python scripts/replay_sensor_windows_from_zips.py --archives /path/to/7scenes-zips --output my-replay.json
```

The comparison uses registered RGB axial depth for reprojection. It also calculates depth-independent Sampson distances from the measured 2D matches and camera matrices. This checks the camera-consistency result from public inputs. The dense point-cloud scores and the TCO adaptations have their own evaluation procedures. [Replay implementation](../../scripts/replay_sensor_windows_from_zips.py) · [Matching and camera conventions](../../manifests/sensor_windows_v16.json)
