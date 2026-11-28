# Conditional Hypersim replication

Hypersim tests whether a five-view method selected on real rooms transfers to synthetic interiors from a different camera and rendering pipeline. A supported six-visit ARKitScenes result must precede any scene acquisition. The three-seed method, comparator step count, metrics, room-level weighting, and 1% accuracy and 3% completeness limits carry forward. Track B stopped at its development gate. Local compute costs USD 0 in RunPod charges. [Preregistered protocol](../manifests/tco_two_track_v1_hypersim.json)

## What is ready

The publisher release is pinned to commit `3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254`. Three source files totaling 3.32 MB identify the scene-level v1 split, per-scene camera matrices and archive URLs. Verification found 59,543 released train frames, 7,386 validation frames and 7,690 test frames across 457 scenes present in the split. The separate camera table contains 482 rows. A search at the frozen project-exposure commit found no earlier Hypersim scene IDs in the tracked manifests, results, or study documents. [Official split](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv) · [Publisher camera calibration](https://github.com/apple-aiml-research/ml-hypersim/tree/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/contrib/mikeroberts3000)

The publisher distributes one ZIP archive per scene, generally 1–20 GB. Header requests for the first train and test scene IDs returned lengths of 2,249,481,604 and 4,553,707,484 bytes and advertised byte ranges. The range reader accepts exact HTTP 206 responses bound to one ETag, verifies ZIP CRCs, hashes downloaded members and caches them locally. One request is limited to 32 MiB and one scene operation to 350 MiB of accepted range payload. Generated ZIP fixtures exercise member selection and refusal of a server that ignores a byte-range request. Actual archive range transfers, dataset frames, and a publisher-data geometry bridge await the scientific gate. [Official archive list](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/code/python/tools/dataset_download_images.py) · [Publisher-hosted selective download example](https://github.com/apple-aiml-research/ml-hypersim/tree/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/contrib/99991)

## Choosing and validating geometry

1. Choose the first eligible training scene and trajectory from the sorted publisher split for a five-frame bridge. Use camera positions, rotations, released frame IDs, source member availability and projection feasibility to select the window. The registered baseline interval is 0.05–0.75 m. Record all skipped candidates. For replication, repeat the same rule on six distinct test scenes after the bridge passes. Every window is locked before reading its RGB, range or world-position payload. If a new exposure has occurred outside the recorded project history, include its scene ID in the committed `results/tco_two_track_v1/hypersim_disclosed_exposures.json` array named `scenes` before either lock.
2. Convert publisher camera axes from right/up/back to right/down/forward and convert camera centres to metres. At each native pixel centre map the normalized screen coordinate through that scene’s published ray matrix. Its independent 4×4 projection matrix must send rays back to their source centres within 0.002 px. This matters for tilted cameras whose third ray component varies across the sensor. [Publisher camera conventions](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/README.md#camera-intrinsics)
3. Use a virtual pinhole camera on a 518-wide raster with the same optical axis. Derive its focal lengths and principal point from the central derivative of the full scene projection. Rectify RGB by bilinear sampling of the publisher’s tone-mapped JPEG. A native pixel with unit OpenCV ray (u) and published radial distance (d) has axial depth (z=d u_z) and camera point (d u). Convert to world metres with the known pose. The evaluation packet copies the surface corresponding to the nearest source pixel at each rectified image centre, retaining finite axial depths of 0.2–10 m. [Source range and position conventions](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/README.md#working-with-the-hypersim-dataset) · [Publisher tone mapping](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/code/python/tools/scene_generate_images_tonemap.py)
4. On the predetermined training window compare these points to the separately stored world-position image. The publisher computes radial range from those same rendered positions, so this bridge checks the loader’s geometric conventions. Report coverage, median and 95th-percentile metre errors, quantization-adjusted errors and native pixel reprojection residuals for all five frames. The publisher stores range and position as FP16. The registered limits are 2 mm and 10 mm after a per-sample quantization allowance, and 0.5 px and 2 px for native reprojection. A failed fixed bridge ends the synthetic replication. A transport or decoding error leaves the bridge unverified. [HDF5 export implementation](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/code/python/tools/generate_hdf5_from_vrimg.py)

## Workstation commands

The isolated local Python 3.9 environment under `E:/ResearchData/TCO/preparation/tco_two_track_v1/hypersim_publisher/venv` has HDF5 support through `h5py==3.11.0`. Its NumPy and Pillow dependencies come from the existing workstation installation. Metadata verification and generated tests can run before the scientific gate.

```powershell
$hypersimPython = 'E:/ResearchData/TCO/preparation/tco_two_track_v1/hypersim_publisher/venv/Scripts/python.exe'
& $hypersimPython -B scripts/prepare_tco_hypersim_replication.py metadata
& $hypersimPython -B scripts/test_tco_hypersim_replication.py
```

After a supported primary result, run these stages in order. Commit each emitted window lock before opening dense data. Preserve the exact source archive ETags and local private packets. The preparation writes only compact per-scene evidence to the repository.

```powershell
& $hypersimPython -B scripts/prepare_tco_hypersim_replication.py lock-train
git add manifests/tco_two_track_v1_hypersim_train_windows.json
git commit -m "Lock disjoint Hypersim training bridge window"
& $hypersimPython -B scripts/prepare_tco_hypersim_replication.py bridge
& $hypersimPython -B scripts/prepare_tco_hypersim_replication.py lock-test
git add manifests/tco_two_track_v1_hypersim_test_windows.json results/tco_two_track_v1/hypersim_bridge.json
git commit -m "Lock six Hypersim replication windows after bridge qualification"
& $hypersimPython -B scripts/prepare_tco_hypersim_replication.py prepare-test
& $hypersimPython -B scripts/plan_tco_two_track_comparators.py fresh-roster --dataset hypersim --inputs results/tco_two_track_v1/hypersim_fresh_inputs.json
```

Run that frozen model roster with the existing five-view worker. Use its existing CPU surface scorer for the real outputs, the two ZERO readouts and the independent rescores. Store all failures and report Hypersim separately from the ARKitScenes primary cohort. Dataset materials use the Creative Commons Attribution-ShareAlike 3.0 Unported license. Credit Roberts and colleagues, *Hypersim*, ICCV 2021. [Publisher licensing and citation](https://github.com/apple-aiml-research/ml-hypersim/blob/3463c5c4a75f3cbfc65ed31cfd6e87204b3a2254/README.md#the-hypersim-dataset)
