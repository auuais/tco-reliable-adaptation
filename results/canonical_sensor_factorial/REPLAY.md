# Replaying the full-model calibration experiment

The experiment crosses two choices in the external camera prior supplied to TCO. The first choice is whether to transform the released camera poses into the RGB sensor frame. The second is whether to use the registered RGB intrinsics. Each of the three 7-Scenes windows runs all four combinations. Every run starts from the same pretrained weights and seed 42 and takes 40 optimizer steps. The five frame IDs per window are 50, 250, 450, 650 and 850. The released and registered dense references enter only during evaluation. [Fixed job list and checksums](../../manifests/canonical_sensor_factorial.json)

### Prepare the inputs

Use a Linux checkout of execution revision `c0b801be8dcd4e60d43d151e19bd510a5273ecea` and the pinned PyTorch, CUDA and precompiled gsplat environment. Install upstream TCO at the recorded commit in `third_party/TCO` and apply `patches/0001-bfloat16-numpy-export.patch` in that checkout. Copy `configs/upstream_data_depth.yaml` and `configs/upstream_data_mv_recon.yaml` into `third_party/TCO/configs/data/` as `depth.yaml` and `mv_recon.yaml`. Keep generated files under the repository's ignored directories. [Environment](../../environment/Dockerfile) · [Pinned dependencies](../../environment/requirements-hashed.txt) · [Reference commits](../../docs/REPRODUCTION.md)

Download the public inputs with `python scripts/download_assets.py --root /data --scenes heads chess fire`. Extract the scene and sequence archives under `/data/data/7scenes` and register the depth maps into the RGB image plane following the linked preprocessing reference. The dry run verifies the bytes of all 45 chosen RGB, pose and registered-depth files. It also verifies the checkpoint. [Depth registration and sources](../../docs/THIRD_PARTY.md)

The upstream dataset loader uses a small sequence-length index. From the repository root, use NumPy 2.1.2 to generate the frozen index. These five sequence lengths describe the original complete sequences. The job manifest selects three of them for this experiment.

```python
from pathlib import Path
import hashlib
import io
import numpy as np

sequences = ["heads/seq-01", "chess/seq-03", "chess/seq-05", "fire/seq-03", "fire/seq-04"]
cache = io.BytesIO()
np.save(cache, dict.fromkeys(sequences, 1000))
data = cache.getvalue()
assert hashlib.sha256(data).hexdigest() == "ce8f28f03d960fc4fc8be5624823715822a41550bddd421af9ca1942295afb8a"
path = Path("environment/development_dataset_cache_v7.npy")
path.parent.mkdir(exist_ok=True)
if path.exists():
    assert path.read_bytes() == data
else:
    path.write_bytes(data)
```

### Execute and score

Run the checks on the intended execution environment, then execute the 12 jobs on one GPU. Preserve `runs/sensor17c_*`, their referenced `runs/diagnostic_inputs` directories, and `reports/canonical_sensor_factorial`. Copy these directories to the scoring machine with their paths intact. Store all original attempts, including any interrupted runs.

```sh
python scripts/run_canonical_sensor_factorial.py dry-run --assets /data
python scripts/run_canonical_sensor_factorial.py run --assets /data
```

Run the following commands on CPU in the same software environment. Use a fresh checkout or worktree for an independent replay so the new prediction seal has its own location. The scoring program checks the new seal against the exact file in the replay repository's `origin/main`. Commit and push that file to your own replay repository between the first and second commands. The original execution revision predates the published seal.

```sh
python scripts/evaluate_canonical_sensor_factorial.py prepare
# Publish results/canonical_sensor_factorial/seal.json in the replay repository.
python scripts/evaluate_canonical_sensor_factorial.py score
python scripts/diagnose_sensor_components.py fetch --study sensor_tco_v17
python scripts/evaluate_canonical_sensor_factorial.py components
python scripts/evaluate_canonical_sensor_factorial.py collect
python scripts/evaluate_canonical_sensor_factorial.py verify
```

The primary comparison uses reconstruction accuracy and completeness on both dense references. The separate component table reports axial depth and camera errors relative to published RGB structure-from-motion poses. Both dense references share measured depth. The independent pose estimates provide a different check on cameras. [Evaluation source](../../scripts/evaluate_sensor_tco.py) · [Component metrics](../../scripts/diagnose_sensor_components.py)
