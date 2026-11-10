#!/usr/bin/env bash
# Run from a clean clone with the existing verified assets/runtime mounted.
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 MAX_JOBS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONHASHSEED=0
export TORCH_CUDA_ARCH_LIST='8.6;8.9'
cd "$(dirname "$0")/.."
mkdir -p environment reports/repeat_calibration_v6 runs/logs
exec 9>/workspace/tco-repeat-calibration-v6.lock
flock -n 9 || exit 1
test ! -e /workspace/tco_repeat_calibration_v6_results.tar.gz
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > environment/repeat_calibration_v6_exit_code.txt
    python - <<'PY'
import pathlib, tarfile
root = pathlib.Path('.')
paths = list(root.glob('runs/repeat6_*')) + list(root.glob('runs/logs/repeat6_*'))
paths += [root/'runs/diagnostic_inputs', root/'reports/repeat_calibration_v6', root/'manifests/repeat_calibration_v6.json']
paths += list(root.glob('environment/repeat_calibration_v6*'))
with tarfile.open('/workspace/tco_repeat_calibration_v6_results.tar.gz', 'x:gz', compresslevel=1) as archive:
    for path in paths:
        if path.exists(): archive.add(path, arcname=str(path))
PY
    sha256sum /workspace/tco_repeat_calibration_v6_results.tar.gz > /workspace/tco_repeat_calibration_v6_results.sha256
    echo "Session exit $result. Evidence packaged. Collect and explicitly stop provider billing."
    exit "$result"
}
trap finish EXIT
python scripts/install_cloud_runtime.py --runtime /workspace/tco-runtime --record environment/repeat_calibration_v6_runtime.json
python scripts/probe_renderer.py --require-precompiled --output environment/repeat_calibration_v6_renderer.json
python - <<'PY'
import hashlib, json, pathlib, torch
assert torch.__version__ == '2.7.1+cu128'
assert torch.cuda.get_device_capability() in [(8, 6), (8, 9)]
assert torch.cuda.get_device_properties(0).total_memory >= 24*2**30
root = pathlib.Path('/workspace/tco-assets')
for name, expected in json.loads((root/'selected_frame_sha256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
digest = hashlib.sha256()
with (root/'checkpoints/VGGT-1B/model.safetensors').open('rb') as source:
    for chunk in iter(lambda: source.read(8*1024*1024), b''): digest.update(chunk)
assert digest.hexdigest() == 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
print('Checkpoint and selected input hashes verified', flush=True)
PY
python -m unittest discover -s scripts -p 'test_*.py'
python -u scripts/run_repeat_calibration_batch.py
python scripts/aggregate_repeat_calibration.py
