#!/usr/bin/env bash
# Run from a clean clone with the existing verified assets/runtime mounted.
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 MAX_JOBS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST='8.6;8.9'
cd "$(dirname "$0")/.."
mkdir -p environment reports/baselines_v4 runs/logs
exec 9>/workspace/tco-baselines-v4.lock
flock -n 9 || exit 1
test ! -e /workspace/tco_baselines_v4_results.tar.gz
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > environment/baselines_v4_exit_code.txt
    python - <<'PY'
import pathlib, tarfile
root = pathlib.Path('.')
paths = list(root.glob('runs/base4_*')) + list(root.glob('runs/logs/base4_*'))
paths += [root/'runs/diagnostic_inputs', root/'reports/baselines_v4', root/'manifests/baselines_v4.json']
paths += list(root.glob('environment/baselines_v4*'))
with tarfile.open('/workspace/tco_baselines_v4_results.tar.gz', 'x:gz', compresslevel=1) as archive:
    for path in paths:
        if path.exists(): archive.add(path, arcname=str(path))
PY
    sha256sum /workspace/tco_baselines_v4_results.tar.gz > /workspace/tco_baselines_v4_results.sha256
    echo "Session exit $result. Evidence packaged. Collect and explicitly stop provider billing."
    exit "$result"
}
trap finish EXIT
python scripts/install_cloud_runtime.py --runtime /workspace/tco-runtime --record environment/baselines_v4_runtime.json
python scripts/probe_renderer.py --require-precompiled --output environment/baselines_v4_renderer.json
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
python -u scripts/run_baseline_screen.py
python scripts/aggregate_baseline_screen.py
