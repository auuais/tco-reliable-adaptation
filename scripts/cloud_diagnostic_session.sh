#!/usr/bin/env bash
# One bounded diagnostic batch. No original run or original session log is overwritten.
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 MAX_JOBS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST='8.6;8.9'
cd /workspace/tco-research
mkdir -p environment reports/diagnostics_v2 runs/logs
exec 9>/workspace/tco-diagnostics-v2.lock
flock -n 9 || exit 1
test ! -e /workspace/tco_diagnostics_v2_results.tar.gz
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > environment/diagnostics_v2_exit_code.txt
    python - <<'PY'
import pathlib,tarfile
root=pathlib.Path('.')
paths=list(root.glob('runs/diag2_*'))+list(root.glob('runs/logs/diag2_*'))
paths += [root/'runs/diagnostic_inputs',root/'reports/diagnostics_v2',root/'manifests/diagnostics_v2.json']
paths += list(root.glob('environment/diagnostics_v2*'))
paths += [root/'scripts'/name for name in ['prior_corruptions.py','run_diagnostic.py','run_diagnostic_batch.py','aggregate_diagnostics.py','cloud_diagnostic_session.sh','test_prior_corruptions.py']]
paths += [root/'configs/diagnostic_profiles.json']
with tarfile.open('/workspace/tco_diagnostics_v2_results.tar.gz','w:gz') as archive:
    for path in paths:
        if path.exists(): archive.add(path,arcname=str(path))
PY
    sha256sum /workspace/tco_diagnostics_v2_results.tar.gz > /workspace/tco_diagnostics_v2_results.sha256
    echo "Diagnostic session finished: exit $result. Collect evidence and stop provider billing."
    exit "$result"
}
trap finish EXIT
python -c 'import torch; assert torch.__version__=="2.7.1+cu128"; assert torch.cuda.get_device_capability() in [(8,6),(8,9)]; assert torch.cuda.get_device_properties(0).total_memory>=40*2**30'
python scripts/install_cloud_runtime.py --runtime /workspace/tco-runtime --record environment/diagnostics_v2_runtime.json
python scripts/probe_renderer.py --require-precompiled --output environment/diagnostics_v2_renderer.json
python scripts/test_prior_corruptions.py
python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/workspace/tco-assets')
for name,expected in json.loads((root/'selected_frame_sha256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
digest=hashlib.sha256()
with (root/'checkpoints/VGGT-1B/model.safetensors').open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
assert digest.hexdigest()=='f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
print('Checkpoint and all selected input hashes passed')
PY
python -u scripts/run_diagnostic_batch.py
python scripts/aggregate_diagnostics.py
