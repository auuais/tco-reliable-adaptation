#!/usr/bin/env bash
# Run inside the pinned PyTorch CUDA 12.8 development image on a rented 48GB GPU.
# Expected layout: /workspace/tco-research and /workspace/tco-assets.
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
cd /workspace/tco-research
mkdir -p environment reports runs/logs
export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 MAX_JOBS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST='8.6;8.9'
mode="${1:-cloud}"
test "$mode" = cloud || test "$mode" = --local-rehearsal
if [ "$mode" = --local-rehearsal ]; then
    test -z "${RUNPOD_POD_ID:-}"
    echo 'LOCAL REHEARSAL: reduced-input validation only; no full-resolution claim'
else
    python -c 'import torch; assert torch.cuda.get_device_properties(0).total_memory >= 40*2**30, "Requires >=40 GiB GPU memory"'
fi
python -c 'import torch,torchvision; assert torch.__version__ == "2.7.1+cu128"; assert torchvision.__version__ == "0.22.1+cu128"; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() in [(8,6),(8,9)], "This prebuilt bundle targets Ampere/Ada only"; print(torch.__version__, torch.cuda.get_device_name())'
# Fetch the checkpoint while installing the already-built dependencies.
python scripts/download_assets.py --root /workspace/tco-assets --checkpoint-only > environment/checkpoint-fetch.log 2>&1 &
checkpoint_pid=$!
trap 'kill "$checkpoint_pid" 2>/dev/null || true' EXIT
python scripts/install_cloud_runtime.py --runtime /workspace/tco-runtime --record environment/cloud_runtime_install.json
if [ ! -d third_party/TCO ]; then
    git clone cloud/TCO.bundle third_party/TCO
    git -C third_party/TCO remote set-url origin https://github.com/cvlab-stonybrook/TCO.git
    git -C third_party/TCO apply ../../patches/0001-bfloat16-numpy-export.patch
    mkdir -p third_party/TCO/configs/data
    cp configs/upstream_data_depth.yaml third_party/TCO/configs/data/depth.yaml
    cp configs/upstream_data_mv_recon.yaml third_party/TCO/configs/data/mv_recon.yaml
fi
test "$(git -C third_party/TCO rev-parse HEAD)" = '65387333877723cf99086f7f29089379dd50c59a'
python -m pip freeze > environment/cloud-pip-freeze.txt
python scripts/probe_renderer.py --require-precompiled --output environment/cloud_renderer_probe.json
wait "$checkpoint_pid"
trap - EXIT
python - <<'PY'
import hashlib,json,pathlib
base=pathlib.Path('/workspace/tco-assets')
for name,digest in json.loads((base/'selected_frame_sha256.json').read_text()).items():
    assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
checkpoint=base/'checkpoints/VGGT-1B/model.safetensors.download.json'
assert json.loads(checkpoint.read_text())['sha256']=='f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
digest=hashlib.sha256()
with (base/'checkpoints/VGGT-1B/model.safetensors').open('rb') as stream:
    for chunk in iter(lambda: stream.read(8*1024*1024),b''):
        digest.update(chunk)
assert digest.hexdigest()=='f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
print('Input and checkpoint checksums verified')
PY
if [ "$mode" = --local-rehearsal ]; then
    python -u scripts/run_baseline.py --run-id rehearsal_precompiled224_s1_seed42 --size 224 --frames 0 200 --steps 1 --data-root /workspace/tco-assets/data/7scenes --dataset-cache /workspace/tco-assets/data/7scenes/full_sequence_metadata.npy --checkpoint /workspace/tco-assets/checkpoints/VGGT-1B
else
    python -u scripts/run_midterm_batch.py --data-root /workspace/tco-assets --max-hours 4 --max-sequence-minutes 20
    python scripts/aggregate_midterm.py
fi
