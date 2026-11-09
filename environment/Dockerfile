FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9
ENV DEBIAN_FRONTEND=noninteractive \
    WANDB_MODE=disabled WANDB_DISABLED=true \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 \
    OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ninja-build git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY third_party/TCO/requirements.txt /tmp/tco-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/tco-requirements.txt
WORKDIR /workspace
