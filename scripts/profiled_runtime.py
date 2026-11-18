"""Shared deterministic arithmetic policy for the local profiled-LoRA study.

The strict replay diagnostic showed that default CUDA backward variation was
amplified by BF16. Strict PyTorch algorithms remove that variation on the pinned
image. This also selects PyTorch's deterministic bilinear forward decomposition,
so every arm, teacher and endpoint must use the same policy. No weights change.
"""
import os


EXPECTED_RUNTIME = {
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "cublas_workspace_config": ":4096:8",
}


def runtime_state():
    import torch
    return dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def configure_runtime(allow_tf32=True):
    import torch
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    required = EXPECTED_RUNTIME["cublas_workspace_config"]
    if existing is None and not torch.cuda.is_initialized():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
    elif existing != required:
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    expected = dict(EXPECTED_RUNTIME, cuda_matmul_allow_tf32=allow_tf32,
                    cudnn_allow_tf32=allow_tf32)
    state = runtime_state()
    if state != expected:
        raise RuntimeError(f"Could not establish the registered arithmetic policy: {state}")
    return state


def verify_runtime():
    state = runtime_state()
    if state != EXPECTED_RUNTIME:
        raise RuntimeError(f"Arithmetic policy changed during execution: {state}")
    return state
