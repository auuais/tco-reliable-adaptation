"""Local operator diagnostic composing the immutable historical numerical probe.

Two-view debugging only. Changes attention backend and disables rendering;
neither a faithful TCO reproduction nor an accuracy/performance comparison.
"""
import argparse
import contextlib
import hashlib
import json
import pathlib
import sys
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import run_numerical_probe as probe


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-policy", choices=["default", "math"], required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    run_id = remaining[remaining.index("--run-id") + 1]
    output = probe.base.ROOT / "runs" / run_id
    metadata = dict(attention_policy=args.attention_policy, effective_lambda_mv_consistency=0.0,
                    purpose="Local backward repeatability isolation; no accuracy or speed claim",
                    wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                    historical_probe_sha256=hashlib.sha256(pathlib.Path(probe.__file__).read_bytes()).hexdigest())
    os.environ["TCO_NUMERICAL_VARIANT"] = "disable_mv"
    policy = sdpa_kernel(SDPBackend.MATH) if args.attention_policy == "math" else contextlib.nullcontext()
    try:
        with policy:
            metadata["enabled_backends"] = dict(math=torch.backends.cuda.math_sdp_enabled(),
                flash=torch.backends.cuda.flash_sdp_enabled(), efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                cudnn=torch.backends.cuda.cudnn_sdp_enabled())
            probe.main()
    finally:
        if output.exists():
            (output / "attention_probe.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
