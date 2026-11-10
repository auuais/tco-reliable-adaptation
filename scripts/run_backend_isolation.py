"""Follow-up to the frozen local attention probe; never edits its evidence."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import run_numerical_probe as probe


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend",choices=["default","flash","math"],required=True)
    parser.add_argument("--rendering",choices=["on","off"],required=True)
    parser.add_argument("--profile-operators",action="store_true")
    args,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest
    output=probe.base.ROOT/"runs"/rest[rest.index("--run-id")+1]
    metadata=dict(arguments=vars(args),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        historical_probe_sha256=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
        scope="Exploratory local operator diagnostic, two small views, no accuracy/speed claim",
        effective_lambda_mv_consistency=0.0 if args.rendering=="off" else None)
    os.environ["TCO_NUMERICAL_VARIANT"]="disable_mv" if args.rendering=="off" else "observe_original"
    policy=contextlib.nullcontext() if args.backend=="default" else sdpa_kernel({"flash":SDPBackend.FLASH_ATTENTION,"math":SDPBackend.MATH}[args.backend])
    profiling=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if args.profile_operators else contextlib.nullcontext()
    try:
        with policy,profiling as profiler:
            metadata["enabled_backends"]=dict(math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),
                efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled())
            probe.main()
        if args.profile_operators:
            metadata["attention_operator_counts"]={event.key:event.count for event in profiler.key_averages() if "attention" in event.key.lower()}
        if args.rendering=="on":
            metadata["effective_lambda_mv_consistency"]=json.loads((output/"run.json").read_text())["configuration"]["lambda_mv_consistency"]
    finally:
        if output.exists():(output/"backend_isolation.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":main()
