"""Enable layer-selective readout only for the audited, pinned TCO camera/depth model."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/"third_party/TCO"
PINNED_MODULE="bbb4c31d8001aa5cbe5d83429ea40cbd4a396c73523de47573901660e726c5c7"


def enable(model):
    if (model.camera_head is None or model.depth_head is None or model.track_head is not None
            or model.point_head is not None):
        raise ValueError("This experiment targets the camera and depth heads with tracking and the point head disabled")
    layers=frozenset([*model.depth_head.intermediate_layer_idx,model.aggregator.depth-1])
    if not layers or min(layers)<0 or max(layers)>=model.aggregator.depth:
        raise ValueError("Invalid head feature indices")
    src=(UPSTREAM/"tco_vggt_lora.py").read_bytes().replace(b"\r\n",b"\n")
    if hashlib.sha256(src).hexdigest()!=PINNED_MODULE:
        raise ValueError("Unexpected upstream backbone module")
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"tco_vggt_lora.py"
        path.write_bytes(src)
        subprocess.run(["git","apply",str(ROOT/"patches/tco-selected-readouts.patch")],cwd=tmp,check=True)
        spec=importlib.util.spec_from_file_location("selected_readout_tco_vggt",path)
        overlay=importlib.util.module_from_spec(spec)
        # The original TCO module and its model dependencies are already imported.
        import sys
        search_path=list(sys.path)
        try:
            spec.loader.exec_module(overlay)
        finally:
            sys.path[:]=search_path
        model.tco_readout_layers=layers
        model.aggregate_with_patch_tokens=MethodType(overlay.TCO_VGGT_LoRA.aggregate_with_patch_tokens,model)
    return sorted(layers)
