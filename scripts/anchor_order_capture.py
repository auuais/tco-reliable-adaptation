"""Capture the same five-view RGB batch with each image in first position."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

import collect_matched_focal_heads as original
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES
from score_saved_geometry import read

DATA = RAW/"anchor_order"
RESULT = PUBLIC/"anchor_order_capture.json"
DESIGN = ROOT/"docs/ANCHOR_ORDER_DEPTH.md"
FIELDS = ("depth","extrinsics","intrinsics")


def capture(checkpoint):
    import torch
    from selected_readouts import enable
    from stage_initial_forward import stage_initial_forward
    sys.path[:0] = [str(original.UPSTREAM),str(original.UPSTREAM/"base_models/vggt")]
    from tco_vggt_lora import TCO_VGGT_LoRA
    original.check_source()
    if original.digest_large(checkpoint/"model.safetensors")!=original.MODEL_SHA:
        raise ValueError("Unexpected weights")
    plan = read(original.INPUT_PLAN)
    old = {r["input_key"]:r for r in read(original.PUBLIC/"capture.json")["rows"]}
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    model = TCO_VGGT_LoRA.from_pretrained(str(checkpoint),enable_track=False).eval()
    model.point_head = None
    enable(model)
    stage_initial_forward(model,lambda *args:None)
    records = {}
    DATA.mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key = scene.split("/")[0]+"_base"
        spec = plan["inputs"][key]
        images = original.INPUT/spec["file"]
        old_path = original.RAW/key/"heads_and_points.npz"
        if sha(images)!=spec["sha256"] or sha(old_path)!=old[key]["archive_sha256"]:
            raise ValueError("Prior capture or RGB changed")
        with np.load(images) as f:
            rgb = f["images"]
        if original.image_digest(rgb)!=spec["rgb_sha256"]: raise ValueError("RGB tensor differs")
        for shift in range(5):
            order = np.roll(np.arange(5),-shift)
            name = f"{key}_{shift}"
            path = DATA/(name+".npz")
            meta = DATA/(name+".json")
            identity = dict(rgb_sha256=spec["rgb_sha256"],weights=original.MODEL_SHA,order=order.tolist(),
                            collector=sha(Path(__file__)))
            if meta.exists():
                row = read(meta)
                if row["identity"]!=identity or sha(path)!=row["sha256"]: raise ValueError("Partial run differs")
            else:
                model.to("cuda")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    x = torch.from_numpy(rgb[order]).unsqueeze(0).to("cuda")
                    tokens,heads = model.forward_original_vggt(x)
                    arrays = {k:heads[k][0].float().cpu().numpy()[np.argsort(order)] for k in FIELDS}
                    del tokens,heads,x
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                if not all(np.isfinite(a).all() for a in arrays.values()) or np.min(arrays["depth"])<=0:
                    raise ValueError("Invalid anchor heads")
                original.write_arrays(path,arrays)
                row = dict(identity=identity,sha256=sha(path),seconds=seconds,
                           peak_bytes=torch.cuda.max_memory_allocated())
                if shift==0:
                    with np.load(old_path) as f:
                        row["max_abs_difference"] = {k:float(np.max(np.abs(f[k]-arrays[k]))) for k in FIELDS}
                dump(meta,row)
            records[name] = row
            print(name,row.get("max_abs_difference"),flush=True)
    return records


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    args = p.parse_args()
    if RESULT.exists(): raise FileExistsError(RESULT)
    rows = capture(args.checkpoint)
    dump(RESULT,dict(protocol=hashlib.sha256(DESIGN.read_text().encode()).hexdigest(), rows=rows))
