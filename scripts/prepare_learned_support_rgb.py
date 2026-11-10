"""Export only RGB tensors, so the matcher can run without reference geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump,sha
from fixed_match_validation import image_digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/learned_support_v8_rgb");args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for scene in ["heads/seq-01","chess/seq-03","fire/seq-03"]:
        key=scene.replace("/","_")
        record=json.loads((args.evidence_root/"runs"/f"repeat6_{key}_zero_released_seed42_r0"/"run.json").read_text())
        with np.load(args.evidence_root/record["evaluation_inputs_relative"],allow_pickle=False) as source:
            images=source["images"]
        assert images.shape==(5,3,392,518) and images.dtype==np.float32
        path=args.output/(key+".npy");np.save(path,images)
        rows.append(dict(scene=scene,file=path.name,file_sha256=sha(path),image_sha256=image_digest(images),shape=list(images.shape)))
    dump(args.output/"inputs.json",dict(rows=rows,reference_geometry_exported=False,source_sha256=sha(Path(__file__))))
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
