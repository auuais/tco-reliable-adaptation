"""Record the hardware, model and numerical stack actually used by the first real hybrid."""
from pathlib import Path
import subprocess

from run_profiled_d0_local import docker_json
from score_tco_final_ln import digest as hash_file
from tco_final_ln_protocol import IMAGE
from tco_two_track_protocol import MANIFEST, PRIVATE, PUBLIC, ROOT, digest, read, seal


def main():
    runtime=read(MANIFEST)["runtime"]
    upstream=ROOT/"third_party/TCO"
    rev=subprocess.check_output(["git","rev-parse","HEAD"],cwd=upstream,text=True).strip()
    modifications=subprocess.check_output(["git","diff","--name-only"],cwd=upstream,text=True).splitlines()
    checkpoint=Path("E:/ResearchData/TCO/checkpoints/VGGT-1B/model.safetensors")
    weight_hash=hash_file(checkpoint)
    img,=docker_json(["image","inspect",IMAGE])
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader,nounits"],
                                text=True).strip().split(", ")
    if rev!="65387333877723cf99086f7f29089379dd50c59a" or modifications!=["utils/interfaces.py"]:
        raise ValueError("Different upstream checkout")
    if (img["Id"]!=IMAGE or gpu[0]!=runtime["gpu"] or gpu[2]!="10240" or
            weight_hash!="f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"):
        raise ValueError("Different local compute stack")
    receipt=PRIVATE/"cases/engineering_heads_real_4_Refresh1/attempt01/model/complete.json"
    run=read(receipt)
    seal(PUBLIC/"runtime.json",dict(gpu=gpu[0],driver=gpu[1],nominal_memory_mib=int(gpu[2]),
         cap_mib=run["cap_mib"],device_bytes=run["cuda_device_total"],worker_runtime=run["runtime"],
         docker_image=IMAGE,checkpoint_sha256=weight_hash,upstream_commit=rev,
         upstream_modified_files={m:digest(upstream/m) for m in modifications},
         compatibility_patch_sha256=digest(ROOT/"patches/0001-bfloat16-numpy-export.patch"),
         engineering_receipt_sha256=digest(receipt),registration_sha256=digest(MANIFEST),rental_compute_usd=0))
    print("Verified 3080, checkpoint, image and evaluator checkout")


if __name__=="__main__":
    main()
