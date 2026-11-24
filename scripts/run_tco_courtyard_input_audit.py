"""Record the decoded five-view inputs in the pinned CPU container before scoring."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from run_profiled_d0 import read, sha, write_new
from tco_courtyard_camera_protocol import DATA, IMAGE, INITIAL, INPUT_AUDIT, PREPARED, ROOT, STUDY
from tco_courtyard_camera_protocol import hashes_for, verify


def inside():
    import numpy as np
    import torch
    from run_tco_one_percent import DATA_ROOT, load_case, signature

    if torch.cuda.is_available():
        raise RuntimeError("The input audit must run without a CUDA device")
    plan = read(INITIAL)
    digest = sha(INITIAL, text=True)
    if read(PREPARED)["registration_sha256"] != digest:
        raise ValueError("Prepared data came from a different registration")
    windows = plan["planned_windows"]
    hashes = hashes_for(windows, DATA_ROOT)
    rows = []
    for job in windows:
        with tempfile.TemporaryDirectory(prefix="tco-input-audit-") as folder:
            supplied, reference = load_case(job, Path(folder), job["width"])
        arrays = {"rgb": supplied["images"].numpy(), "poses": supplied["extrs"],
                  "intrinsics": supplied["intrs"]}
        surface = {"mask": reference["valid_mask"], "surface": reference["pointclouds"]}
        valid = np.count_nonzero(surface["mask"])
        if (valid <= 0 or not np.isfinite(surface["surface"][surface["mask"]]).all()
                or any(not np.isfinite(array).all() for array in arrays.values())):
            raise ValueError(f"Invalid decoded geometry in {job['scene']}")
        row = {field: job[field] for field in ("dataset", "scene", "frames", "width")}
        rows.append(dict(**row, input_sha256=signature(arrays), reference_sha256=signature(surface),
                         raster_shape=list(arrays["rgb"].shape), valid_points=int(valid),
                         valid_points_by_view=np.count_nonzero(surface["mask"], axis=(1, 2)).tolist()))
        print(f"{job['scene']}: {int(valid)} valid fixed-mask points", flush=True)
    write_new(INPUT_AUDIT, dict(study=STUDY, registration_sha256=digest, windows=rows,
                                input_source_sha256=hashes, rental_compute_usd=0))


def host():
    from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
    from run_profiled_d1_local import host_lock

    verify("initial")
    if INPUT_AUDIT.exists():
        raise ValueError("A decoded input receipt exists. Preserve it and verify its registration digest.")
    if not PREPARED.is_file():
        raise FileNotFoundError("Run the registered cached ETH3D input preparation first")
    name = "tco-courtyard-input-audit"
    command = DOCKER + ["create", "--name", name, "--pull", "never", "--network", "none",
                        "--label", f"tco.reconstruction.study={STUDY}", "--workdir", "/workspace",
                        "--env", "CUDA_VISIBLE_DEVICES=", "--env", "MKL_NUM_THREADS=4",
                        "--env", "OPENBLAS_NUM_THREADS=4", "--env", "OMP_NUM_THREADS=4",
                        "--mount", f"type=bind,source={ROOT.as_posix()},target=/workspace",
                        "--mount", f"type=bind,source={DATA.as_posix()},target=/data,readonly",
                        IMAGE, "python", "-u", "scripts/run_tco_courtyard_input_audit.py", "--inside"]
    with host_lock():
        identifier = subprocess.check_output(command, text=True, timeout=45).strip()
        try:
            actual, = docker_json(["inspect", identifier])
            if (actual["Image"] != IMAGE or actual["HostConfig"]["NetworkMode"] != "none"
                    or actual["HostConfig"].get("DeviceRequests")
                    or not any(m["Destination"] == "/data" and m["RW"] is False for m in actual["Mounts"])):
                raise RuntimeError("The input audit container differs from the registered environment")
            subprocess.run(DOCKER + ["start", "--attach", identifier], check=True, timeout=1200)
        finally:
            final, errors = ensure_stopped(identifier)
            if final is None or final["State"]["Running"]:
                raise RuntimeError(f"Unable to verify that the input audit stopped: {errors}")
            # Only the temporary container is removed. All inputs and evidence are bind-mounted.
            subprocess.run(DOCKER + ["rm", identifier], check=True, timeout=30)
    if not INPUT_AUDIT.is_file():
        raise RuntimeError("The offline input audit produced no receipt")
    print(f"Verified raw and decoded input signatures in {INPUT_AUDIT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    inside() if args.inside else host()


if __name__ == "__main__":
    main()
