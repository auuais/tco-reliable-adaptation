"""Release CPU-replayable engineering evidence from all registered model controls."""
import argparse
import math

import numpy as np

from qualify_tco_final_ln_bridge import bridge, RELATIVE, COSINE, ROBUST_FRACTION
from score_tco_final_ln import check_optimizer
from tco_final_ln_protocol import IMAGE, MANIFEST, PUBLIC, RAW, ROOT, read, seal, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finished(job):
    folders = sorted((RAW/"cases"/job["id"]).glob("attempt[0-9][0-9]"))
    valid = []
    attempts = []
    for folder in folders:
        term = folder/"terminal.json"
        if not term.is_file():
            raise ValueError("Attempt is still open: " + str(folder))
        record = read(term)
        attempts.append(dict(attempt=record["attempt"], status=record["status"], terminal_sha256=sha(term)))
        if record["status"]=="completed":
            model = folder/"model"
            require(record["stopped"] and record["rental_compute_usd"]==0, "Missing local termination receipt")
            require(sha(model/"complete.json")==record["complete_sha256"], "Changed completion")
            valid.append(model)
    require(len(valid)==1,"Expected exactly one valid result for "+job["id"])
    return valid[0],attempts


def fixed_evidence(jobs):
    folders = {job["id"]:finished(job)[0] for job in jobs if job["profile"] in ("fixed","fixed_bridge")}
    reference = folders["engineering_bridge_fixed"]
    pairs = [bridge(reference,folders[case]) for case in ("engineering_standard_fixed","engineering_zerolr_fixed")]
    return dict(passed=all(row["passed"] for row in pairs), comparisons=pairs,
                relative_l2_limit=RELATIVE, minimum_cosine=COSINE, robust_relative_to_largest=ROBUST_FRACTION)


def signed_check(optimizer, steps):
    ln = optimizer["ln_names"]
    for grads in optimizer["signed_ln_gradients"]:
        require(set(grads)==set(ln), "Mismatched signed gradient names")
        for item in grads.values():
            require(math.isfinite(item["norm"]) and item["norm"]>0 and
                    all(math.isfinite(x) for x in item["first"]+item["last"]), "Missing finite real LN gradient")
    for role in ("lora","ln"):
        expected = steps if role=="lora" or ln else 0
        for key in ("preclip_norms", "postclip_norms"):
            vals = optimizer[key][role]
            require(len(vals)==expected and all(math.isfinite(v) and v>=0 for v in vals),"Invalid "+key)
        require(all(x<=1.00002 for x in optimizer["postclip_norms"][role]), "Failed independent clipping")


def actual_comparison(ref_path, path):
    with np.load(ref_path/"prediction.npz",allow_pickle=False) as ref, np.load(path/"prediction.npz",allow_pickle=False) as alt:
        out={}
        for key in ("canonical","depth","depth_conf","extrinsics","intrinsics"):
            a,b = ref[key], alt[key]
            require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite native bridge")
            delta = a.astype(np.float64)-b
            out[key] = dict(relative_l2=float(np.sqrt(np.sum(delta**2)/np.sum(a.astype(np.float64)**2))),
                            root_mean_squared_difference=float(np.sqrt(np.mean(delta**2))), exact=bool(np.array_equal(a,b)))
    return out


def release():
    manifest=read(MANIFEST)
    cpu=read(RAW/"cpu_qualification.json")
    require(cpu["passed"] and cpu["lr_zero_layernorm_unchanged"] and cpu["lr_positive_all_changed"]
            and cpu["adam_independent_reference"] and cpu["zero_lr_keeps_lora_update"]
            and cpu["optimizer_states"]==388 and cpu["one_channel_readout_max_change"]>.001
            and len(cpu["directions"])==4 and all(r["sign_flip_rejected"] for r in cpu["directions"].values())
            and cpu["rgb_ed_cpu"]["exact_signed_means_and_opacity"],"CPU qualification failed")
    fixed=fixed_evidence(manifest["engineering"])
    require(fixed["passed"],"Deterministic bridge failed fixed numerical criteria")
    baseline_hash=None
    rows,folders,raw={}, {}, {}
    for job in manifest["engineering"]:
        folder,attempts=finished(job)
        row=read(folder/"complete.json")
        require(all(row.get(k)==v for k,v in job.items()), "Worker and registration disagree")
        pins=manifest["pins"]
        require(row["input_sha256"]==job["input_sha256"] and sha(folder/"prediction.npz")==row["predictions_sha256"]
                and row["runtime"]["torch"]==pins["torch"] and row["runtime"]["gsplat_sha256"]==pins["gsplat_sha256"]
                and row["runtime"]["gpu"]==pins["gpu"] and row["runtime"]["allocator"]==pins["allocator"]
                and row["cap_mib"]==8960 and 0<row["peak_allocated"]<=8960*2**20
                and row["rental_compute_usd"]==0, "Input, prediction or runtime mismatch")
        require(row["native_depth_batches"]==[[5]] and row["readout"]["dtype"]=="float32"
                and row["readout"]["shape"]==[5,*row["raster_shape"][-2:],3]
                and row["render_calls"]==["RGB+ED"]*job["steps"] and row["camera_checkpoint"]
                and row["head_taps"]==[4,11,17,23], "Wrong registered model, calls, or FP32 readout")
        check_optimizer(row)
        opt=row["optimizer"]
        signed_check(opt,job["steps"])
        if baseline_hash is None:
            baseline_hash=opt["ln_initial_sha256"]
        require(baseline_hash==opt["ln_initial_sha256"], "Fresh jobs have different original LN weights")
        require((opt["ln_final_sha256"]!=baseline_hash)==(job["ln_lr"]>0), "Missing or impermissible movement")
        require(opt["frozen_version_count"]>384 and len(opt["steps"])==2, "Missing frozen base version checks")
        rows[job["id"]]=dict(complete_sha256=sha(folder/"complete.json"),attempts=attempts,
            file_sha256=row["predictions_sha256"], input_sha256=row["input_sha256"], initial_lora_sha256=opt["lora_initial_sha256"],
            lr=job["ln_lr"], selected_layernorm=opt["ln_names"], step_count=len(opt["steps"]),
            adam_states=len(opt["named_counts"]), frozen_versions=opt["frozen_version_count"], losses=row["losses"],
            layernorm_displacements=opt["ln_delta_norms"], signed_layernorm=opt["signed_ln_gradients"],
            preclip_norms=opt["preclip_norms"], postclip_norms=opt["postclip_norms"], scene_seconds=row["scene_seconds"],
            peak_allocated_mib=row["peak_allocated"]/2**20,peak_reserved_mib=row["peak_reserved"]/2**20)
        folders[job["id"]],raw[job["id"]]=folder,row
    production=[k for k in rows if k.endswith("_native")]
    ref="engineering_standard_native"
    fields=("view_permutations","input_sha256","seed","native_depth_batches","render_calls")
    for k in production:
        require(all(raw[k][f]==raw[ref][f] for f in fields)
                and raw[k]["optimizer"]["lora_initial_sha256"]==raw[ref]["optimizer"]["lora_initial_sha256"],
                "Production controls use different starting conditions")
    comparisons={k:actual_comparison(folders[ref],folders[k]) for k in production if k!=ref}
    # Production repeat spread is descriptive; fixed signed derivatives provide the tolerance-based bridge.
    compact=dict(passed=True, study=manifest["study"], registration_sha256=sha(MANIFEST), image=IMAGE,
                 cpu=cpu,cpu_sha256=sha(RAW/"cpu_qualification.json"), fixed=fixed, cases=rows,
                 production_reference=ref, production_prediction_differences=comparisons,
                 reference_role_isolation="Model inputs carry only RGB, W2C camera matrices and intrinsics in all eight cases.",
                 initial_pretrained_ln_sha256=baseline_hash,rental_compute_usd=0)
    seal(PUBLIC/"qualification.json",compact)
    seal(RAW/"qualification_pass.json",dict(passed=True,engineering_cases=len(rows),rental_compute_usd=0,
                                           public_result_sha256=sha(PUBLIC/"qualification.json")))
    return compact


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed-only",action="store_true")
    args=p.parse_args()
    if args.fixed_only:
        result=fixed_evidence(read(MANIFEST)["engineering"])
        seal(RAW/"fixed_bridge.json",result)
        print(result["passed"], [r["fraction_bitwise_equal"] for r in result["comparisons"]])
        if not result["passed"]:
            raise SystemExit(1)
    else:
        out=release()
        print("Qualification passed:",len(out["cases"]),"model controls")
