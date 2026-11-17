"""Matched surface test for sealed depth offsets and the training-only scale control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import shutil
from statistics import mean
import numpy as np

from score_partner_shape import PUBLIC as PARENT, RAW as TRAIN, ROOT, SCENES, SIDES, FRAMES, read, sha, dump

PUBLIC, RAW = PARENT/"surface", TRAIN/"surface"
ARMS = ("zero", "bias", "scale")
REFS = ("registered_rgb", "released")
DISTANCES = ("accuracy_mean_m", "completeness_mean_m")


def ratios(name):
    with np.load(PARENT/name/"fitting.npz") as data:
        out = []
        for i in range(len(FRAMES)):
            base, observed = data[f"base_{i}"], data[f"observed_{i}"]
            keep = base > 0
            if keep.sum() < 8:
                raise ValueError("Insufficient positive metric training samples")
            out.append(float(np.clip(np.median(observed[keep]/base[keep]), .95, 1.05)))
    return out


def corrected(z, row, arm):
    if arm == "zero":
        return z
    if arm == "scale":
        return z*np.array(row["scale"])[:,None,None]
    return z+np.array(row["bias_m"])[:,None,None]/row["gauge"]


def project(path, row, arm):
    from score_mast3r_transfer import resample
    from fixed_camera_depth_scale import point_map
    with np.load(path) as native:
        z = corrected(native["depth"], row, arm)
        depth, k = resample(z, native["intrinsics"].astype(float))
        poses = np.linalg.inv(native["c2w"].astype(float))[:,:3,:]
        pts = point_map(depth, k, poses, np.ones(len(depth)))
    if not np.isfinite(depth).all() or not np.isfinite(pts).all():
        raise ValueError("Nonfinite dense correction")
    return depth, pts


def predict():
    from collect_matched_focal_heads import tensor_digest
    from prepare_dense_focal_pilot import write_arrays
    (RAW/"points").mkdir(parents=True, exist_ok=True)
    (PUBLIC/"native").mkdir(parents=True, exist_ok=True)
    offsets = read(PARENT/"bias_seal.json")
    rows = {}
    for scene in SCENES:
        for side in SIDES:
            name = f"{scene}_{side}"
            trained = read(PARENT/name/"trained.json")
            native = TRAIN/name/"alignment.npz"
            if sha(native) != trained["files"]["alignment.npz"]:
                raise ValueError("Training-side native depth changed")
            dest = PUBLIC/"native"/(name+".npz")
            if not dest.exists():
                shutil.copyfile(native,dest)
            if sha(dest) != sha(native):
                raise ValueError("Public native depth copy changed")
            row = dict(native_sha256=sha(native), gauge=trained["metric_scale"], bias_m=offsets[name],
                       scale=ratios(name), arms={})
            for arm in ARMS:
                depth, pts = project(native, row, arm)
                h = tensor_digest(pts)
                write_arrays(RAW/"points"/(h+".npz"), dict(points=pts))
                row["arms"][arm] = dict(points=h, nonpositive=int((depth<=0).sum()))
            rows[name] = row
    seal = dict(parent_seal=sha(PARENT/"seal.json"), bias_seal=sha(PARENT/"bias_seal.json"),
                protocol=sha(ROOT/"docs/MAST3R_VIEW_BIAS_SURFACE.md"), cases=rows)
    if (PUBLIC/"seal.json").exists() and seal != read(PUBLIC/"seal.json"):
        raise ValueError("Dense geometry seal changed")
    dump(PUBLIC/"seal.json",seal)
    print("Sealed", sum(len(c["arms"]) for c in rows.values()), "dense predictions", flush=True)


def surface(name, hashes):
    from score_depth_scale_study import evaluate_points
    return evaluate_points(name+"_base", hashes, RAW)


def score():
    from prepare_dense_focal_pilot import REF
    from collect_matched_focal_heads import tensor_digest
    seal = read(PUBLIC/"seal.json")
    refs = read(ROOT/"manifests/dense_focal_inputs.json")["references"]
    camera_path = ROOT/"results/mast3r_geometry_matching_v1/match_audit.json"
    cams = read(camera_path)["reference_cameras"]
    observations, axial = {}, {}
    with ProcessPoolExecutor(3) as pool:
        jobs = {s:pool.submit(surface, s, {arm["points"] for d in SIDES
                   for arm in seal["cases"][f"{s}_{d}"]["arms"].values()}) for s in SCENES}
        for scene in SCENES:
            key = scene+"_base"
            ref = refs[key]
            if sha(REF/ref["file"]) != ref["sha256"]:
                raise ValueError("Changed evaluation reference")
            with np.load(REF/ref["file"]) as data:
                points = data["registered_rgb"]
                mask = data["valid_mask"].astype(bool)
                camera = np.asarray(cams[scene]["world_to_camera"])
                truth = np.einsum("nhwj,nj->nhw",points,camera[:,2,:3])+camera[:,2,3,None,None]
            if int(mask.sum()) != ref["valid_pixels"] or not np.all(truth[mask]>0):
                raise ValueError("Invalid registered axial reference")
            for side in SIDES:
                name = f"{scene}_{side}"
                row = seal["cases"][name]
                axial[name] = {}
                for arm in ARMS:
                    depth, pts = project(PUBLIC/"native"/(name+".npz"), row, arm)
                    if tensor_digest(pts) != row["arms"][arm]["points"]:
                        raise ValueError("Native-to-world-map replay differs")
                    error = np.abs(row["gauge"]*depth-truth)
                    axial[name][arm] = dict(valid_per_view=mask.sum(axis=(1,2)).tolist(),
                        mae_m_per_view=[float(np.mean(e[m])) for e,m in zip(error,mask)])
            _, metrics = jobs[scene].result()
            observations[key] = {h:{r:read(RAW/"evaluation"/key/h/r/"checked.json") for r in REFS} for h in metrics}
    dump(PUBLIC/"observations.json", observations)
    dump(PUBLIC/"axial.json", dict(camera_source_sha256=sha(camera_path), cases=axial))
    dump(PUBLIC/"scores.json", summarize())


def summarize():
    seal = read(PUBLIC/"seal.json")
    observations = read(PUBLIC/"observations.json")
    axial = read(PUBLIC/"axial.json")["cases"]
    refs = read(ROOT/"manifests/dense_focal_inputs.json")["references"]
    if seal["parent_seal"] != sha(PARENT/"seal.json") or seal["bias_seal"] != sha(PARENT/"bias_seal.json"):
        raise ValueError("Dense prediction parent changed")
    cases = {}
    for s in SCENES:
        key = s+"_base"
        for side in SIDES:
            name = f"{s}_{side}"
            row = seal["cases"][name]
            if sha(PUBLIC/"native"/(name+".npz")) != row["native_sha256"] or row["scale"] != ratios(name):
                raise ValueError("Native input or scale fit differs")
            if row["bias_m"] != read(PARENT/"bias_seal.json")[name] or row["gauge"] != read(PARENT/name/"trained.json")["metric_scale"]:
                raise ValueError("Metric correction differs")
            cases[name] = {}
            for ref in REFS:
                obs = observations[key]
                baseline = obs[row["arms"]["zero"]["points"]][ref]["metrics"]
                cells = {}
                for arm in ARMS:
                    h = row["arms"][arm]["points"]
                    record = obs[h][ref]
                    identity = dict(prediction=h, ref_archive=refs[key]["sha256"], reference=ref, valid_pixels=refs[key]["valid_pixels"])
                    if record["identity"] != identity:
                        raise ValueError("Surface association changed")
                    cell = {m:record["metrics"][m]/baseline[m] for m in DISTANCES}
                    cells[arm] = {**cell, "score":mean(cell.values())}
                cases[name][ref] = cells
    means = {d:{r:{a:{m:mean(cases[f"{s}_{d}"][r][a][m] for s in SCENES)
                    for m in (*DISTANCES,"score")} for a in ARMS} for r in REFS} for d in SIDES}
    checks = {}
    for d in SIDES:
        primary = means[d][REFS[0]]
        best = min(("zero","scale"), key=lambda a:primary[a]["score"])
        gain = 1-primary["bias"]["score"]/primary[best]["score"]
        scene = {s:1-cases[f"{s}_{d}"][REFS[0]]["bias"]["score"]/cases[f"{s}_{d}"][REFS[0]][best]["score"] for s in SCENES}
        passed = (gain>=.05 and min(scene.values())>=0 and all(primary["bias"][m]<=1.02*primary[best][m] for m in DISTANCES)
                  and means[d][REFS[1]]["bias"]["score"]<=means[d][REFS[1]][best]["score"])
        checks[d] = dict(control=best,gain=gain,scene_gains=scene,pass_all=passed)
    metric = {d:{a:mean(mean(axial[f"{s}_{d}"][a]["mae_m_per_view"]) for s in SCENES) for a in ARMS} for d in SIDES}
    return dict(seal_sha256=sha(PUBLIC/"seal.json"), observation_sha256=sha(PUBLIC/"observations.json"),
                axial_sha256=sha(PUBLIC/"axial.json"), cases=cases, means=means, axial_mae_m=metric,
                checks=checks, advance=all(v["pass_all"] for v in checks.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict","score","verify"))
    action = parser.parse_args().action
    if action == "predict":
        predict()
    elif action == "score":
        score()
    elif summarize() != read(PUBLIC/"scores.json"):
        raise ValueError("Calibrated surface replay differs")
    if action != "predict":
        print(read(PUBLIC/"scores.json")["checks"])
