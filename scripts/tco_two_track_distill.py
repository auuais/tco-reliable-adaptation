"""Gauge-invariant training targets and named Adam state for the bounded suffix pilot."""
import hashlib
import math
import re
from statistics import mean

import torch
import torch.nn.functional as F
from tco_one_percent_geometry import MIN_BASELINE, camera_centres, scene_scale


ARMS = ("Distilled", "OfflineOriginal")


def training_order(jobs):
    if len(jobs) != 6:
        raise ValueError("Six distinct native teachers are required")
    for job in jobs:
        if (job["split"] != "Training" or job.get("profile") != "clean" or job["arm"] != "Full40"
                or job["steps"] != 40 or job["seed"] != 42 or len(set(job["frames"])) != 5
                or not re.fullmatch(r"[0-9]+", str(job["visit"]))
                or not re.fullmatch(r"[A-Za-z0-9_-]+", job["slug"])
                or not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"])):
            raise ValueError("Invalid Full40 training-only teacher")
    if (len({int(job["visit"]) for job in jobs}) != 6 or
            any(len({str(job[k]) for job in jobs}) != 6 for k in ("room", "id", "slug", "input_sha256"))):
        raise ValueError("Duplicate training teacher or input")
    return sorted(jobs, key=lambda job: int(job["visit"]))


def relative_camera(poses, rig):
    """Compare predicted world-to-camera transforms in the coordinates of view zero."""
    if rig.shape not in ((5, 3, 4), (5, 4, 4)) or poses.shape not in ((5, 3, 4), (5, 4, 4)):
        raise ValueError("Expected five calibrated poses")
    if not (torch.isfinite(rig).all() and torch.isfinite(poses).all()):
        raise ValueError("Invalid camera geometry")
    scale, _ = scene_scale(rig, poses)
    if scale is None:
        raise ValueError("Unobservable teacher or student camera scale")
    baseline = torch.pdist(camera_centres(rig))
    centre = camera_centres(poses)
    unit = torch.quantile(baseline[baseline > MIN_BASELINE], 0.5)
    rotation = poses[..., :3, :3]
    orientation = rotation[1:] @ rotation[0].T
    position = (centre[1:] - centre[:1]) @ rotation[0].T * (scale / unit)
    return scale, orientation, position


def teacher_terms(pred, target, rig):
    """Return two raw scene losses. All geometry and logarithms run in FP32."""
    with torch.autocast(device_type=rig.device.type, enabled=False):
        rig = rig.detach().float()
        s, Rs, Cs = relative_camera(pred["extrinsics"][0].float(), rig)
        with torch.no_grad():
            t, Rt, Ct = relative_camera(target["extrinsics"][0].float(), rig)
            target_depth = target["depth"].detach().float()
            valid = torch.isfinite(target_depth) & (target_depth > 0)
        z = pred["depth"].float()
        if (z.shape != target_depth.shape or z.ndim != 5 or z.shape[:2] != (1,5)
                or z.shape[-1] != 1 or not valid.reshape(5,-1).any(dim=1).all()):
            raise ValueError("Invalid five-view teacher depth support")
        selected = z[valid]
        if not (torch.isfinite(selected).all() and (selected > 0).all()):
            raise ValueError("Student depth invalid on teacher support")
        depth = F.smooth_l1_loss(torch.log(selected * s), torch.log(target_depth[valid] * t))
        rotation = (Rs - Rt).square().sum(dim=(-1,-2)).mean() / 8
        translation = (Cs - Ct).square().sum(dim=-1).mean()
        camera = (rotation + translation) / 2
        if not (torch.isfinite(depth) and torch.isfinite(camera)):
            raise ValueError("Nonfinite teacher objective")
        return depth, camera, int(valid.sum().item())


def normalizers(initial_terms):
    if len(initial_terms) != 6 or any(len(row) != 2 or not all(math.isfinite(x) and x >= 0 for x in row)
                                       for row in initial_terms):
        raise ValueError("Expected six initial losses per training term")
    return tuple(max(1e-8, mean(row[i] for row in initial_terms)) for i in (0, 1))


def cpu_copy(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def factors_digest(factors):
    digest = hashlib.sha256()
    for name, value in sorted(factors.items()):
        arr = value.detach().cpu().contiguous().numpy()
        digest.update(repr((name, arr.shape, arr.dtype.str)).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


class NamedAdam:
    """Carry Adam by stable parameter names across isolated scene-model instances."""
    def __init__(self):
        self.values = {}
        self.moments = {}
        self.updates = 0
        self.initial = None

    def install(self, factors, optimizer):
        group, = optimizer.param_groups
        if (set(map(id, group["params"])) != set(map(id, factors.values()))
                or len(group["params"]) != len(factors) or group["lr"] != .0001
                or tuple(group["betas"]) != (.9, .999) or group["eps"] != 1e-8
                or group["weight_decay"] or group["amsgrad"] or optimizer.state):
            raise ValueError("Unexpected Adam settings or inventory")
        if any(value.dtype != torch.float32 for value in factors.values()):
            raise ValueError("Expected FP32 trainable suffix factors")
        native_digest = factors_digest(factors)
        if self.initial is None:
            self.initial = native_digest
        if native_digest != self.initial:
            raise ValueError("Name-seeded native initialization changed")
        if not self.updates:
            return
        if factors.keys() != self.values.keys() or factors.keys() != self.moments.keys():
            raise ValueError("Suffix inventory changed on resume")
        with torch.no_grad():
            for name, p in factors.items():
                stored = self.values[name]
                if stored.shape != p.shape:
                    raise ValueError("Suffix factor shape changed")
                p.copy_(stored)
                state = self.moments[name]
                # PyTorch Adam keeps a CPU step tensor for ordinary, uncaptured Adam.
                optimizer.state[p] = {k: v.clone() if k == "step" else v.to(p.device, copy=True)
                                      for k, v in state.items()}

    def capture(self, factors, optimizer):
        updated = self.updates + 1
        for name, p in factors.items():
            state = optimizer.state[p]
            if (set(state) != {"step", "exp_avg", "exp_avg_sq"} or int(state["step"].item()) != updated
                    or p.grad is None or not torch.isfinite(p.grad).all()
                    or not torch.isfinite(p).all() or not torch.isfinite(state["exp_avg"]).all()
                    or not torch.isfinite(state["exp_avg_sq"]).all()):
                raise ValueError("Invalid signed gradient, factor or persistent Adam moment: " + name)
            self.values[name] = cpu_copy(p)
            self.moments[name] = {k: cpu_copy(v) for k, v in state.items()}
        self.updates = updated
