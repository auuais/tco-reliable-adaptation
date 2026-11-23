"""Build a sparse, fixed correspondence graph for the scene's boundary adapter."""

import math

import torch
from torch import nn
from torch.nn import functional as F


RANK = 4
MIN_BASELINE = 1e-6
RELATIVE_DEPTH_TOLERANCE = 0.05


def camera_centres(extrinsics):
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3:]
    return -(rotation.transpose(-1, -2) @ translation).squeeze(-1)


def scene_scale(prior_extrinsics, predicted_extrinsics):
    expected = torch.pdist(camera_centres(prior_extrinsics))
    estimated = torch.pdist(camera_centres(predicted_extrinsics))
    ratios = expected / estimated
    valid = ((expected > MIN_BASELINE) & (estimated > MIN_BASELINE)
             & torch.isfinite(expected) & torch.isfinite(estimated) & torch.isfinite(ratios))
    ratios = ratios[valid].sort().values
    if len(ratios) < 2:
        return None, {"pairs": len(ratios), "scale": None, "scale_range": None}
    mid = len(ratios) // 2
    scale = ratios[mid] if len(ratios) % 2 else ratios[mid - 1:mid + 1].mean()
    return scale, {"pairs": len(ratios), "scale": float(scale),
                   "scale_range": [float(ratios[0]), float(ratios[-1])]}


def sample_depth(depth, xy):
    """Gather bilinear axial depth, requiring all contributing pixels to be valid."""
    height, width = depth.shape
    x, y = xy.unbind(-1)
    supported = (torch.isfinite(x) & torch.isfinite(y) & (x >= 0) & (y >= 0)
                 & (x <= width - 1) & (y <= height - 1))
    x = torch.where(supported, x, 0)
    y = torch.where(supported, y, 0)
    left, top = x.floor().long(), y.floor().long()
    right = (left + 1).clamp(max=width - 1)
    bottom = (top + 1).clamp(max=height - 1)
    dx, dy = x - left, y - top
    weights = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy))
    samples = torch.stack((depth[top, left], depth[top, right], depth[bottom, left], depth[bottom, right]))
    valid = torch.isfinite(samples) & (samples > 0)
    values = (samples.masked_fill(~valid, 0) * weights).sum(0)
    return values, supported & (valid | (weights == 0)).all(0)


def project(points, intrinsics):
    pixels = points @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:]


def lift(xy, depth, inverse_intrinsics):
    pixels = F.pad(xy, (0, 1), value=1)
    return (pixels @ inverse_intrinsics.T) * depth[:, None]


@torch.no_grad()
def correspondences(depth, prior_extrinsics, predicted_extrinsics, intrinsics, stride):
    """Return donor IDs per [receiving view, patch, donating view], using -1 for no edge."""
    with torch.autocast(depth.device.type, enabled=False):
        return _correspondences(depth.float(), prior_extrinsics.float(), predicted_extrinsics.float(),
                                intrinsics.float(), stride)


def _correspondences(depth, prior, predicted, intrinsics, stride):
    views, height, width = depth.shape
    rows, cols = height // stride, width // stride
    offset = (stride - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(rows, device=depth.device) * stride + offset,
                            torch.arange(cols, device=depth.device) * stride + offset, indexing="ij")
    centres = torch.stack((xx.flatten(), yy.flatten()), -1)
    patches = len(centres)
    edges = torch.full((views, patches, views), -1, dtype=torch.long, device=depth.device)
    scale, record = scene_scale(prior, predicted)
    if scale is not None:
        depth = depth * scale
        rotation, translation = prior[:, :3, :3], prior[:, :3, 3]
        inverse = torch.linalg.inv(intrinsics)
        for source in range(views):
            source_depth, source_valid = sample_depth(depth[source], centres)
            source_points = lift(centres, source_depth, inverse[source])
            for target in range(views):
                if source == target:
                    continue
                relative_rotation = rotation[target] @ rotation[source].T
                relative_translation = translation[target] - relative_rotation @ translation[source]
                target_points = source_points @ relative_rotation.T + relative_translation
                xy = project(target_points, intrinsics[target])
                target_depth, target_valid = sample_depth(depth[target], xy)
                observed = lift(xy, target_depth, inverse[target])
                returned = (observed - relative_translation) @ relative_rotation
                reverse_error = (project(returned, intrinsics[source]) - centres).norm(dim=-1)
                valid = (source_valid & target_valid & (target_points[:, 2] > 0) & (returned[:, 2] > 0)
                         & ((target_points[:, 2] - target_depth).abs()
                            <= RELATIVE_DEPTH_TOLERANCE * target_depth)
                         & (reverse_error <= stride / 2))
                safe_xy = torch.where(valid[:, None], xy, 0)
                # A projected point halfway between centres rounds towards the larger index.
                row = ((safe_xy[:, 1] - offset) / stride + 0.5).floor().long().clamp(0, rows - 1)
                col = ((safe_xy[:, 0] - offset) / stride + 0.5).floor().long().clamp(0, cols - 1)
                edges[source, :, target] = torch.where(valid, target * patches + row * cols + col, -1)
    partners = (edges >= 0).sum(-1)
    record.update(fallback=("undefined_scale" if scale is None else
                            "empty_graph" if not partners.any().item() else None),
                  patches_per_view=patches, supported_patches=(partners > 0).sum(-1).tolist(),
                  partner_histogram=torch.bincount(partners.flatten(), minlength=views).tolist())
    return edges, record


def shuffle_edges(edges, seed):
    """Keep support and each ordered pair's donor multiset fixed."""
    generator = torch.Generator().manual_seed(seed)
    original = edges.cpu()
    shuffled = original.clone()
    for source in range(edges.shape[0]):
        for target in range(edges.shape[2]):
            ids = original[source, :, target]
            supported = ids >= 0
            donors = ids[supported]
            if len(donors) > 1:
                shift = int(torch.randint(1, len(donors), (), generator=generator))
                shuffled[source, supported, target] = donors.roll(shift)
    changed = shuffled != original
    return shuffled.to(edges.device), {"changed_edges": int(changed.sum()),
                                      "changed_receivers": int(changed.any(-1).sum()),
                                      "valid_edges": int((original >= 0).sum())}


class BoundaryCorrection(nn.Module):
    """LoRA factors for a residual on patch tokens only."""

    def __init__(self, channels, edges, mode, seed):
        super().__init__()
        self.mode = mode
        device = edges.device
        self.register_buffer("valid", edges >= 0)
        self.register_buffer("donor", edges.clamp_min(0))
        self.register_buffer("count", self.valid.sum(-1))
        self.lora_A = nn.Parameter(torch.empty(RANK, channels, device=device))
        self.lora_B = nn.Parameter(torch.zeros(channels, RANK, device=device))
        generator = torch.Generator(device=device).manual_seed(seed)
        limit = 1 / math.sqrt(channels)
        with torch.no_grad():
            self.lora_A.uniform_(-limit, limit, generator=generator)

    def forward(self, tokens, special):
        views, patches = self.count.shape
        channels = tokens.shape[-1]
        grid = tokens.reshape(views, special + patches, channels)
        features = grid[:, special:]
        projected = F.linear(features, self.lora_A)
        if self.mode != "own":
            borrowed = projected.reshape(-1, RANK)[self.donor]
            averaged = borrowed.masked_fill(~self.valid[..., None], 0).sum(-2)
            projected = averaged / self.count.clamp_min(1)[..., None] - projected
        delta = F.linear(projected, self.lora_B)
        delta = delta.masked_fill(self.count[..., None] == 0, 0)
        corrected = torch.cat((grid[:, :special], features + delta.to(features.dtype)), dim=1)
        return corrected.reshape_as(tokens)
