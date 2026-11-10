"""Omit a single external camera from the pinned TCO loss while retaining every image."""
from functools import update_wrapper
from types import FunctionType


def without_camera(loss, index):
    """Build a worker-local copy of the upstream loss with only camera constraints sliced."""
    if index is None:
        return loss
    if type(index) is not int or index < 0:
        raise ValueError("A nonnegative camera index is required")
    namespace = dict(loss.__globals__)
    pose = namespace["direct_pose_energy"]
    intrinsics = namespace["direct_intrinsics_energy"]
    rays = namespace["intrinsics_ray_consistency_energy_v2"]

    def retain(predicted, supplied):
        if predicted.shape[:2] != supplied.shape[:2] or predicted.ndim < 4 or supplied.ndim != 4:
            raise ValueError("Expected matching batched camera/view dimensions")
        count = predicted.shape[1]
        if index >= count:
            raise IndexError(index)
        views = [i for i in range(count) if i != index]
        return predicted[:,views], supplied[:,views]

    def selected_pose(predicted_poses, gt_extrinsics, **options):
        p, g = retain(predicted_poses, gt_extrinsics)
        if p.shape[1] < 2 and options["canonicalize_to_first_view"]:
            return p.sum()*0, p.sum()*0
        if p.shape[1] == 0:
            return p.sum()*0, p.sum()*0
        if index == 0 and options["canonicalize_to_first_view"]:
            # Both original world frames must now be expressed in the same surviving camera.
            options["pose_type"] = "rel"
        return pose(predicted_poses=p, gt_extrinsics=g, **options)

    def selected_intrinsics(pred, gt):
        p, g = retain(pred, gt)
        return intrinsics(p, g) if p.shape[1] else p.sum()*0

    def selected_rays(pred, gt):
        p, g = retain(pred, gt)
        return rays(p, g) if p.shape[1] else p.sum()*0

    namespace.update(direct_pose_energy=selected_pose, direct_intrinsics_energy=selected_intrinsics,
                     intrinsics_ray_consistency_energy_v2=selected_rays)
    scoped = FunctionType(loss.__code__, namespace, loss.__name__, loss.__defaults__, loss.__closure__)
    scoped.__kwdefaults__ = loss.__kwdefaults__
    return update_wrapper(scoped, loss)
