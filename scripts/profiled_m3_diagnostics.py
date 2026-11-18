"""Pure fixed-mask boundary and metric axial-depth diagnostics for M3.

No files, models, fitting or authorization live here. The caller must verify
the sealed predictions, reference archive, common mask, camera provenance and
kernel result before supplying arrays. Surface diagnostics consume the exact
kernel-aligned points; the supplied scene transform is checked, never refitted.
"""
import hashlib
import json

import numpy as np
from scipy.ndimage import distance_transform_edt

from prepare_profiled_m3_references import tensor_sha as preparation_camera_sha256


REGIONS = ("missing_neighbor", "jump_neighbor", "interior")
TRANSFORM_FIELDS = ("similarity_scale", "similarity_rotation",
                    "similarity_translation", "icp_transformation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(mask,radius):
    return distance_transform_edt(~mask)<=radius if mask.any() else np.zeros_like(mask)


def partition(points,valid):
    jump = np.zeros_like(valid)
    for axis in (0,1):
        left = (slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
        right = (slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        gap = valid[left] & valid[right] & (np.linalg.norm(points[left]-points[right],axis=-1)>.05)
        jump[left] |= gap
        jump[right] |= gap
    missing = valid & near(~valid,2)
    edge = valid & ~missing & near(jump,2)
    return (missing,edge,valid & ~missing & ~edge)


def tensor_sha256(array):
    """Same dtype/shape-aware mask convention as profiled_m3_points."""
    array = np.asarray(array)
    descriptor = json.dumps([array.dtype.str,list(array.shape)],separators=(",",":")).encode()
    return hashlib.sha256(descriptor+b"\0"+array.tobytes(order="C")).hexdigest()


def floating(value, name, shape):
    array = np.asarray(value)
    require(array.dtype in (np.dtype("float32"),np.dtype("float64")), name+" must use float32 or float64")
    require(array.shape==shape, name+" shape differs")
    return array


def finite(value, name):
    require(np.isfinite(value).all(), name+" contains nonfinite values")


def transform_check(prediction, reference, mask, aligned, alignment):
    """Check supplied kernel output against its one scene transform, without fitting."""
    require(isinstance(alignment,dict) and all(name in alignment for name in TRANSFORM_FIELDS),
            "The kernel's complete scene transform is required")
    scale = np.asarray(alignment["similarity_scale"])
    require(scale.shape==() and np.issubdtype(scale.dtype,np.number) and
            np.isfinite(scale).all() and float(scale)>0,"Invalid similarity scale")
    rotation = np.asarray(alignment["similarity_rotation"],dtype=np.float64)
    translation = np.asarray(alignment["similarity_translation"],dtype=np.float64)
    icp = np.asarray(alignment["icp_transformation"],dtype=np.float64)
    require(rotation.shape==(3,3) and translation.shape==(3,1) and icp.shape==(4,4),
            "Scene transform shapes differ from kernel convention")
    for name,value in (("similarity rotation",rotation),("similarity translation",translation),("ICP",icp)):
        finite(value,name)
    # Legacy float32 SVD can produce a float64 R after multiplication by eye(3),
    # retaining float32 rotation error. This is an integrity check, not a repair.
    rotation_tol = 128.*max(np.finfo(prediction.dtype).eps,np.finfo(reference.dtype).eps)
    for name,value in (("similarity",rotation),("ICP",icp[:3,:3])):
        require(np.linalg.det(value)>0 and
                np.allclose(value.T@value,np.eye(3),rtol=0,atol=rotation_tol),
                name+" rotation is incompatible with the declared source precision")
    require(np.array_equal(icp[3],[0.,0.,0.,1.]),"ICP must have an affine homogeneous row")
    # Preserve the original einsum Sim3 arithmetic. Open3D's final matrix
    # multiplication can differ by a few float64 operations' rounding.
    with np.errstate(over="ignore",invalid="ignore"):
        pre_icp = float(scale)*np.einsum("nhwj,ij->nhwi",prediction,rotation)+translation.T
        expected = pre_icp[mask]@icp[:3,:3].T+icp[:3,3]
    finite(expected,"Recomposed scene transform")
    rounding = 512.*np.finfo(np.float64).eps
    magnitude = max(1.,float(np.max(np.abs(pre_icp[mask]))),
                    float(np.max(np.abs(expected))),float(np.max(np.abs(icp[:3,3]))))
    atol = rounding*magnitude
    require(np.allclose(aligned,expected,rtol=rounding,atol=atol),
            "Aligned rows do not match the one scene transform and fixed mask order")
    return dict(transform_refitted=False,source_rotation_tolerance=float(rotation_tol),
                recomposition_rtol=float(rounding),recomposition_atol_m=float(atol),
                max_recomposition_difference_m=float(np.max(np.abs(aligned-expected))),
                metric_points="supplied exact kernel-aligned points; recomposition is an integrity check only")


def reference_camera_check(cameras, views):
    camera = floating(cameras,"reference world-to-camera",(views,4,4))
    finite(camera,"reference world-to-camera")
    tolerance = 64.*np.finfo(camera.dtype).eps
    require(np.allclose(camera[:,3,:],[0.,0.,0.,1.],rtol=0,atol=tolerance),
            "Reference camera homogeneous row is invalid")
    rotation = camera[:,:3,:3]
    determinant = np.linalg.det(rotation)
    require(np.all(determinant>0),"Reference cameras must be right-handed and invertible")
    try:
        inverse = np.linalg.inv(camera)
    except np.linalg.LinAlgError as error:
        raise ValueError("Reference cameras must be invertible") from error
    finite(inverse,"Reference camera inverse")
    # The archived reference convention uses approximately rigid source poses.
    # Never replace its true inverse with an ideal-rotation transpose here.
    defect = float(np.max(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3))))
    return camera,dict(camera_tensor_sha256=tensor_sha256(camera),
                       camera_preparation_sha256=preparation_camera_sha256(camera),
                       camera_hash_algorithms={
                           "camera_tensor_sha256":dict(algorithm="SHA256",
                               schema="profiled_dtype_shape_compact_json_nul_c_bytes_v1",
                               encoding="UTF8 json.dumps([dtype.str,list(shape)],separators=(',',':')) + NUL byte + C-order array bytes"),
                           "camera_preparation_sha256":dict(algorithm="SHA256",
                               schema="profiled_shape_dtype_default_json_c_bytes_v1",
                               encoding="UTF8 json.dumps([list(shape),dtype.str]) with default separators + contiguous C-order array bytes")},
                       max_rotation_orthogonality_error=defect,
                       min_rotation_determinant=float(determinant.min()),
                       matrices_modified=False,
                       provenance_verified=False,
                       provenance_requirement="caller verifies registered_rgb reference_poses/effective_camera archive linkage")


def error_stats(values, signed=False):
    values = np.asarray(values,dtype=np.float64)
    finite(values,"Diagnostic errors")
    count = int(values.size)
    absolute = np.abs(values)
    with np.errstate(over="ignore",invalid="ignore"):
        squared = values*values
    finite(squared,"Squared diagnostic errors")
    total_abs = float(absolute.sum())
    total_sq = float(squared.sum())
    total_signed = float(values.sum()) if signed else None
    require(np.isfinite(total_abs) and np.isfinite(total_sq) and
            (total_signed is None or np.isfinite(total_signed)),"Diagnostic accumulation overflowed")
    result = dict(status="ok" if count else "empty",n=count,
                  absolute_sum_m=total_abs,squared_sum_m2=total_sq,
                  mean_absolute_m=total_abs/count if count else None,
                  rmse_m=float(np.sqrt(total_sq/count)) if count else None,
                  max_absolute_m=float(absolute.max()) if count else None)
    if signed:
        result.update(signed_sum_m=total_signed,bias_m=total_signed/count if count else None)
    return result


def aggregate_views(views, signed=False):
    pooled,equal = {},{}
    for region in ("all",)+REGIONS:
        cells = [view[region] for view in views]
        n = sum(cell["n"] for cell in cells)
        nonempty = [cell for cell in cells if cell["n"]]
        absolute = sum(cell["absolute_sum_m"] for cell in cells)
        squared = sum(cell["squared_sum_m2"] for cell in cells)
        pooled[region] = dict(status="ok" if n else "empty",n=n,
            absolute_sum_m=absolute,squared_sum_m2=squared,
            mean_absolute_m=absolute/n if n else None,
            rmse_m=float(np.sqrt(squared/n)) if n else None,
            max_absolute_m=max((cell["max_absolute_m"] for cell in nonempty),default=None))
        equal[region] = dict(nonempty_views=len(nonempty),empty_views=len(cells)-len(nonempty),
            mean_view_absolute_m=float(np.mean([cell["mean_absolute_m"] for cell in nonempty])) if nonempty else None,
            root_mean_view_mse_m=float(np.sqrt(np.mean([cell["squared_sum_m2"]/cell["n"] for cell in nonempty]))) if nonempty else None)
        if signed:
            total = sum(cell["signed_sum_m"] for cell in cells)
            pooled[region].update(signed_sum_m=total,bias_m=total/n if n else None)
            equal[region]["mean_view_bias_m"] = float(np.mean([cell["bias_m"] for cell in nonempty])) if nonempty else None
    return dict(by_view=views,pooled_pixels=pooled,equal_nonempty_views=equal,
                aggregation="pooled_pixels weights pixels equally; equal_nonempty_views weights each nonempty view equally; empty bins remain null")


def diagnose(pred_depth_m, prediction, reference, mask, aligned_valid_points, *,
             reference_label, alignment, reference_w2c=None):
    """Measure diagnostics on the supplied shared raster mask, without selection.

    ``aligned_valid_points`` is dense_kernel arrays['aligned_predicted'], in
    NumPy row-major boolean-mask order. ``alignment`` contains that same kernel's
    four TRANSFORM_FIELDS. This validates numeric relationships but cannot
    establish external byte seals, source/runtime identity or camera provenance.
    Released-reference axial depth is intentionally unsupported.
    """
    require(reference_label in ("registered_rgb","released"),"Unknown reference label")
    mask = np.asarray(mask)
    require(mask.dtype==np.bool_ and mask.ndim==3 and all(size>0 for size in mask.shape),
            "A nonempty bool [view,height,width] fixed mask is required")
    require(mask.any(),"The fixed reference mask has no valid points")
    shape = mask.shape
    prediction = floating(prediction,"prediction",shape+(3,))
    reference = floating(reference,"reference",shape+(3,))
    depth = floating(pred_depth_m,"predicted metric depth",shape)
    aligned = floating(aligned_valid_points,"kernel-aligned points",(int(mask.sum()),3))
    finite(prediction,"prediction")
    finite(depth,"predicted metric depth")
    finite(reference[mask],"fixed-mask reference")
    finite(aligned,"kernel-aligned points")
    check = transform_check(prediction,reference,mask,aligned,alignment)
    with np.errstate(over="ignore",invalid="ignore"):
        distances = np.linalg.norm(aligned-reference[mask],axis=-1)
    finite(distances,"Paired pixel distances")
    paired = np.zeros(shape,dtype=np.float64)
    paired[mask] = distances
    regions = [[] for _ in REGIONS]
    for view in range(shape[0]):
        with np.errstate(invalid="ignore",over="ignore"):
            bins = partition(reference[view],mask[view])
        require(np.array_equal(np.sum(np.stack(bins),axis=0),mask[view].astype(int)),
                "Reference bins must partition the fixed mask exactly")
        for bucket,region in zip(regions,bins):
            bucket.append(region)
    regions = [np.stack(bucket) for bucket in regions]

    def summarize(errors,signed=False):
        result = []
        for view in range(shape[0]):
            result.append({name:error_stats(errors[view][selected[view]],signed=signed)
                           for name,selected in [("all",mask)]+list(zip(REGIONS,regions))})
        return aggregate_views(result,signed=signed)

    surface = summarize(paired)
    surface.update(quantity="Euclidean distance between corresponding pixels after the common scene alignment; not nearest-neighbor surface distance",
                   transform_refitted=False)
    if reference_label=="registered_rgb":
        require(reference_w2c is not None,"registered_rgb axial error requires its own verified reference cameras")
        cameras,camera_audit = reference_camera_check(reference_w2c,shape[0])
        with np.errstate(invalid="ignore",over="ignore"):
            axial = np.einsum("nhwj,nj->nhw",reference,cameras[:,2,:3])+cameras[:,2,3,None,None]
            signed_errors = depth-axial
        finite(axial[mask],"Reference axial depth")
        finite(signed_errors[mask],"Axial depth error")
        axial_report = summarize(signed_errors,signed=True)
        axial_report.update(supported=True,quantity="predicted raw metric axial depth minus registered_rgb reference-camera z, in meters",
            evaluation_alignment_applied=False,extra_metric_gauge_applied=False,
            reference_nonpositive_on_mask=int((axial[mask]<=0).sum()),camera=camera_audit)
    else:
        require(reference_w2c is None,"Released axial convention is unsupported; do not supply a substitute camera")
        axial_report = dict(supported=False,reason="released reference axial camera convention has not been separately registered and verified")
    output = dict(schema="profiled_m3_boundary_axial_diagnostics_v1",reference_label=reference_label,
        mask_tensor_sha256=tensor_sha256(mask),mask_count=int(mask.sum()),
        mask_counts_by_view=[int(view.sum()) for view in mask],
        point_order="NumPy row-major boolean-mask order",candidate_dependent_exclusions=0,
        predicted_nonpositive_on_mask=int((depth[mask]<=0).sum()),
        partition=dict(regions=list(REGIONS),missing_radius_px=2,jump_radius_px=2,
            jump_threshold_m=.05,priority="missing_neighbor before jump_neighbor before interior",
            source="exact near/partition bodies from scripts/depth_spatial_support.py",
            reference_only=True,region_mask_sha256={name:tensor_sha256(value) for name,value in zip(REGIONS,regions)}),
        alignment_check=check,paired_pixel_distance=surface,metric_axial_depth=axial_report,
        files_opened=False,release_authorized=False,
        external_provenance_verified=False)
    # Catch any invalid aggregate that slipped through even when all inputs were finite.
    json.dumps(output,allow_nan=False)
    return output
