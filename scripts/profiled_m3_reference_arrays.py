"""Pure semantic checks for reference arrays after authorized M3 access.

No file IO, prediction access, fitting, mask changes or release authority.
The caller must verify the prepared archive and camera metadata bytes, record
first access, and only then pass their arrays here. Both archived conventions
derive from the same sensor data; they are not independent ground truth.
"""
import numpy as np

from profiled_m3_diagnostics import reference_camera_check, tensor_sha256


SHAPE = (5,392,518)
FIELDS = ("registered_rgb", "released", "valid_mask")
MAX_NATIVE_PIXEL_ERROR = .01


def require(value, message):
    if not value:
        raise ValueError(message)


def project(points, pose, intrinsic):
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, camera[:, 2]


def validate(arrays, scene_spec, known_K):
    """Check the archived shared mask and unfitted primary camera convention."""
    require(isinstance(arrays,dict) and set(arrays)==set(FIELDS), "Reference field inventory differs")
    require(scene_spec.get("reference_fields")==list(FIELDS), "Prepared reference fields differ")
    mask=arrays["valid_mask"]
    require(isinstance(mask,np.ndarray) and mask.dtype==np.dtype('bool') and mask.shape==SHAPE,
            "Reference requires an exact bool production-raster mask")
    spec=scene_spec["fixed_mask"]
    require(spec["shape"]==list(SHAPE) and spec["dtype"]=="bool" and spec["shared_between_references"] is True,
            "Prepared mask convention differs")
    count=int(mask.sum())
    per_view=mask.reshape(5,-1).sum(axis=1)
    require(type(spec["declared_count"]) is int and count==spec["declared_count"], "Reference mask count changed")
    require(bool((per_view>0).all()), "Every reference view must be nonempty")
    mask_hash=tensor_sha256(mask)
    require(spec["tensor_sha256"] in (None,mask_hash), "Previously bound mask tensor changed")
    for label in FIELDS[:2]:
        points=arrays[label]
        require(isinstance(points,np.ndarray) and points.dtype==np.dtype('float32') and points.shape==SHAPE+(3,),
                "Archived reference must retain float32 production shape: "+label)
        require(bool(np.isfinite(points[mask]).all()), "Nonfinite reference on common mask: "+label)
    camera_spec=scene_spec["primary_axial_camera"]
    require(camera_spec["reference"]=="registered_rgb" and camera_spec["dtype"]==np.dtype('float64').str
            and camera_spec["shape"]==[5,4,4] and camera_spec["matrices_orthogonalized"] is False,
            "Prepared primary camera convention differs")
    camera,camera_audit=reference_camera_check(np.asarray(camera_spec["world_to_camera"],dtype=np.float64),5)
    require(camera_audit["camera_preparation_sha256"]==camera_spec["tensor_sha256"],
            "Primary camera differs from prepared matrix tensor")
    intrinsic=np.asarray(known_K)
    require(intrinsic.dtype in (np.dtype('float32'),np.dtype('float64')) and intrinsic.shape==(3,3)
            and bool(np.isfinite(intrinsic).all()), "Invalid registered native intrinsic")
    require(intrinsic[0,0]>0 and intrinsic[1,1]>0 and intrinsic[0,1]==0 and intrinsic[1,0]==0
            and np.array_equal(intrinsic[2],[0.,0.,1.]), "Native intrinsic must be positive zero-skew pinhole K")
    yy,xx=np.indices(SHAPE[1:])
    expected=(np.column_stack([xx.ravel(),yy.ravel()])+.5)*[512/518,384/392]-.5
    errors=[]; depth_ranges=[]
    for view in range(5):
        # Same arithmetic as the original camera audit, including evaluating all
        # raster rows before applying the fixed reference mask.
        pixels,depth=project(arrays["registered_rgb"][view].reshape(-1,3),camera[view],intrinsic)
        valid=mask[view].ravel()
        require(bool(np.isfinite(depth[valid]).all()) and bool((depth[valid]>0).all()),
                "Nonpositive valid primary reference depth")
        error=np.linalg.norm(pixels[valid]-expected[valid],axis=1)
        require(bool(np.isfinite(error).all()), "Nonfinite primary reference projection")
        errors.append(float(error.max()))
        depth_ranges.append([float(depth[valid].min()),float(depth[valid].max())])
    require(max(errors)<MAX_NATIVE_PIXEL_ERROR, "Reference projection convention failed")
    return dict(schema="profiled_m3_reference_semantics_v1",pass_all=True,
                mask_tensor_sha256=mask_hash,mask_count=count,mask_count_per_view=per_view.tolist(),
                reference_shape=list(SHAPE)+[3],reference_dtype="float32",
                primary_camera=camera_audit,known_K_tensor_sha256=tensor_sha256(intrinsic),
                max_native_pixel_error_per_view=errors,strict_max_native_pixel_error=MAX_NATIVE_PIXEL_ERROR,
                primary_axial_depth_range_m_per_view=depth_ranges,
                alternate_axial_camera_status="unsupported_not_registered",
                mask_modified=False,candidate_dependent_exclusions=0,parameters_fitted=0,
                archive_provenance_verified=False,score_access_authorized=False,
                physical_calibration_accuracy_established=False,
                reference_interpretation="two archived conventions of shared sensor data, not independent ground truth")
