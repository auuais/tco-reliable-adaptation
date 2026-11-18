"""Pure CPU conversion of sealed M3 endpoints to the pinned dense raster.

No files, samples, reference geometry, models or scores are loaded here. The
caller must verify prospective registration and endpoint byte hashes before
decoding arrays. Conversion takes no reference mask; the separate validity
counter never filters points. This module cannot authorize score access.
"""
import hashlib
import json

import cv2
import numpy as np


NATIVE_SHAPE = (5,384,512)
DENSE_SHAPE = (5,392,518)
NATIVE_FIELDS = frozenset(("depth","intrinsics","c2w"))
FINAL_FIELDS = NATIVE_FIELDS | {"g0","scales"}
POINT_FIELDS = frozenset(("depth_m","intrinsics","c2w","points"))


def require(condition,message):
    if not condition:
        raise ValueError(message)


def floating(value,name,shape=None):
    array = np.asarray(value)
    require(array.dtype in (np.dtype("float32"),np.dtype("float64")),
            name+" must use float32 or float64 artifact arithmetic")
    require(shape is None or array.shape==shape,name+" shape differs")
    require(np.isfinite(array).all(),name+" contains nonfinite artifact fields")
    return array


def tolerance(*values):
    return 64.*max(float(np.finfo(np.asarray(value).dtype).eps) for value in values)


def agreement(actual,expected,name,*precision_sources):
    rtol = tolerance(*precision_sources)
    require(np.isfinite(expected).all(),name+" expected arithmetic is nonfinite")
    atol = rtol*max(1.,float(np.max(np.abs(expected))))
    require(np.allclose(actual,expected,rtol=rtol,atol=atol),name+" relationship differs")
    return dict(rtol=rtol,atol=atol,max_absolute_error=float(np.max(np.abs(actual-expected))))


def camera_checks(cameras,precision_source,name):
    """Native *optimized output* rotations must support pinned rigid inversion.

    This is not a check of the approximately rigid supplied source cameras.
    No matrix is orthogonalized or otherwise repaired.
    """
    eps = tolerance(precision_source)
    require(np.allclose(cameras[:,3,:],[0.,0.,0.,1.],rtol=0,atol=eps),name+" has invalid homogeneous row")
    rotation = cameras[:,:3,:3]
    defect = float(np.max(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3))))
    determinant = np.linalg.det(rotation)
    require(defect<=eps and np.allclose(determinant,1.,rtol=0,atol=eps),
            name+" optimized rotations are incompatible with pinned rigid-camera point conversion")
    return dict(max_rotation_orthogonality_error=defect,min_rotation_determinant=float(determinant.min()),
                max_rotation_determinant=float(determinant.max()),absolute_rigidity_tolerance=eps,
                matrices_modified=False)


def intrinsic_checks(intrinsics,name):
    require((intrinsics[:,[0,1],[0,1]]>0).all(),name+" requires positive focal lengths")
    # The pinned point_map uses the zero-skew pinhole form, not general K^-1.
    require(np.array_equal(intrinsics[:,2,:],np.tile([0.,0.,1.],(len(intrinsics),1))),name+" requires canonical pinhole bottom row")
    require((intrinsics[:,0,1]==0).all() and (intrinsics[:,1,0]==0).all(),name+" requires zero skew")


def validate_endpoints(native,final):
    """Validate the exact production field inventories and metric relationships."""
    require(isinstance(native,dict) and set(native)==NATIVE_FIELDS,"Native endpoint field inventory differs")
    require(isinstance(final,dict) and set(final)==FINAL_FIELDS,"Final endpoint field inventory differs")
    nd = floating(native["depth"],"native depth",NATIVE_SHAPE)
    fd = floating(final["depth"],"final depth",NATIVE_SHAPE)
    nk = floating(native["intrinsics"],"native intrinsics",(5,3,3))
    fk = floating(final["intrinsics"],"final intrinsics",(5,3,3))
    nc = floating(native["c2w"],"native c2w",(5,4,4))
    fc = floating(final["c2w"],"final c2w",(5,4,4))
    g0 = floating(final["g0"],"g0",())
    scales = floating(final["scales"],"relative scales",(5,))
    require(float(g0)>0 and (scales>0).all(),"g0 and relative scales must be positive")
    intrinsic_checks(nk,"native intrinsics")
    intrinsic_checks(fk,"final intrinsics")
    # Both are copied unchanged by the common final refiner. A tolerance here
    # would silently admit a different final camera/intrinsics operation.
    require(np.array_equal(nk,fk),"Final intrinsics must equal native optimized intrinsics")
    require(np.array_equal(nc[:,:3,:3],fc[:,:3,:3]),"Final rotations must equal native optimized rotations")
    native_camera = camera_checks(nc.astype(np.float64),nc,"native c2w")
    final_camera = camera_checks(fc.astype(np.float64),nc,"final c2w")
    with np.errstate(over="ignore",invalid="ignore"):
        native_metric = float(g0)*nd.astype(np.float64)
        expected_depth = (scales.astype(np.float64)*float(g0))[:,None,None]*nd.astype(np.float64)
        expected_centers = float(g0)*(nc[:,:3,3].astype(np.float64)-nc[0,:3,3].astype(np.float64))
    require(np.isfinite(native_metric).all(),"Native metric conversion is nonfinite")
    depth_check = agreement(fd,expected_depth,"Final depth = scales*g0*native",fd,g0,scales)
    center_check = agreement(fc[:,:3,3],expected_centers,"Final centers = g0*(native centers-native center0)",fc,g0)
    # Native secondary has the same globally metric origin and centers as final,
    # but no fitted per-view depth multipliers.
    metric_cameras = nc.astype(np.float64).copy()
    metric_cameras[:,:3,3] = expected_centers
    return dict(native_metric_depth=native_metric,final_metric_depth=fd.astype(np.float64).copy(),
        native_intrinsics=nk.astype(np.float64).copy(),final_intrinsics=fk.astype(np.float64).copy(),
        native_metric_c2w=metric_cameras,final_metric_c2w=fc.astype(np.float64).copy(),
        audit=dict(g0=float(g0),relative_scales=scales.astype(float).tolist(),
            depth_consistency=depth_check,camera_center_consistency=center_check,
            native_camera=native_camera,final_camera=final_camera,
            native_secondary="g0*native_depth, shared metric centers, no relative depth scales",
            final_main="saved final depth and metric centers; no additional gauge/scale operation",
            invalid_pixel_exclusions=0,reference_inputs_used=False))


def resize_depth_intrinsics(depth,intrinsics,output_hw):
    """Pinned OpenCV float64 bilinear resize with the half-pixel K transform.

    This low-level helper admits small synthetic rasters. The production entry
    point does not expose a shape override.
    """
    depth = floating(depth,"resize depth")
    require(depth.ndim==3 and min(depth.shape)>0,"Resize depth needs nonempty V,H,W")
    intrinsics = floating(intrinsics,"resize intrinsics",(len(depth),3,3))
    intrinsic_checks(intrinsics,"resize intrinsics")
    require(isinstance(output_hw,(tuple,list)) and len(output_hw)==2 and
            all(type(value) is int and value>0 for value in output_hw),"Invalid output raster")
    height,width = depth.shape[1:]
    out_h,out_w = output_hw
    resize = np.diag([out_w/width,out_h/height,1.])
    resize[:2,2] = (resize[[0,1],[0,1]]-1)/2
    adjusted = resize@intrinsics.astype(np.float64)
    resized = np.stack([cv2.resize(d.astype(float),(out_w,out_h),interpolation=cv2.INTER_LINEAR) for d in depth])
    require(np.isfinite(resized).all() and np.isfinite(adjusted).all(),"Resize arithmetic produced nonfinite values")
    return resized,adjusted


def world_points(depth,intrinsics,c2w):
    """Exact isolated fixed_camera_depth_scale.point_map unit-scale arithmetic.

    That function takes W2C. Here C2W is explicit and inverted once; the result
    remains float32 to match the established dense evaluator input convention.
    The production entry point validates camera rigidity using the original
    artifact precision before the float64 conversion supplied to this helper.
    """
    depth = floating(depth,"point depth")
    require(depth.ndim==3 and min(depth.shape)>0,"Point depth needs nonempty V,H,W")
    intrinsics = floating(intrinsics,"point intrinsics",(len(depth),3,3))
    c2w = floating(c2w,"point c2w",(len(depth),4,4))
    intrinsic_checks(intrinsics,"point intrinsics")
    poses = np.linalg.inv(c2w.astype(np.float64))[:,:3,:]
    require(np.isfinite(poses).all(),"Camera inverse is nonfinite")
    centers = -np.einsum("ni,nij->nj",poses[:,:3,3],poses[:,:3,:3])
    y,x = np.indices(depth.shape[1:],dtype=float)
    output = []
    with np.errstate(over="ignore",invalid="ignore"):
        for view,z in enumerate(depth):
            k = intrinsics[view]
            rays = np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones_like(x)),axis=-1)
            displacement = (rays@poses[view,:3,:3])*z[...,None]
            output.append((centers[view]+displacement).astype(np.float32))
    points = np.stack(output)
    require(np.isfinite(points).all(),"Float32 world-point conversion is nonfinite")
    return points


def convert_endpoints(native,final):
    """Production-only conversion, without source samples or reference masks."""
    checked = validate_endpoints(native,final)
    result = {}
    for endpoint in ("native","final"):
        depth,k = resize_depth_intrinsics(checked[endpoint+"_metric_depth"],
                                          checked[endpoint+"_intrinsics"],DENSE_SHAPE[1:])
        camera = checked[endpoint+"_metric_c2w"]
        points = world_points(depth,k,camera)
        require(depth.shape==DENSE_SHAPE and points.shape==DENSE_SHAPE+(3,),"Production dense shape changed")
        result[endpoint] = dict(depth_m=depth,intrinsics=k,c2w=camera.copy(),points=points)
    result["validation"] = checked["audit"]
    result["validation"].update(native_shape=list(NATIVE_SHAPE),dense_shape=list(DENSE_SHAPE),
        point_dtype="float32",depth_dtype="float64",opencv=cv2.__version__,numpy=np.__version__,
        score_computed=False,files_opened=False,production_shape_override=False)
    return result


def fixed_mask_validity(converted,mask):
    """Count invalid predictions on a supplied immutable mask; never select pixels.

    This does not establish mask provenance. The future scorer must verify it
    against the registered reference and use every mask pixel for metrics.
    """
    require(isinstance(converted,dict) and set(converted)=={"native","final","validation"},"Converted endpoint inventory differs")
    mask = np.asarray(mask)
    require(mask.dtype==np.bool_ and mask.shape==DENSE_SHAPE,"Reference mask must be bool on the production dense raster")
    count = int(mask.sum())
    require(count>0,"Empty fixed reference mask")
    detail,totals = {}, {}
    for name in ("native","final"):
        endpoint = converted[name]
        require(isinstance(endpoint,dict) and set(endpoint)==POINT_FIELDS,"Converted array field inventory differs")
        shapes = dict(depth_m=DENSE_SHAPE,intrinsics=(5,3,3),c2w=(5,4,4),points=DENSE_SHAPE+(3,))
        for field,shape in shapes.items():
            require(isinstance(endpoint[field],np.ndarray) and endpoint[field].shape==shape and
                    endpoint[field].dtype in (np.dtype("float32"),np.dtype("float64")),"Converted field shape/dtype differs")
        nonfinite = {field:int((~np.isfinite(value)).sum()) for field,value in endpoint.items()}
        depth = endpoint["depth_m"]
        nonpositive = mask & np.isfinite(depth) & (depth<=0)
        detail[name] = dict(nonpositive_by_view=[int(value.sum()) for value in nonpositive],
            nonpositive_on_mask=int(nonpositive.sum()),nonpositive_outside_mask=int((~mask&np.isfinite(depth)&(depth<=0)).sum()),
            nonfinite_depth_on_mask=int((mask&~np.isfinite(depth)).sum()),nonfinite_scalar_counts=nonfinite)
        totals[name] = sum(nonfinite.values())
    descriptor = json.dumps([mask.dtype.str,list(mask.shape)],separators=(",",":")).encode()
    return dict(mask_count=count,evaluated_mask_count=count,
        main_nonpositive_depth=detail["final"]["nonpositive_on_mask"],
        native_nonpositive_depth=detail["native"]["nonpositive_on_mask"],
        main_nonfinite_fields=totals["final"],native_nonfinite_fields=totals["native"],
        candidate_main_depth_valid=detail["final"]["nonpositive_on_mask"]==0 and totals["final"]==0,
        mask_tensor_sha256=hashlib.sha256(descriptor+b"\0"+mask.tobytes(order="C")).hexdigest(),
        detail=detail,candidate_dependent_exclusions=0,mask_modified=False,points_filtered=False,
        gate_validity_fields=["mask_count","evaluated_mask_count","main_nonpositive_depth","native_nonpositive_depth",
                              "main_nonfinite_fields","native_nonfinite_fields"])
