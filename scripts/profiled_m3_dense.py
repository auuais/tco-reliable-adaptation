"""Pure M3 scene-surface evaluator using the established dense metric kernel.

The three metric helpers below are copied verbatim from the pinned
third_party/TCO/evaluation/mv_recon/utils.py and checked by AST in CPU tests.
Open3D is imported lazily. The caller must bind runtime, inputs, masks and
sources, seal predictions, authorize reference access and set the registered
seeds/threads before calling. This leaf performs no file IO, caching, model
work, score authorization, reference filtering or per-view alignment.

Reference values outside the fixed mask may be undefined. Prediction values
must be finite everywhere; reference values must be finite on the fixed mask.
Finite negative coordinates are retained. This surface metric alone does not
establish metric-depth validity or absolute calibration accuracy.
"""
import numpy as np
from scipy.spatial import cKDTree as KDTree


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median

def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _open3d():
    import open3d
    return open3d


def validate_inputs(prediction, reference, mask):
    """Validate without changing dtypes, point order, values or the fixed mask."""
    require(isinstance(prediction, np.ndarray) and isinstance(reference, np.ndarray),
            "Prediction and reference must be NumPy arrays")
    require(prediction.dtype in (np.dtype("float32"), np.dtype("float64")) and
            reference.dtype in (np.dtype("float32"), np.dtype("float64")),
            "Prediction/reference must use float32 or float64")
    require(prediction.ndim == 4 and prediction.shape[-1] == 3 and
            all(n > 0 for n in prediction.shape[:-1]), "Expected nonempty [V,H,W,3] rasters")
    require(reference.shape == prediction.shape, "Prediction/reference raster shapes differ")
    require(isinstance(mask, np.ndarray) and mask.dtype == np.dtype("bool") and
            mask.shape == prediction.shape[:-1], "Expected exact bool [V,H,W] fixed mask")
    require(np.isfinite(prediction).all(), "Nonfinite prediction, including outside fixed mask")
    require(np.isfinite(reference[mask]).all(), "Nonfinite fixed-mask reference")
    count = int(mask.sum())
    require(count >= 3, "At least three fixed-mask points are required")
    variances = {}
    for name, array in (("prediction", prediction), ("reference", reference)):
        # Preserve the same dtype and variance arithmetic used by the pinned
        # Umeyama function; no epsilon threshold, rank trimming or fallback.
        points = array[mask].T
        mean = points.mean(axis=1).reshape(-1, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            variance = np.square(points - mean).sum(axis=0).mean()
        require(np.isfinite(variance) and variance > 0, name + " has zero/nonfinite scene variance")
        variances[name] = float(variance)
    return dict(valid_points=count, valid_points_per_view=mask.reshape(mask.shape[0], -1).sum(axis=1).tolist(),
                centered_variance=variances, prediction_dtype=prediction.dtype.str,
                reference_dtype=reference.dtype.str, reference_outside_mask_finite_required=False,
                point_order="NumPy row-major boolean-mask order", mask_modified=False)


def paired_rmse(prediction, reference):
    """Diagnostic paired-pixel Euclidean RMSE, distinct from NN/ICP metrics."""
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1))))


def evaluate(prediction, reference, mask):
    """Return {metrics, diagnostics, arrays} with exact fixed-mask geometry.

    arrays['aligned_predicted'] is the actual post-ICP Open3D geometry in the
    original boolean-mask order. Boundary diagnostics must reuse these points,
    not fit another transform. The ordinary metrics match the pinned evaluate
    function; extra diagnostics do not change the numerical kernel. No image
    colors are required: this registration/metric path never uses them.
    """
    validation = validate_inputs(prediction, reference, mask)
    o3d = _open3d()
    scale, rotation, translation = umeyama(prediction[mask].T, reference[mask].T)
    require(np.isfinite(scale) and scale > 0 and np.isfinite(rotation).all() and
            np.isfinite(translation).all(), "Invalid scene similarity; no fallback is permitted")
    # Keep legacy dtype, expression and operation order exactly.
    aligned = scale * np.einsum("nhwj,ij->nhwi", prediction, rotation) + translation.T
    require(np.isfinite(aligned).all(), "Nonfinite similarity-aligned prediction")
    pc_pred, pc_gt = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pc_pred.points = o3d.utility.Vector3dVector(aligned[mask].reshape(-1, 3))
    pc_gt.points = o3d.utility.Vector3dVector(reference[mask].reshape(-1, 3))
    pre_icp_points = np.asarray(pc_pred.points).copy()
    icp = o3d.pipelines.registration.registration_icp(
        pc_pred, pc_gt, 0.1, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    require(np.asarray(icp.transformation).shape == (4, 4) and
            np.isfinite(icp.transformation).all(), "Invalid ICP transform; no retry is permitted")
    pc_pred.transform(icp.transformation)
    pc_pred.estimate_normals()
    pc_gt.estimate_normals()
    normal_pred, normal_gt = np.asarray(pc_pred.normals), np.asarray(pc_gt.normals)
    require(np.isfinite(normal_pred).all() and np.isfinite(normal_gt).all(), "Nonfinite estimated normals")
    acc, acc_med, nc1, nc1_med = accuracy(pc_gt.points, pc_pred.points, normal_gt, normal_pred)
    comp, comp_med, nc2, nc2_med = completion(pc_gt.points, pc_pred.points, normal_gt, normal_pred)
    metrics = {"accuracy_mean_m": float(acc), "accuracy_median_m": float(acc_med),
               "completeness_mean_m": float(comp), "completeness_median_m": float(comp_med),
               "normal_consistency_mean": float((nc1 + nc2) / 2),
               "normal_consistency_median": float((nc1_med + nc2_med) / 2),
               "nc1_mean": float(nc1), "nc2_mean": float(nc2),
               "valid_points": int(mask.sum()), "similarity_scale": float(scale),
               "similarity_rotation": rotation.tolist(), "similarity_translation": translation.tolist(),
               "icp_transformation": icp.transformation.tolist(),
               "protocol": "upstream_Umeyama_then_rigid_ICP_0.1m_then_nearest_neighbor"}
    arrays = dict(aligned_predicted=np.asarray(pc_pred.points).copy(),
                  reference_points=np.asarray(pc_gt.points).copy(),
                  prediction_normals=normal_pred.copy(), reference_normals=normal_gt.copy(),
                  icp_correspondences=np.asarray(icp.correspondence_set).copy(),
                  similarity_aligned_predicted=pre_icp_points)
    require(all(np.isfinite(value).all() for value in arrays.values()), "Nonfinite dense output")
    require(arrays["aligned_predicted"].shape == arrays["reference_points"].shape == (validation["valid_points"], 3),
            "Fixed-mask point population changed")
    correspondences = arrays["icp_correspondences"]
    require(correspondences.ndim == 2 and correspondences.shape[1] == 2 and
            np.issubdtype(correspondences.dtype, np.integer), "Invalid ICP correspondence array")
    require((correspondences >= 0).all() and (correspondences < validation["valid_points"]).all(),
            "ICP correspondence indices exceed fixed-mask point population")
    if len(correspondences):
        arrays["icp_correspondences_sorted"] = correspondences[np.lexsort((correspondences[:, 1], correspondences[:, 0]))].copy()
    else:
        arrays["icp_correspondences_sorted"] = correspondences.copy()
    diagnostics = dict(validation=validation, icp_fitness=float(icp.fitness),
                       icp_inlier_rmse_m=float(icp.inlier_rmse), icp_correspondence_count=len(correspondences),
                       icp_iterations=None, icp_iterations_status="unavailable in pinned RegistrationResult API",
                       paired_pixel_rmse_before_similarity_m=paired_rmse(prediction[mask], reference[mask]),
                       paired_pixel_rmse_after_similarity_m=paired_rmse(pre_icp_points, reference[mask]),
                       paired_pixel_rmse_after_icp_m=paired_rmse(arrays["aligned_predicted"], reference[mask]),
                       icp_threshold_m=0.1, final_metric_distance_cutoff=None,
                       normals_policy="Open3D default KNN30, fast=True; unsigned nearest-neighbor dot products",
                       supplied_mask_unchanged=True, color_used=False, per_view_alignment=False,
                       release_authorized=False)
    require(all(np.isfinite(value) for value in (
        acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med,
        diagnostics["icp_fitness"], diagnostics["icp_inlier_rmse_m"],
        diagnostics["paired_pixel_rmse_before_similarity_m"],
        diagnostics["paired_pixel_rmse_after_similarity_m"], diagnostics["paired_pixel_rmse_after_icp_m"])),
        "Nonfinite surface metric or diagnostic")
    return dict(metrics=metrics, diagnostics=diagnostics, arrays=arrays)
