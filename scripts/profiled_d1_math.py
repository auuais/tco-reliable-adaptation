"""Pure float64 five-scale diagnostics for the single registered D1 probe.

Inputs are fixed-preparation support/query matrices, never model parameters.
The final-minus-initial matrix direction is an output-space secant.  A full and
detached derivative are compared at exactly the same final forward state.  No
fold averaging, candidate selection, data access or model operation occurs here.
"""
import math

import numpy as np


VIEWS = 5
FD_STEPS = (0.01, 0.003, 0.001)
IDENTITY_RTOL = 1e-10
IDENTITY_ATOL = 1e-10
STATIONARITY_BACKWARD_ERROR_MAX = 1e-12
FD_RTOL = 1e-5
FD_ATOL = 1e-8
FD_MINIMUM_PASSES = 2
FD_SMALLEST_STEP_MUST_PASS = True
# These conservative resolution labels use the same scale as the identity
# tolerance. Synthetic tie/boundary/rescaling fixtures pin their meaning. They
# are neither statistical thresholds nor universal roundoff-error guarantees.
LOSS_RESOLUTION_ATOL = 1e-10
LOSS_RESOLUTION_RTOL = 1e-10
NORM_RESOLUTION_ATOL = 1e-10
NORM_RESOLUTION_RTOL = 1e-10


def tolerance_spec():
    """Prospective numerical contract, selected using only synthetic fixtures."""
    return dict(
        arithmetic="numpy_float64", columns=VIEWS,
        identity_rtol=IDENTITY_RTOL, identity_atol=IDENTITY_ATOL,
        stationarity_relative_backward_error_max=STATIONARITY_BACKWARD_ERROR_MAX,
        finite_difference_steps=list(FD_STEPS), finite_difference_rtol=FD_RTOL,
        finite_difference_atol=FD_ATOL, finite_difference_minimum_passes=FD_MINIMUM_PASSES,
        finite_difference_smallest_step_must_pass=FD_SMALLEST_STEP_MUST_PASS,
        finite_difference_rule="Both residual-vector and scalar-loss checks must pass per step",
        query_loss_resolution=dict(atol=LOSS_RESOLUTION_ATOL, rtol=LOSS_RESOLUTION_RTOL,
                                   formula="atol+rtol*max(abs(J0),abs(J1))",
                                   resolved_rule="abs(J0-J1) must be strictly greater than the floor"),
        compensation_norm_resolution=dict(atol=NORM_RESOLUTION_ATOL, rtol=NORM_RESOLUTION_RTOL,
                                          formula="atol+rtol*max(norm(t),norm(c),norm(t+c))",
                                          attenuation_rule="opposing nonzero t,c AND norm(t)-norm(t+c) strictly exceeds floor"),
        interpretation_scope="Float64 arithmetic resolution only; not statistical significance or accuracy",
        zero_denominator="undefined/null; never add epsilon or discard a row",
        scale_policy="All five fitted scales, including both FD perturbations, must be finite and positive",
        zero_ridge="Allowed only with full support-column rank; production wrapper requires its registered positive ridge",
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _real_array(value, name, ndim):
    array = np.asarray(value)
    _require(array.ndim == ndim and all(array.shape), name + " has wrong or empty shape")
    _require(array.dtype.kind in "fiu", name + " must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    _require(np.isfinite(array).all(), name + " must be finite; no row filtering is permitted")
    return array


def _scalar(value, name, nonnegative=False):
    _require(not isinstance(value, (bool, np.bool_)), name + " cannot be boolean")
    _require(isinstance(value, (int, float, np.integer, np.floating)), name + " must be numeric")
    result = float(value)
    _require(math.isfinite(result), name + " must be finite")
    _require(not nonnegative or result >= 0, name + " must be nonnegative")
    return result


def _finite(value):
    result = float(value)
    _require(math.isfinite(result), "Float64 arithmetic produced a nonfinite result")
    return result


def _norm(value):
    return _finite(np.linalg.norm(value))


def _list(value):
    _require(np.isfinite(value).all(), "Nonfinite vector cannot be serialized as diagnostic evidence")
    return np.asarray(value, dtype=np.float64).tolist()


def _cosine(first, second):
    first_norm, second_norm = _norm(first), _norm(second)
    if first_norm == 0 or second_norm == 0:
        return None
    value = _finite(np.dot(first / first_norm, second / second_norm))
    _require(-1-1e-12 <= value <= 1+1e-12, "Cosine outside floating-point roundoff range")
    return min(1.0, max(-1.0, value))


def classify_query_loss_change(J0, J1):
    """Resolve permitted-query loss changes above a fixed arithmetic floor.

    This preserves the raw change even when its direction is unresolved. It
    performs no statistical test, supplies no uncertainty interval and cannot
    turn adaptation-query observations into accuracy/generalization evidence.
    """
    J0, J1 = _scalar(J0, "J0", True), _scalar(J1, "J1", True)
    delta = _finite(J0-J1)
    floor = _finite(LOSS_RESOLUTION_ATOL + LOSS_RESOLUTION_RTOL*max(abs(J0), abs(J1)))
    classification = "decrease" if delta > floor else "increase" if delta < -floor else "unresolved"
    return dict(
        schema="profiled_d1_query_loss_interpretation_v1", J0=J0, J1=J1,
        J0_minus_J1=delta, arithmetic_floor=floor, classification=classification,
        resolved_decrease=classification == "decrease", resolved_increase=classification == "increase",
        within_arithmetic_resolution=abs(delta) <= floor,
        scope="Adaptation-query loss and float64 arithmetic resolution only",
        statistical_test_performed=False, accuracy_improvement_established=False,
    )


def classify_compensation(t, c):
    """Label output-space attenuation only when opposition and norm tests agree.

    Norm reductions/amplifications smaller than the declared scale-dependent
    floor are neutral at this arithmetic resolution. A zero direct vector makes
    attenuation/amplification relative to it undefined, even if c is nonzero.
    Compensation can oppose t yet amplify it by overshooting; this is reported.
    """
    t, c = _real_array(t, "direct vector", 1), _real_array(c, "compensation vector", 1)
    _require(t.shape == c.shape, "Direct and compensation vector shapes differ")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        combined = t+c
        nt, nc, combined_norm = _norm(t), _norm(c), _norm(combined)
        dot = _finite(np.dot(t, c))
        reduction = _finite(nt-combined_norm)
        floor = _finite(NORM_RESOLUTION_ATOL + NORM_RESOLUTION_RTOL*max(nt, nc, combined_norm))
        opposing = None if nt == 0 or nc == 0 else dot < 0
        consistent = True
        if nt == 0:
            classification, reason = "undefined", "zero_direct_vector"
        elif reduction > floor:
            if opposing:
                classification, reason = "attenuation", "opposition_and_resolved_norm_reduction"
            else:
                classification, reason = "undefined", "inconsistent_norm_and_dot_geometry"
                consistent = False
        elif reduction < -floor:
            classification, reason = "amplification", "resolved_norm_increase"
        else:
            classification, reason = "neutral", "norm_change_within_arithmetic_resolution"
    return dict(
        schema="profiled_d1_compensation_interpretation_v1",
        norm_direct=nt, norm_compensation=nc, norm_combined=combined_norm,
        direct_dot_compensation=dot, cosine_direct_compensation=_cosine(t, c), opposing=opposing,
        norm_reduction=reduction, arithmetic_floor=floor,
        survival_ratio=None if nt == 0 else _finite(combined_norm/nt),
        classification=classification, reason=reason, arithmetic_geometry_consistent=consistent,
        resolved_attenuation=classification == "attenuation",
        resolved_amplification=classification == "amplification",
        scope="Measured output-space vectors and float64 arithmetic resolution only",
        statistical_test_performed=False, harmfulness_established=False,
        parameter_space_generalization_established=False, redesign_authorized=False,
    )


def _support(A, b, ridge):
    A = _real_array(A, "A", 2)
    b = _real_array(b, "b", 1)
    ridge = _scalar(ridge, "ridge", True)
    _require(A.shape[1] == VIEWS and b.shape == (len(A),), "Expected A[N,5] and b[N]")
    singular_values = np.linalg.svd(A, compute_uv=False)
    rank_tolerance = _finite(np.finfo(np.float64).eps * max(A.shape) * singular_values[0])
    rank = int((singular_values > rank_tolerance).sum())
    _require(ridge > 0 or rank == VIEWS, "Zero ridge requires full support-column rank")
    return A, b, ridge, singular_values, rank, rank_tolerance


def _state(A, b, ridge):
    A, b, ridge, singular_values, rank, rank_tolerance = _support(A, b, ridge)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        H = A.T @ A + ridge * np.eye(VIEWS, dtype=np.float64)
        rhs = A.T @ b + ridge * np.ones(VIEWS, dtype=np.float64)
        _require(np.isfinite(H).all() and np.isfinite(rhs).all(), "Support normal equations overflow")
        try:
            L = np.linalg.cholesky(H)
            s = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Support normal equations are not numerically positive definite") from exc
        _require(np.isfinite(s).all() and (s > 0).all(), "All fitted scales must be finite and positive")
        r = A @ s - b
        stationarity = A.T @ r + ridge * (s - 1)
        norm_equation_residual = H @ s - rhs
        backward_denominator = _finite(np.linalg.norm(H, ord=2) * np.linalg.norm(s) + np.linalg.norm(rhs))
        _require(backward_denominator > 0, "Undefined support backward-error denominator")
        stationarity_relative = _finite(np.linalg.norm(stationarity) / backward_denominator)
        normal_relative = _finite(np.linalg.norm(norm_equation_residual) / backward_denominator)
        condition = _finite(np.linalg.cond(H, p=2))
    diagnostics = dict(
        scales=_list(s), ridge=ridge, support_rows=len(A), support_data_rank=rank,
        support_singular_values=_list(singular_values), support_rank_tolerance=rank_tolerance,
        H_condition=condition, support_residual_l2=_norm(r),
        stationarity_residual_l2=_norm(stationarity),
        normal_equation_residual_l2=_norm(norm_equation_residual),
        backward_error_denominator=backward_denominator,
        stationarity_relative_backward_error=stationarity_relative,
        normal_equation_relative_backward_error=normal_relative,
        stationarity_pass=bool(stationarity_relative <= STATIONARITY_BACKWARD_ERROR_MAX
                               and normal_relative <= STATIONARITY_BACKWARD_ERROR_MAX),
        scale_positive=True,
        identifiability_note="Ridge solvability is separate from support data rank",
    )
    return A, b, ridge, H, L, s, r, diagnostics


def solve_scales(A, b, ridge):
    """Return five positive float64 ridge scales with prior center exactly ones."""
    return _state(A, b, ridge)[5]


def _query(Q, d):
    Q = _real_array(Q, "Q", 2)
    d = _real_array(d, "d", 1)
    _require(Q.shape[1] == VIEWS and d.shape == (len(Q),), "Expected Q[M,5] and d[M]")
    return Q, d


def _loss(e):
    return _finite(0.5 * np.dot(e, e))


def finite_response(A0, Q0, A1, Q1, b, d, ridge):
    """Exact finite response with fixed b,d,ridge and all declared rows retained.

    t=(Q1-Q0)s0 is a frozen-s0 counterfactual, not the forward-identical detached
    derivative. c=Q1(s1-s0) completes the exact identity e1-e0=t+c.
    """
    st0, st1 = _state(A0, b, ridge), _state(A1, b, ridge)
    A0, A1 = st0[0], st1[0]
    _require(A0.shape == A1.shape, "Initial/final support membership or matrix shape differs")
    Q0, d = _query(Q0, d)
    Q1, _ = _query(Q1, d)
    _require(Q0.shape == Q1.shape, "Initial/final query membership or matrix shape differs")
    s0, s1 = st0[5], st1[5]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        e0, e1 = Q0 @ s0 - d, Q1 @ s1 - d
        t, c = (Q1-Q0) @ s0, Q1 @ (s1-s0)
        change, combined = e1-e0, t+c
        identity_error = change-combined
        J0, J1 = _loss(e0), _loss(e1)
        norm_t, norm_c, norm_combined = _norm(t), _norm(c), _norm(combined)
        identity_pass = bool(np.allclose(change, combined, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL))
    compensation_info = classify_compensation(t, c)
    return dict(
        schema="profiled_d1_finite_response_v1",
        state0=st0[7], state1=st1[7], query_rows=len(Q0), lambda_value=st0[2],
        s0=_list(s0), s1=_list(s1), e0=_list(e0), e1=_list(e1),
        direct_t=_list(t), compensation_c=_list(c), e1_minus_e0=_list(change),
        t_plus_c=_list(combined), norm_t=norm_t, norm_c=norm_c,
        cosine_t_c=_cosine(t, c), norm_t_plus_c=norm_combined,
        norm_t_plus_c_over_norm_t=None if norm_t == 0 else _finite(norm_combined/norm_t),
        zero_direct_change=norm_t == 0, zero_compensation=norm_c == 0,
        identity_residual_l2=_norm(identity_error),
        identity_residual_max_abs=_finite(np.max(np.abs(identity_error))), identity_pass=identity_pass,
        J0=J0, J1=J1, J0_minus_J1=_finite(J0-J1),
        query_loss_interpretation=classify_query_loss_change(J0, J1),
        compensation_interpretation=compensation_info,
        pass_all=bool(identity_pass and st0[7]["stationarity_pass"] and st1[7]["stationarity_pass"]
                      and compensation_info["arithmetic_geometry_consistent"]),
        interpretation="Exact finite output-space response; not a parameter Jacobian or accuracy claim",
        fixed_s0_counterfactual_is_detached_gradient_arm=False,
    )


def finite_difference_acceptance(rows):
    """All fixed steps must be present; two pass, including the smallest step."""
    _require(len(rows) == len(FD_STEPS), "Incomplete finite-difference inventory")
    _require([row.get("h") for row in rows] == list(FD_STEPS), "Changed finite-difference steps/order")
    decisions = []
    for row in rows:
        fields = ("residual_vector_pass", "loss_pass", "perturbed_stationarity_pass", "pass_check")
        _require(all(type(row.get(key)) is bool for key in fields), "Malformed finite-difference decision")
        actual = row["residual_vector_pass"] and row["loss_pass"] and row["perturbed_stationarity_pass"]
        _require(row["pass_check"] == actual, "Stored finite-difference decision differs from its components")
        decisions.append(actual)
    return bool(sum(decisions) >= FD_MINIMUM_PASSES
                and (decisions[-1] or not FD_SMALLEST_STEP_MUST_PASS))


def directional_response(A, Q, b, d, ridge, dotA, dotQ):
    """Full response and detached control at the SAME final forward state.

    dotA,dotQ are a declared output-space direction.  The exact derivative keeps
    the residual-response term dotA.T@r.  Every finite-difference perturbation
    refits all five scales; no detached scale is used in the full FD calculation.
    """
    state = _state(A, b, ridge)
    A, b, ridge, H, L, s, r, diagnostics = state
    Q, d = _query(Q, d)
    dotA, dotQ = _real_array(dotA, "dotA", 2), _real_array(dotQ, "dotQ", 2)
    _require(dotA.shape == A.shape and dotQ.shape == Q.shape, "Directional matrix shapes differ")

    def hsolve(value):
        return np.linalg.solve(L.T, np.linalg.solve(L, value))

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        e = Q @ s - d
        tS, tQ = dotA @ s, dotQ @ s
        direct_rhs, residual_rhs = A.T @ tS, dotA.T @ r
        ds_direct = -hsolve(direct_rhs)
        ds_residual = -hsolve(residual_rhs)
        ds = -hsolve(direct_rhs + residual_rhs)
        compensation = Q @ ds
        residual_component = Q @ ds_residual
        direct_component = Q @ ds_direct
        de = tQ + compensation
        full = _finite(np.dot(e, de))
        detached = _finite(np.dot(e, tQ))
        same_loss = _loss(e)
        response_identity_error = de - (tQ + direct_component + residual_component)
        response_identity_pass = bool(np.allclose(de, tQ+direct_component+residual_component,
                                                 rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL))
        norm_tQ, norm_de = _norm(tQ), _norm(de)
        rows = []
        for h in FD_STEPS:
            endpoints = []
            for sign in (-1, 1):
                moved = _state(A + sign*h*dotA, b, ridge)
                moved_e = (Q + sign*h*dotQ) @ moved[5] - d
                endpoints.append((moved, moved_e, _loss(moved_e)))
            minus, plus = endpoints
            fd_e = (plus[1]-minus[1]) / (2*h)
            fd_J = _finite((plus[2]-minus[2]) / (2*h))
            error = fd_e-de
            vector_pass = bool(np.allclose(fd_e, de, rtol=FD_RTOL, atol=FD_ATOL))
            scalar_tolerance = FD_ATOL + FD_RTOL * abs(full)
            scalar_error = _finite(abs(fd_J-full))
            loss_pass = bool(scalar_error <= scalar_tolerance)
            both_stationary = bool(minus[0][7]["stationarity_pass"] and plus[0][7]["stationarity_pass"])
            rows.append(dict(
                h=h, minus_scales=_list(minus[0][5]), plus_scales=_list(plus[0][5]),
                minus_residual=_list(minus[1]), plus_residual=_list(plus[1]),
                minus_loss=minus[2], plus_loss=plus[2],
                finite_difference_residual=_list(fd_e), finite_difference_loss=fd_J,
                residual_error_l2=_norm(error), residual_error_max_abs=_finite(np.max(np.abs(error))),
                residual_relative_l2=None if norm_de == 0 else _finite(np.linalg.norm(error)/norm_de),
                loss_error_absolute=scalar_error, loss_tolerance=scalar_tolerance,
                residual_vector_pass=vector_pass, loss_pass=loss_pass,
                perturbed_stationarity_pass=both_stationary,
                pass_check=bool(vector_pass and loss_pass and both_stationary),
            ))
    fd_pass = finite_difference_acceptance(rows)
    compensation_info = classify_compensation(tQ, compensation)
    return dict(
        schema="profiled_d1_directional_response_v1", state=diagnostics,
        query_rows=len(Q), lambda_value=ridge, scales=_list(s), residual=_list(e),
        direction="Declared output-space direction; no network Jacobian claim",
        zero_matrix_direction=bool(not np.any(dotA) and not np.any(dotQ)),
        support_direct_tS=_list(tS), query_direct_tQ=_list(tQ),
        direct_stationarity_rhs=_list(direct_rhs), residual_stationarity_rhs=_list(residual_rhs),
        scale_response=_list(ds), query_compensation=_list(compensation),
        residual_response_component=_list(residual_component),
        gauss_newton_response_component=_list(direct_component),
        query_full_response=_list(de),
        norm_tQ=norm_tQ, norm_query_compensation=_norm(compensation), norm_full_response=norm_de,
        cosine_direct_compensation=_cosine(tQ, compensation),
        survival_ratio=None if norm_tQ == 0 else _finite(norm_de/norm_tQ),
        residual_response_component_norm=_norm(residual_component),
        full_outer_derivative=full, detached_at_same_state_outer_derivative=detached,
        full_forward_loss=same_loss, detached_forward_loss=same_loss,
        forward_state_identical=True, exact_residual_response_term_included=True,
        response_identity_residual_l2=_norm(response_identity_error), response_identity_pass=response_identity_pass,
        finite_differences=rows, finite_difference_pass=fd_pass,
        compensation_interpretation=compensation_info,
        pass_all=bool(diagnostics["stationarity_pass"] and response_identity_pass and fd_pass
                      and compensation_info["arithmetic_geometry_consistent"]),
    )


def analyze_response(A0, Q0, A1, Q1, b, d, ridge, *, with_derivative=True):
    """One path and one fold only; caller retains paths/folds separately.

    The derivative is at the final matrix state with direction A1-A0,Q1-Q0.
    Production calls this only for the B1-B0 direction selected by the design;
    the FP32 path may use ``with_derivative=False`` for its finite response.
    """
    _require(type(with_derivative) is bool, "with_derivative must be boolean")
    finite = finite_response(A0, Q0, A1, Q1, b, d, ridge)
    derivative = None
    if with_derivative:
        derivative = directional_response(A1, Q1, b, d, ridge,
                                          np.asarray(A1, dtype=np.float64)-np.asarray(A0, dtype=np.float64),
                                          np.asarray(Q1, dtype=np.float64)-np.asarray(Q0, dtype=np.float64))
    return dict(
        schema="profiled_d1_path_response_v1", finite_response=finite, derivative=derivative,
        tolerances=tolerance_spec(),
        pass_all=bool(finite["pass_all"] and (derivative is None or derivative["pass_all"])),
        folds_aggregated=False, numerical_check_only=True,
        accuracy_improvement_established=False, redesign_authorized=False, rental_authorized=False,
    )
