"""Prior-only camera calibration interventions; never mutates scoring targets."""
import numpy as np
from calibration_audit import NOMINAL_FOCAL, REGISTRATION_RGB_FOCAL, resized_k

VARIANTS = ("released", "axis_nominal", "axis_registration")


def calibration_prior(intrinsics, width, height, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown calibration intervention")
    supplied = np.array(intrinsics, copy=True)
    # Guard against applying this 7-Scenes transform to another crop/loader.
    expected = resized_k(NOMINAL_FOCAL, width, height, uniform=True)
    np.testing.assert_allclose(supplied, np.broadcast_to(expected, supplied.shape), rtol=0, atol=5e-5)
    if variant != "released":
        focal = NOMINAL_FOCAL if variant == "axis_nominal" else REGISTRATION_RGB_FOCAL
        supplied[:] = resized_k(focal, width, height, uniform=False)
    return supplied
