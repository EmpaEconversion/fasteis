"""Candidate estimators for the impedance scale `k` and frequency scale `w_c`.

The scales should ideally collapse circuits with different parameters onto a
similar normalized space.

The estimators take angular frequencies and complex impedances, and return
`(k, w_c)` as floats.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Estimator = Callable[[NDArray[np.float64], NDArray[np.complex128]], tuple[float, float]]

# below this total weight the weighted estimators have nothing to average over
_MIN_WEIGHT = 1e-9


def window(w: NDArray[np.float64], z: NDArray[np.complex128]) -> tuple[float, float]:
    """Depends only on the sweep, not the curve. The known-bad control."""
    return float(np.median(np.abs(z))), float(np.exp(np.mean(np.log(w))))


def _weighted(
    w: NDArray[np.float64], z: NDArray[np.complex128], u: NDArray[np.float64]
) -> tuple[float, float]:
    """Log-space weighted means of |Z| and w, falling back to `window` if u vanishes."""
    total = u.sum()
    if total < _MIN_WEIGHT:
        return window(w, z)
    k = float(np.exp(np.sum(u * np.log(np.abs(z))) / total))
    w_c = float(np.exp(np.sum(u * np.log(w)) / total))
    return k, w_c


def reactive_centroid(
    w: NDArray[np.float64], z: NDArray[np.complex128]
) -> tuple[float, float]:
    """Weights by -sin(phase), bounded in [0, 1] and scale-free.

    Points measured in a featureless region carry ~0 weight, so widening the
    sweep into one barely moves the result.
    """
    u = np.clip(-z.imag / np.abs(z), 0.0, None)
    return _weighted(w, z, u)


def imag_weighted(
    w: NDArray[np.float64], z: NDArray[np.complex128]
) -> tuple[float, float]:
    """Weights by -Im(Z) unnormalised, so the Warburg tail dominates."""
    u = np.clip(-z.imag, 0.0, None)
    return _weighted(w, z, u)


def imag_peak(w: NDArray[np.float64], z: NDArray[np.complex128]) -> tuple[float, float]:
    """Peak of -Im(Z): k = 2*max(-Im Z) recovers R_ct for an ideal single arc.

    Physically meaningful but reads a single noisy point, and is meaningless
    when the peak lies outside the measured window.
    """
    neg_im = -z.imag
    i = int(np.argmax(neg_im))
    if neg_im[i] <= 0.0:
        return window(w, z)
    return 2.0 * float(neg_im[i]), float(w[i])


ESTIMATORS: dict[str, Estimator] = {
    "window": window,
    "reactive_centroid": reactive_centroid,
    "imag_weighted": imag_weighted,
    "imag_peak": imag_peak,
}

DEFAULT = "reactive_centroid"


def estimate(
    w: NDArray[np.float64], z: NDArray[np.complex128], name: str = DEFAULT
) -> tuple[float, float]:
    """Apply a named estimator."""
    return ESTIMATORS[name](w, z)
