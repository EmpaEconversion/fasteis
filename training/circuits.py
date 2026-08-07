"""The fixed circuit the model is trained for, and its normalisation algebra.

Randles impedance has two exact scaling symmetries: `Z -> k*Z` and `w -> w/s`. The
network is fed spectra with both removed, so it only ever learns curve shape, and
the scales are multiplied back onto its predictions afterwards.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

CIRCUIT_STRING = "R0-(CPE1,R1-W1)"

# order matches fasteis.Circuit(CIRCUIT_STRING).param_names()
PARAM_NAMES: tuple[str, ...] = ("R0.r", "CPE1.q", "CPE1.alpha", "R1.r", "W1.aw")
N_PARAMS = len(PARAM_NAMES)

R0, Q, ALPHA, R1, AW = range(N_PARAMS)

# alpha is a fractional exponent and stays in linear space; the rest span decades
LOG_PARAMS: tuple[int, ...] = (R0, Q, R1, AW)

# alpha is predicted unbounded, so it is clamped before the circuit is evaluated
ALPHA_RANGE = (0.05, 1.0)

# How each parameter picks up the impedance scale k and frequency scale w_c:
#
#   physical = normalised * k**a * w_c**(b + c * params[i])
#
# with i = -1 when the exponent has no parameter dependence. One row per parameter,
# columns (a, b, c, i). This covers every element in the library -- resistances are
# (1, 0, 0, -1), capacitances (-1, -1, 0, -1), inductances (1, -1, 0, -1), time
# constants (0, -1, 0, -1) -- so adding a circuit means adding rows, not code.
SCALING: NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, -1],  # R0.r
        [-1.0, 0.0, -1.0, ALPHA],  # CPE1.q, exponent -alpha
        [0.0, 0.0, 0.0, -1],  # CPE1.alpha
        [1.0, 0.0, 0.0, -1],  # R1.r
        [1.0, 0.5, 0.0, -1],  # W1.aw
    ]
)


def _scale_factors(
    params: NDArray[np.float64],
    k: float | NDArray[np.float64],
    w_c: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """`k**a * w_c**(b + c*param)` per parameter, shape (batch, N_PARAMS)."""
    a, b, c, idx = SCALING[:, 0], SCALING[:, 1], SCALING[:, 2], SCALING[:, 3].astype(int)

    exponent = np.broadcast_to(b, params.shape).copy()
    dependent = idx >= 0
    if dependent.any():
        exponent[:, dependent] += c[dependent] * params[:, idx[dependent]]

    return k**a * w_c**exponent


def to_normalised(
    params: NDArray[np.float64],
    k: float | NDArray[np.float64],
    w_c: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Physical parameters -> parameters of the same curve after Z/k and w/w_c."""
    params = np.atleast_2d(params)
    k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
    w_c = np.asarray(w_c, dtype=np.float64).reshape(-1, 1)
    # exponents depending on alpha use it from the parameters, and alpha is invariant
    return params / _scale_factors(params, k, w_c)


def to_physical(
    params_hat: NDArray[np.float64],
    k: float | NDArray[np.float64],
    w_c: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Inverse of `to_normalised`."""
    params_hat = np.atleast_2d(params_hat)
    k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
    w_c = np.asarray(w_c, dtype=np.float64).reshape(-1, 1)
    return params_hat * _scale_factors(params_hat, k, w_c)


def to_targets(params_hat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalised parameters -> regression targets (log10 except alpha)."""
    out = np.array(params_hat, dtype=np.float64, copy=True)
    out[..., LOG_PARAMS] = np.log10(out[..., LOG_PARAMS])
    return out


def from_targets(targets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of `to_targets`."""
    out = np.array(targets, dtype=np.float64, copy=True)
    out[..., LOG_PARAMS] = 10.0 ** out[..., LOG_PARAMS]
    return out


def scales_from_params(
    params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get scale factors from model parameters.

    The (k, w_c) implied by a parameter vector itself, for two-pass inference.
    Uses the charge-transfer resistance as the impedance scale and the CPE
    relaxation rate as the frequency scale.
    """
    params = np.atleast_2d(params)
    r1 = params[:, R1]
    alpha = params[:, ALPHA]
    tau = (r1 * params[:, Q]) ** (1.0 / alpha)
    return r1, 1.0 / tau
