"""Checks the normalisation in circuits.py against the Rust implementation.

Tests the scaling table for circuit parameters, i.e., how the impedance/
frequency rescaling affects parameters like R1.r, CEP1.q, W1.aw etc.
"""

from __future__ import annotations

import numpy as np
import pytest

import fasteis
from training import circuits, scales

RANDLES = circuits.get("randles")
TAU = 2.0 * np.pi

# spans the range the priors will produce, plus alpha at both ends
PARAM_CASES = [
    (10.0, 1e-5, 0.85, 100.0, 50.0),
    (0.01, 1e-2, 1.0, 0.5, 0.2),
    (1e4, 1e-9, 0.5, 1e5, 1e4),
    (1.0, 1.0, 0.999, 1.0, 1.0),
    (2.5, 3e-4, 0.62, 40.0, 7.5),
]


def _freqs(n: int = 64, lo: float = -2.0, hi: float = 6.0) -> np.ndarray:
    return np.logspace(lo, hi, n)


def _impedance(params: tuple[float, ...], freqs: np.ndarray) -> np.ndarray:
    circuit = fasteis.Circuit(RANDLES.circuit_str).with_values(list(params))
    return np.asarray(circuit.impedance(list(freqs)), dtype=np.complex128)


def test_param_names_match_the_rust_circuit() -> None:
    circuit = fasteis.Circuit(RANDLES.circuit_str)
    assert tuple(circuit.param_names()) == RANDLES.param_names


@pytest.mark.parametrize("params", PARAM_CASES)
@pytest.mark.parametrize("estimator", sorted(scales.ESTIMATORS))
def test_normalise_denormalise_round_trip(
    params: tuple[float, ...], estimator: str
) -> None:
    freqs = _freqs()
    w = TAU * freqs
    z = _impedance(params, freqs)
    k, w_c = scales.estimate(w, z, estimator)

    normalised = RANDLES.to_normalised(np.array([params]), k, w_c)
    recovered = RANDLES.to_physical(normalised, k, w_c)

    assert recovered[0] == pytest.approx(params, rel=1e-12)


@pytest.mark.parametrize("params", PARAM_CASES)
def test_target_encoding_round_trip(params: tuple[float, ...]) -> None:
    normalised = RANDLES.to_normalised(np.array([params]), 3.0, 700.0)
    recovered = RANDLES.from_targets(RANDLES.to_targets(normalised))

    assert recovered[0] == pytest.approx(normalised[0], rel=1e-12)


@pytest.mark.parametrize("params", PARAM_CASES)
@pytest.mark.parametrize("estimator", sorted(scales.ESTIMATORS))
def test_normalised_params_reproduce_the_normalised_spectrum(
    params: tuple[float, ...], estimator: str
) -> None:
    """Main check: the scaled parameters must generate the scaled curve."""
    freqs = _freqs()
    w = TAU * freqs
    z = _impedance(params, freqs)
    k, w_c = scales.estimate(w, z, estimator)

    normalised = RANDLES.to_normalised(np.array([params]), k, w_c)[0]
    # fasteis takes Hz and multiplies by 2*pi internally, so feed w_hat / 2*pi
    z_hat = _impedance(tuple(normalised), (w / w_c) / TAU)

    assert z_hat.real == pytest.approx((z / k).real, rel=1e-11, abs=1e-13)
    assert z_hat.imag == pytest.approx((z / k).imag, rel=1e-11, abs=1e-13)


@pytest.mark.parametrize("params", PARAM_CASES)
def test_scales_from_params_matches_the_cpe_relaxation(
    params: tuple[float, ...],
) -> None:
    k, w_c = RANDLES.scales_from_params(np.array([params]))

    assert k[0] == pytest.approx(params[3])
    tau = (params[3] * params[1]) ** (1.0 / params[2])
    assert w_c[0] == pytest.approx(1.0 / tau)
