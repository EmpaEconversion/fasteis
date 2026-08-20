# Copyright © 2026, Empa.
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

# spans the range the priors will produce, plus alpha at both ends.
PARAM_CASES = [
    {"R0.r": 10.0, "CPE1.q": 1e-5, "CPE1.alpha": 0.85, "R1.r": 100.0, "W1.aw": 50.0},
    {"R0.r": 0.01, "CPE1.q": 1e-2, "CPE1.alpha": 1.0, "R1.r": 0.5, "W1.aw": 0.2},
    {"R0.r": 1e4, "CPE1.q": 1e-9, "CPE1.alpha": 0.5, "R1.r": 1e5, "W1.aw": 1e4},
    {"R0.r": 1.0, "CPE1.q": 1.0, "CPE1.alpha": 0.999, "R1.r": 1.0, "W1.aw": 1.0},
    {"R0.r": 2.5, "CPE1.q": 3e-4, "CPE1.alpha": 0.62, "R1.r": 40.0, "W1.aw": 7.5},
]


def _values(case: dict[str, float]) -> tuple[float, ...]:
    """Return a case as a positional vector in `param_names` order."""
    return tuple(case[name] for name in RANDLES.param_names)


def _freqs(n: int = 64, lo: float = -2.0, hi: float = 6.0) -> np.ndarray:
    return np.logspace(lo, hi, n)


def _impedance(params: tuple[float, ...], freqs: np.ndarray) -> np.ndarray:
    circuit = fasteis.Circuit(RANDLES.circuit_str).with_values(list(params))
    return np.asarray(circuit.impedance(list(freqs)), dtype=np.complex128)


def test_param_names_match_the_rust_circuit() -> None:
    circuit = fasteis.Circuit(RANDLES.circuit_str)
    assert tuple(circuit.param_names()) == RANDLES.param_names


@pytest.mark.parametrize("case", PARAM_CASES)
@pytest.mark.parametrize("estimator", sorted(scales.ESTIMATORS))
def test_normalise_denormalise_round_trip(case: dict[str, float], estimator: str) -> None:
    params = _values(case)
    freqs = _freqs()
    w = TAU * freqs
    z = _impedance(params, freqs)
    k, w_c = scales.estimate(w, z, estimator)

    normalised = RANDLES.to_normalised(np.array([params]), k, w_c)
    recovered = RANDLES.to_physical(normalised, k, w_c)

    assert recovered[0] == pytest.approx(params, rel=1e-12)


@pytest.mark.parametrize("case", PARAM_CASES)
def test_target_encoding_round_trip(case: dict[str, float]) -> None:
    normalised = RANDLES.to_normalised(np.array([_values(case)]), 3.0, 700.0)
    recovered = RANDLES.from_targets(RANDLES.to_targets(normalised))

    assert recovered[0] == pytest.approx(normalised[0], rel=1e-12)


@pytest.mark.parametrize("case", PARAM_CASES)
@pytest.mark.parametrize("estimator", sorted(scales.ESTIMATORS))
def test_normalised_params_reproduce_the_normalised_spectrum(
    case: dict[str, float], estimator: str
) -> None:
    """Main check: the scaled parameters must generate the scaled curve."""
    params = _values(case)
    freqs = _freqs()
    w = TAU * freqs
    z = _impedance(params, freqs)
    k, w_c = scales.estimate(w, z, estimator)

    normalised = RANDLES.to_normalised(np.array([params]), k, w_c)[0]
    # fasteis takes Hz and multiplies by 2*pi internally, so feed w_hat / 2*pi
    z_hat = _impedance(tuple(normalised), (w / w_c) / TAU)

    assert z_hat.real == pytest.approx((z / k).real, rel=1e-11, abs=1e-13)
    assert z_hat.imag == pytest.approx((z / k).imag, rel=1e-11, abs=1e-13)


@pytest.mark.parametrize("case", PARAM_CASES)
def test_scales_from_params_matches_the_cpe_relaxation(
    case: dict[str, float],
) -> None:
    k, w_c = RANDLES.scales_from_params(np.array([_values(case)]))

    assert k[0] == pytest.approx(case["R1.r"])
    tau = (case["R1.r"] * case["CPE1.q"]) ** (1.0 / case["CPE1.alpha"])
    assert w_c[0] == pytest.approx(1.0 / tau)
