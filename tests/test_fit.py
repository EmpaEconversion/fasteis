"""Tests for Circuit.fit()."""
from __future__ import annotations

import numpy as np
import pytest

import eis

FREQS: list[float] = list(np.logspace(-1, 6, 60))


def _synthetic(circuit: eis.Circuit, freqs: list[float] = FREQS) -> np.ndarray:
    return np.asarray(circuit.impedance(freqs), dtype=np.complex128)


@pytest.mark.parametrize(
    ("name", "truth_params", "guess_params"),
    [
        ("R", (100.0,), (60.0,)),
        ("CPE", (3e-4, 0.85), (1e-3, 0.5)),
        ("Zarc", (50.0, 0.2, 0.9), (10.0, 0.05, 0.4)),
    ],
)
def test_fit_recovers_single_element_params_by_name(
    name: str, truth_params: tuple[float, ...], guess_params: tuple[float, ...]
) -> None:
    truth = getattr(eis.Circuit, name)(*truth_params)
    guess = getattr(eis.Circuit, name)(*guess_params)
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z))

    assert result.success
    prefix = name.replace("CPE", "Cpe")
    fields = _field_names(name)
    expected_names = {f"{prefix}0.{field}" for field in fields}
    assert set(result.params) == expected_names
    for field, expected in zip(fields, truth_params, strict=True):
        assert result.params[f"{prefix}0.{field}"] == pytest.approx(expected, rel=1e-4)


def _field_names(name: str) -> list[str]:
    return {
        "R": ["r"],
        "CPE": ["q", "alpha"],
        "Zarc": ["r", "tau_k", "gamma"],
    }[name]


def _make_randles(rs: float, rct: float, cdl: float, aw: float) -> eis.Circuit:
    return eis.Circuit.series([
        eis.Circuit.R(rs),
        eis.Circuit.parallel([
            eis.Circuit.series([eis.Circuit.R(rct), eis.Circuit.W(aw)]),
            eis.Circuit.C(cdl),
        ]),
    ])


def _make_three_branch_parallel(r0: float, r1: float, c: float) -> eis.Circuit:
    return eis.Circuit.parallel([
        eis.Circuit.R(r0),
        eis.Circuit.R(r1),
        eis.Circuit.C(c),
    ])


@pytest.mark.parametrize(
    ("truth", "guess"),
    [
        (_make_randles(20.0, 150.0, 20e-6, 60.0), _make_randles(35.0, 90.0, 8e-6, 90.0)),
        (
            _make_three_branch_parallel(100.0, 500.0, 1e-6),
            _make_three_branch_parallel(60.0, 900.0, 4e-6),
        ),
    ],
    ids=["randles_cell", "three_branch_parallel"],
)
def test_fit_recovers_impedance_for_composed_topologies(truth: eis.Circuit, guess: eis.Circuit) -> None:
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z))

    assert result.success
    got = np.asarray(result.circuit.impedance(FREQS), dtype=np.complex128)
    np.testing.assert_allclose(got, z, rtol=1e-4, atol=1e-8)


def test_fit_weight_modulus_vs_unit_differ() -> None:
    # A circuit whose impedance contributions span several orders of magnitude
    # across the sweep, plus light synthetic noise, so unweighted least-squares
    # (dominated by the largest-|Z| points) diverges from modulus weighting.
    truth = eis.Circuit.series([eis.Circuit.R(1.0), eis.Circuit.CPE(1e-2, 0.7)])
    guess = eis.Circuit.series([eis.Circuit.R(3.0), eis.Circuit.CPE(5e-3, 0.5)])
    freqs = list(np.logspace(0, 6, 40))
    z = _synthetic(truth, freqs)
    rng = np.random.default_rng(0)
    z = z * (1.0 + 0.01 * rng.standard_normal(z.shape))

    modulus = guess.fit(freqs, list(z), weight="modulus")
    unit = guess.fit(freqs, list(z), weight="unit")

    assert modulus.params != pytest.approx(unit.params)


def test_fit_reports_success_and_finite_stderr() -> None:
    truth = _make_randles(20.0, 150.0, 20e-6, 60.0)
    guess = _make_randles(25.0, 120.0, 1.5e-5, 70.0)
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z))

    assert result.success
    assert result.iterations < 200
    assert result.stderr is not None
    for value in result.stderr.values():
        assert np.isfinite(value)
        assert value >= 0.0


def test_fit_rejects_mismatched_lengths() -> None:
    circuit = eis.Circuit.R(100.0)
    with pytest.raises(ValueError):
        circuit.fit(FREQS, [complex(1.0, 0.0)])


def test_fit_rejects_unknown_weight() -> None:
    circuit = eis.Circuit.R(100.0)
    z = _synthetic(circuit)
    with pytest.raises(ValueError):
        circuit.fit(FREQS, list(z), weight="bogus")
