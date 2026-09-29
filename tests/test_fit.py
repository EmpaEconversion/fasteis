# Copyright © 2026, Empa.
"""Tests for Circuit.fit()."""

from __future__ import annotations

import numpy as np
import pytest

import fasteis
from tests.circuit_cases import element

FREQS: list[float] = list(np.logspace(-1, 6, 60))


def _synthetic(circuit: fasteis.Circuit, freqs: list[float] = FREQS) -> np.ndarray:
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
    truth = fasteis.Series([element(name, truth_params)])
    guess = fasteis.Series([element(name, guess_params)])
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z))

    assert result.success
    prefix = name.replace("CPE", "Cpe")
    fields = _field_names(name)
    expected_names = {f"{prefix}0.{field}" for field in fields}
    assert set(result.params) == expected_names
    assert len(fields) == len(truth_params)
    for field, expected in zip(fields, truth_params):
        assert result.params[f"{prefix}0.{field}"] == pytest.approx(expected, rel=1e-4)


def _field_names(name: str) -> list[str]:
    return {
        "R": ["r"],
        "CPE": ["q", "alpha"],
        "Zarc": ["r", "tau_k", "gamma"],
    }[name]


def _make_randles(rs: float, rct: float, cdl: float, aw: float) -> fasteis.Circuit:
    return fasteis.Series(
        [
            fasteis.R(rs),
            fasteis.Parallel(
                [
                    fasteis.Series([fasteis.R(rct), fasteis.W(aw)]),
                    fasteis.C(cdl),
                ]
            ),
        ]
    )


def _make_three_branch_parallel(r: float, c: float, l: float) -> fasteis.Circuit:
    # R, C, L in parallel
    return fasteis.Parallel(
        [
            fasteis.R(r),
            fasteis.C(c),
            fasteis.L(l),
        ]
    )


@pytest.mark.parametrize(
    ("truth", "guess"),
    [
        (
            _make_randles(20.0, 150.0, 20e-6, 60.0),
            _make_randles(35.0, 90.0, 8e-6, 90.0),
        ),
        (
            _make_three_branch_parallel(100.0, 1e-6, 1e-3),
            _make_three_branch_parallel(60.0, 4e-6, 4e-3),
        ),
    ],
    ids=["randles_cell", "three_branch_parallel"],
)
def test_fit_recovers_impedance_for_composed_topologies(
    truth: fasteis.Circuit, guess: fasteis.Circuit
) -> None:
    """Fit synthetic circuits."""
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z))

    assert result.success
    got = np.asarray(result.circuit.impedance(FREQS), dtype=np.complex128)
    np.testing.assert_allclose(got, z, rtol=1e-4, atol=1e-8)


def test_fit_weight_modulus_vs_unit_differ() -> None:
    """Fit synthetic circuits with noise across large freq range.

    Weighting by unit vs modulus results in different fits.
    """
    truth = fasteis.Series([fasteis.R(1.0), fasteis.Cpe(1e-2, 0.7)])
    guess = fasteis.Series([fasteis.R(3.0), fasteis.Cpe(5e-3, 0.5)])
    freqs = list(np.logspace(0, 6, 40))
    z = _synthetic(truth, freqs)
    rng = np.random.default_rng(0)
    z = z * (1.0 + 0.01 * rng.standard_normal(z.shape))

    modulus = guess.fit(freqs, list(z), weight="modulus")
    unit = guess.fit(freqs, list(z), weight="unit")

    assert modulus.params != pytest.approx(unit.params)


def test_fit_reports_success_and_finite_stderr() -> None:
    """Fit reports success, has sensible stderr."""
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
    """Error on mismatched f and Z lengths."""
    circuit = fasteis.Series([fasteis.R(100.0)])
    with pytest.raises(ValueError):
        circuit.fit(FREQS, [complex(1.0, 0.0)])


def test_fit_rejects_unknown_weight() -> None:
    """Error on bad inputs."""
    circuit = fasteis.Series([fasteis.R(100.0)])
    z = _synthetic(circuit)
    with pytest.raises(ValueError):
        circuit.fit(FREQS, list(z), weight="bogus")


def test_fit_fixed_dict_holds_value_and_recovers_the_rest() -> None:
    """A dict holds its parameter at the given value, the rest are fitted."""
    truth = _make_randles(20.0, 150.0, 20e-6, 60.0)
    guess = _make_randles(25.0, 120.0, 1.5e-5, 70.0)
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z), fixed={"R0.r": 20.0})

    assert result.success
    assert result.params["R0.r"] == 20.0
    assert result.circuit.param_values()[0] == 20.0
    for name, expected in zip(truth.param_names(), truth.param_values(), strict=True):
        assert result.params[name] == pytest.approx(expected, rel=1e-4)
    assert result.stderr is not None
    assert set(result.stderr) == set(truth.param_names())
    assert np.isnan(result.stderr["R0.r"])
    assert all(np.isfinite(e) for name, e in result.stderr.items() if name != "R0.r")


def test_fit_fixed_list_holds_current_value() -> None:
    """A list holds its parameter at the circuit's value."""
    truth = _make_randles(20.0, 150.0, 20e-6, 60.0)
    guess = _make_randles(30.0, 120.0, 1.5e-5, 70.0)
    z = _synthetic(truth)

    result = guess.fit(FREQS, list(z), fixed=["R0.r"])

    assert result.params["R0.r"] == 30.0
    assert result.params != pytest.approx(guess.fit(FREQS, list(z)).params)


def test_fit_fixed_overrides_ml_guess() -> None:
    """The guess starts the free parameters, the fixed value is kept."""
    truth = fasteis.Circuit("sei_randles").with_named_values(
        {
            "R0.r": 100.0,
            "R1.r": 50.0,
            "CPE1.q": 1e-5,
            "CPE1.alpha": 0.9,
            "R2.r": 200.0,
            "W2.aw": 30.0,
            "CPE2.q": 1e-3,
            "CPE2.alpha": 0.8,
        }
    )
    z = _synthetic(truth)

    result = fasteis.Circuit("sei_randles").fit(FREQS, list(z), fixed={"R0.r": 100.0})

    assert result.params["R0.r"] == 100.0
    got = np.asarray(result.circuit.impedance(FREQS), dtype=np.complex128)
    np.testing.assert_allclose(got, z, rtol=1e-3)


def test_fit_fixed_list_rejects_circuit_without_values() -> None:
    """A list would hold a placeholder, so it needs a circuit with values."""
    circuit = fasteis.Circuit("R0-(R1,C1)")
    z = _synthetic(
        fasteis.Series([fasteis.R(10.0), fasteis.Parallel([fasteis.R(5.0), fasteis.C(1e-3)])])
    )
    with pytest.raises(ValueError, match="dict"):
        circuit.fit(FREQS, list(z), fixed=["R0.r"])


def test_fit_fixed_rejects_unknown_name() -> None:
    """Unknown names get a suggestion."""
    circuit = _make_randles(20.0, 150.0, 20e-6, 60.0)
    z = _synthetic(circuit)
    with pytest.raises(ValueError, match='did you mean "R0.r"'):
        circuit.fit(FREQS, list(z), fixed={"R0.R": 20.0})


def test_fit_fixed_rejects_every_parameter_fixed() -> None:
    """Nothing left to fit."""
    circuit = _make_randles(20.0, 150.0, 20e-6, 60.0)
    z = _synthetic(circuit)
    with pytest.raises(ValueError, match="no free parameters"):
        circuit.fit(FREQS, list(z), fixed=circuit.param_names())
