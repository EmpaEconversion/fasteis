"""Tests for Circuit.guess() ensuring that element order does not matter."""

from __future__ import annotations

import numpy as np
import pytest

import fasteis

FREQS: list[float] = list(np.logspace(-1, 6, 60))


def _spectrum(circuit_str: str, **params: float) -> list[complex]:
    circuit = fasteis.Circuit(circuit_str).with_named_values(params)
    return list(np.asarray(circuit.impedance(FREQS), dtype=np.complex128))


TRUTH = {"R0.r": 5.0, "R1.r": 40.0, "C1.c": 2e-5}
Z = _spectrum("R0-(R1,C1)", **TRUTH)


@pytest.mark.parametrize(
    "circuit_str",
    ["R0-(R1,C1)", "R0-(C1,R1)", "(R1,C1)-R0", "(C1,R1)-R0"],
)
def test_guess_is_independent_of_the_order_elements_are_written_in(
    circuit_str: str,
) -> None:
    circuit = fasteis.Circuit(circuit_str)
    guess = dict(zip(circuit.param_names(), circuit.guess(FREQS, Z), strict=True))

    reference = fasteis.Circuit("R0-(R1,C1)")
    expected = dict(zip(reference.param_names(), reference.guess(FREQS, Z), strict=True))
    assert guess == pytest.approx(expected)


@pytest.mark.parametrize(
    "circuit_str",
    ["R0-(R1,C1)", "R0-(C1,R1)", "(R1,C1)-R0", "(C1,R1)-R0"],
)
def test_reordered_circuit_fits_from_its_guess(circuit_str: str) -> None:
    result = fasteis.Circuit(circuit_str).fit(FREQS, Z, guess_init=True)

    assert result.success
    for name, expected in TRUTH.items():
        assert result.params[name] == pytest.approx(expected, rel=1e-3)


def test_guess_reaches_labelled_and_reordered_parallel_branches() -> None:
    # W inside the branch pins the pairing, so the permutation is unambiguous
    circuit = fasteis.Circuit("(Cpe9,R9-W9)-R8")
    values = dict(zip(circuit.param_names(), circuit.guess(FREQS, Z), strict=True))

    assert set(values) == {"Cpe9.q", "Cpe9.alpha", "R9.r", "W9.aw", "R8.r"}
    assert 0.0 < values["Cpe9.alpha"] <= 1.0
    assert values["R8.r"] > 0.0


def test_guess_still_rejects_a_circuit_with_no_trained_model() -> None:
    with pytest.raises(ValueError, match="No training data on this circuit"):
        fasteis.Circuit("R0-(R1,C1)-(R2,C2)-(R3,C3)").guess(FREQS, Z)
