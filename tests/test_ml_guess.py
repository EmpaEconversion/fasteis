"""Tests for Circuit.guess(), element ordering, and when fit() guesses by default."""

from __future__ import annotations

import warnings

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


UNTRAINED = "R0-(R1,C1)-(R2,C2)-(R3,C3)"


def test_circuit_without_values_guesses() -> None:
    """When no initial values are given, fit() guesses by default."""
    circuit = fasteis.Circuit("R0-(R1,C1)")

    automatic = circuit.fit(FREQS, Z)
    requested = circuit.fit(FREQS, Z, guess_init=True)

    assert automatic.success
    assert automatic.iterations == requested.iterations
    assert automatic.chi_square == pytest.approx(requested.chi_square, rel=1e-12)
    for name, expected in TRUTH.items():
        assert automatic.params[name] == pytest.approx(expected, rel=1e-3)


def test_circuit_with_values_does_not_guess() -> None:
    """When values are supplied, fit() does not do an ML guess."""
    circuit = fasteis.Circuit("R0-(R1,C1)").with_named_values(
        {"R0.r": 8.0, "R1.r": 25.0, "C1.c": 5e-5}
    )

    automatic = circuit.fit(FREQS, Z)
    suppressed = circuit.fit(FREQS, Z, guess_init=False)
    guessed = circuit.fit(FREQS, Z, guess_init=True)

    assert automatic.iterations == suppressed.iterations
    assert automatic.iterations != guessed.iterations


def test_circuit_from_result_does_not_guess() -> None:
    """Values from a fit result are treated as given values, so do not guess."""
    first = fasteis.Circuit("R0-(R1,C1)").fit(FREQS, Z)

    refit = first.circuit.fit(FREQS, Z)

    assert refit.iterations == first.circuit.fit(FREQS, Z, guess_init=False).iterations


def test_untrained_circuit_warns() -> None:
    """An untrained circuit warns that it cannot guess, uses default values."""
    circuit = fasteis.Circuit(UNTRAINED)

    with pytest.warns(UserWarning, match="no ML model for this circuit") as record:
        result = circuit.fit(FREQS, Z)

    assert "guess_init=False" in str(record[0].message)
    fallback = circuit.fit(FREQS, Z, guess_init=False)
    assert result.iterations == fallback.iterations


def test_untrained_circuit_with_values_does_not_warn() -> None:
    """An untrained circuit with supplied initial parameters does not warn."""
    names = fasteis.Circuit(UNTRAINED).param_names()
    circuit = fasteis.Circuit(UNTRAINED).with_values([1.0] * len(names))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        circuit.fit(FREQS, Z)


def test_untrained_circuit_explicit_guess_raises() -> None:
    """Giving an untrained circuit `guess_init=True` raises."""
    with pytest.raises(ValueError, match="No training data on this circuit"):
        fasteis.Circuit(UNTRAINED).fit(FREQS, Z, guess_init=True)
