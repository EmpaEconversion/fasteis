"""Tests for eis.Circuit.from_string() and the associated value-setting API."""

from __future__ import annotations

import numpy as np
import pytest

import eis

FREQS: list[float] = list(np.logspace(-1, 6, 60))


def test_from_string_flat_series_param_names() -> None:
    circuit = eis.Circuit.from_string("R0-C1")
    assert circuit.param_names() == ["R0.r", "C1.c"]


def test_from_string_series_ending_in_parallel_param_names() -> None:
    circuit = eis.Circuit.from_string("R0-p(R1,Cpe1)")
    assert circuit.param_names() == ["R0.r", "R1.r", "Cpe1.q", "Cpe1.alpha"]


def test_from_string_series_inside_parallel_branches() -> None:
    circuit = eis.Circuit.from_string("R0-p(R1-C1,R2-Cpe2)")
    assert circuit.param_names() == [
        "R0.r",
        "R1.r",
        "C1.c",
        "R2.r",
        "Cpe2.q",
        "Cpe2.alpha",
    ]


def test_from_string_rejects_unknown_code() -> None:
    with pytest.raises(ValueError):
        eis.Circuit.from_string("Q0")


def test_from_string_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError):
        eis.Circuit.from_string("R0-R0")


def test_with_values_sets_params_positionally() -> None:
    circuit = eis.Circuit.from_string("R0-p(R1,Cpe1)").with_values(
        [100.0, 200.0, 3e-4, 0.8]
    )
    assert circuit.param_names() == ["R0.r", "R1.r", "Cpe1.q", "Cpe1.alpha"]

    z = np.asarray(circuit.impedance(FREQS), dtype=np.complex128)
    expected = eis.Circuit.series(
        [
            eis.Circuit.R(100.0),
            eis.Circuit.parallel([eis.Circuit.R(200.0), eis.Circuit.CPE(3e-4, 0.8)]),
        ]
    )
    np.testing.assert_allclose(
        z, np.asarray(expected.impedance(FREQS), dtype=np.complex128)
    )


def test_with_values_rejects_wrong_length() -> None:
    circuit = eis.Circuit.from_string("R0-C1")
    with pytest.raises(ValueError):
        circuit.with_values([1.0])


def test_with_named_values_sets_params_by_label() -> None:
    circuit = eis.Circuit.from_string("R0-p(R1,Cpe1)").with_named_values(
        {"R0.r": 100.0, "R1.r": 200.0, "Cpe1.q": 3e-4, "Cpe1.alpha": 0.8}
    )
    z = np.asarray(circuit.impedance(FREQS), dtype=np.complex128)
    expected = eis.Circuit.series(
        [
            eis.Circuit.R(100.0),
            eis.Circuit.parallel([eis.Circuit.R(200.0), eis.Circuit.CPE(3e-4, 0.8)]),
        ]
    )
    np.testing.assert_allclose(
        z, np.asarray(expected.impedance(FREQS), dtype=np.complex128)
    )


def test_with_named_values_rejects_missing_name() -> None:
    circuit = eis.Circuit.from_string("R0-C1")
    with pytest.raises(ValueError):
        circuit.with_named_values({"R0.r": 100.0})


def test_with_named_values_rejects_unknown_name() -> None:
    circuit = eis.Circuit.from_string("R0-C1")
    with pytest.raises(ValueError):
        circuit.with_named_values({"R0.r": 100.0, "C1.c": 1e-6, "bogus": 1.0})


def test_from_string_circuit_can_be_fit() -> None:
    truth = eis.Circuit.series(
        [
            eis.Circuit.R(20.0),
            eis.Circuit.parallel([eis.Circuit.R(200.0), eis.Circuit.W(50.0)]),
            eis.Circuit.C(1e-5),
        ]
    )
    z = np.asarray(truth.impedance(FREQS), dtype=np.complex128)

    guess = eis.Circuit.from_string("R0-p(R1,W1)-C1").with_values(
        [25.0, 150.0, 65.0, 1.3e-5]
    )
    result = guess.fit(FREQS, list(z))

    assert result.success
    got = np.asarray(result.circuit.impedance(FREQS), dtype=np.complex128)
    np.testing.assert_allclose(got, z, rtol=1e-4, atol=1e-8)
