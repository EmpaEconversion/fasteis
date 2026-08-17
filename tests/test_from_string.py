"""Tests for the eis.Circuit(str) constructor and the associated value-setting API."""

from __future__ import annotations

import numpy as np
import pytest

import fasteis

FREQS: list[float] = list(np.logspace(-1, 6, 60))


def test_from_string_flat_series_param_names() -> None:
    circuit = fasteis.Circuit("R0-C1")
    assert circuit.param_names() == ["R0.r", "C1.c"]


def test_from_string_series_ending_in_parallel_param_names() -> None:
    circuit = fasteis.Circuit("R0-p(R1,Cpe1)")
    assert circuit.param_names() == ["R0.r", "R1.r", "Cpe1.q", "Cpe1.alpha"]


def test_from_string_series_inside_parallel_branches() -> None:
    circuit = fasteis.Circuit("R0-p(R1-C1,R2-Cpe2)")
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
        fasteis.Circuit("Q0")


def test_bare_parens_parallel_matches_p_parens() -> None:
    bare = fasteis.Circuit("R0-(R1,Cpe1)").with_named_values(
        {"R0.r": 1.0, "R1.r": 2.0, "Cpe1.q": 3e-4, "Cpe1.alpha": 0.8}
    )
    with_p = fasteis.Circuit("R0-p(R1,Cpe1)").with_named_values(
        {"R0.r": 1.0, "R1.r": 2.0, "Cpe1.q": 3e-4, "Cpe1.alpha": 0.8}
    )
    np.testing.assert_allclose(
        np.asarray(bare.impedance(FREQS), dtype=np.complex128),
        np.asarray(with_p.impedance(FREQS), dtype=np.complex128),
    )


def test_parse_error_lists_syntax_and_available_elements() -> None:
    with pytest.raises(ValueError) as excinfo:
        fasteis.Circuit("R")
    message = str(excinfo.value)
    assert "series" in message
    assert "parallel" in message
    # Every element code is listed so the user can discover valid syntax
    # without consulting external docs.
    for code in [
        "R",
        "C",
        "L",
        "La",
        "CPE",
        "W",
        "Wo",
        "Ws",
        "G",
        "Gs",
        "K",
        "Zarc",
        "TLMQ",
        "T",
    ]:
        assert code in message


def test_from_string_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError):
        fasteis.Circuit("R0-R0")


def test_with_values_sets_params_positionally() -> None:
    circuit = fasteis.Circuit("R0-p(R1,Cpe1)").with_values([100.0, 200.0, 3e-4, 0.8])
    assert circuit.param_names() == ["R0.r", "R1.r", "Cpe1.q", "Cpe1.alpha"]

    z = np.asarray(circuit.impedance(FREQS), dtype=np.complex128)
    expected = fasteis.Series(
        [
            fasteis.R(100.0),
            fasteis.Parallel([fasteis.R(200.0), fasteis.Cpe(3e-4, 0.8)]),
        ]
    )
    np.testing.assert_allclose(z, np.asarray(expected.impedance(FREQS), dtype=np.complex128))


def test_with_values_rejects_wrong_length() -> None:
    circuit = fasteis.Circuit("R0-C1")
    with pytest.raises(ValueError):
        circuit.with_values([1.0])


def test_with_named_values_sets_params_by_label() -> None:
    circuit = fasteis.Circuit("R0-p(R1,Cpe1)").with_named_values(
        {"R0.r": 100.0, "R1.r": 200.0, "Cpe1.q": 3e-4, "Cpe1.alpha": 0.8}
    )
    z = np.asarray(circuit.impedance(FREQS), dtype=np.complex128)
    expected = fasteis.Series(
        [
            fasteis.R(100.0),
            fasteis.Parallel([fasteis.R(200.0), fasteis.Cpe(3e-4, 0.8)]),
        ]
    )
    np.testing.assert_allclose(z, np.asarray(expected.impedance(FREQS), dtype=np.complex128))


def test_with_named_values_rejects_missing_name() -> None:
    circuit = fasteis.Circuit("R0-C1")
    with pytest.raises(ValueError, match="missing parameter"):
        circuit.with_named_values({"R0.r": 100.0})


def test_with_named_values_rejects_unknown_name() -> None:
    circuit = fasteis.Circuit("R0-C1")
    with pytest.raises(ValueError) as excinfo:
        circuit.with_named_values({"R0.r": 100.0, "C1.c": 1e-6, "bogus": 1.0})
    message = str(excinfo.value)
    assert "bogus" in message
    # All parameter names listed in error message
    for name in circuit.param_names():
        assert name in message


def test_with_named_values_suggests_close_typo() -> None:
    circuit = fasteis.Circuit("R0-p(R1,Cpe1)")
    with pytest.raises(ValueError, match='did you mean "Cpe1.alpha"'):
        circuit.with_named_values({"R0.r": 1.0, "R1.r": 2.0, "Cpe1.q": 3e-4, "Cpe1.alph": 0.8})


def test_param_units_matches_param_names_length() -> None:
    circuit = fasteis.Circuit("R0-p(R1,Cpe1)")
    assert circuit.param_units() == ["ohm", "ohm", "ohm^-1*s^alpha", "-"]


def test_repr_lists_every_param_name_value_unit_and_bound() -> None:
    circuit = fasteis.Circuit("R0-Cpe1").with_named_values(
        {"R0.r": 100.0, "Cpe1.q": 3e-4, "Cpe1.alpha": 0.8}
    )
    text = repr(circuit)
    for name in circuit.param_names():
        assert name in text
    assert "ohm" in text
    assert "ohm^-1*s^alpha" in text
    assert "100" in text


def test_from_string_circuit_can_be_fit() -> None:
    truth = fasteis.Series(
        [
            fasteis.R(20.0),
            fasteis.Parallel([fasteis.R(200.0), fasteis.W(50.0)]),
            fasteis.C(1e-5),
        ]
    )
    z = np.asarray(truth.impedance(FREQS), dtype=np.complex128)

    guess = fasteis.Circuit("R0-p(R1,W1)-C1").with_values([25.0, 150.0, 65.0, 1.3e-5])
    result = guess.fit(FREQS, list(z))

    assert result.success
    got = np.asarray(result.circuit.impedance(FREQS), dtype=np.complex128)
    np.testing.assert_allclose(got, z, rtol=1e-4, atol=1e-8)
