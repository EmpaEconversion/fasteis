"""Tests for reading from battery data format dataframe."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import fasteis

PREF_COLUMNS = {
    "f": "Frequency / Hz",
    "real": "Real Impedance / ohm",
    "imag": "Imaginary Impedance / ohm",
    "abs": "Absolute Impedance / ohm",
    "phase": "Phase / deg",
}
MACHINE_COLUMNS = {
    "f": "frequency_hertz",
    "real": "real_impedance_ohm",
    "imag": "imaginary_impedance_ohm",
    "abs": "absolute_impedance_ohm",
    "phase": "phase_degree",
}

CIRCUIT = "R0-(R1,CPE1)"
TRUTH = {"R0.r": 12.0, "R1.r": 80.0, "CPE1.q": 4e-4, "CPE1.alpha": 0.86}


@pytest.fixture
def spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and impedances of a known circuit."""
    freqs = np.logspace(-1, 5, 60)
    circuit = fasteis.Circuit(CIRCUIT).with_named_values(TRUTH)
    return freqs, np.asarray(circuit.impedance(freqs), dtype=np.complex128)


def _cartesian(spectrum: tuple[np.ndarray, np.ndarray], names: dict[str, str]) -> dict[str, Any]:
    freqs, z = spectrum
    return {
        names["f"]: freqs,
        names["real"]: z.real,
        names["imag"]: z.imag,
    }


def _polar(spectrum: tuple[np.ndarray, np.ndarray], names: dict[str, str]) -> dict[str, Any]:
    freqs, z = spectrum
    return {
        names["f"]: freqs,
        names["abs"]: np.abs(z),
        names["phase"]: np.degrees(np.angle(z)),
    }


@pytest.mark.parametrize("frame", [pl.DataFrame, pd.DataFrame, dict])
@pytest.mark.parametrize("names", [PREF_COLUMNS, MACHINE_COLUMNS], ids=["pref", "machine"])
@pytest.mark.parametrize("columns", [_cartesian, _polar], ids=["cartesian", "polar"])
def test_fit_from_dataframe_matches_two_arguments(
    spectrum: tuple[np.ndarray, np.ndarray],
    frame: Any,
    names: dict[str, str],
    columns: Any,
) -> None:
    """Test all combinations of dataframe type, column names, and column types work the same."""
    expected = fasteis.Circuit(CIRCUIT).fit(*spectrum)
    result = fasteis.Circuit(CIRCUIT).fit(frame(columns(spectrum, names)))
    assert result.success
    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-6)


def test_cartesian_columns_win_over_polar(spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    """If cartesian and polar cols are present, cartesian is used."""
    columns = _cartesian(spectrum, PREF_COLUMNS)
    columns[PREF_COLUMNS["abs"]] = np.ones(len(columns[PREF_COLUMNS["real"]]))
    columns[PREF_COLUMNS["phase"]] = np.zeros(len(columns[PREF_COLUMNS["real"]]))
    expected = fasteis.Circuit(CIRCUIT).fit(*spectrum)
    result = fasteis.Circuit(CIRCUIT).fit(pl.DataFrame(columns))
    assert result.success
    for name, value in expected.params.items():
        assert result.params[name] == pytest.approx(value, rel=1e-6)


def test_guess_and_residuals_accept_a_dataframe(spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    """Guess, residuals, jacobian accept dataframes."""
    frame = pl.DataFrame(_cartesian(spectrum, PREF_COLUMNS))
    circuit = fasteis.Circuit(CIRCUIT)
    assert circuit.guess(frame) == circuit.guess(*spectrum)
    params = list(TRUTH.values())
    assert circuit.residuals(params, frame) == circuit.residuals(params, *spectrum)
    assert circuit.jacobian(params, frame) == circuit.jacobian(params, *spectrum)


def test_impedance_with_df_uses_freq(spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    """Passing a dataframe to impedance() just uses the frequency column."""
    _freqs, z = spectrum
    frame = pl.DataFrame(_cartesian(spectrum, PREF_COLUMNS))
    circuit = fasteis.Circuit(CIRCUIT).with_named_values(TRUTH)
    assert np.allclose(circuit.impedance(frame), z)


def test_real_data_with_dataframe() -> None:
    """Check that fits agree between a real bdf dataframe."""
    frame = pl.read_parquet("tests/data/reda_20338409.parquet").filter(pl.col("EIS Index / 1") == 1)
    freqs = frame[PREF_COLUMNS["f"]].to_numpy()
    z = frame[PREF_COLUMNS["real"]].to_numpy() + 1j * frame[PREF_COLUMNS["imag"]].to_numpy()
    circuit = fasteis.Circuit(CIRCUIT)
    assert circuit.fit(frame).params == circuit.fit(freqs, z).params


def _padded(columns: dict[str, Any], rows: int) -> dict[str, Any]:
    """Frame with `rows` zero-filled padding rows appended."""
    return {name: np.concatenate([values, np.zeros(rows)]) for name, values in columns.items()}


@pytest.mark.parametrize("frame", [pl.DataFrame, pd.DataFrame, dict])
@pytest.mark.parametrize("columns", [_cartesian, _polar], ids=["cartesian", "polar"])
def test_zero_frequency_padding_rows_are_ignored(
    spectrum: tuple[np.ndarray, np.ndarray],
    frame: Any,
    columns: Any,
) -> None:
    """Rows with zero frequency are dropped."""
    measured = columns(spectrum, PREF_COLUMNS)

    expected = fasteis.Circuit(CIRCUIT).fit(frame(measured))
    result = fasteis.Circuit(CIRCUIT).fit(frame(_padded(measured, 40)))

    assert result.success
    assert result.params == expected.params


def test_padding_is_dropped_for_guess_residuals_and_jacobian(
    spectrum: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every dataframe entry point sees the same measured points."""
    measured = _cartesian(spectrum, PREF_COLUMNS)
    clean = pl.DataFrame(measured)
    padded = pl.DataFrame(_padded(measured, 15))
    circuit = fasteis.Circuit(CIRCUIT)
    params = list(TRUTH.values())

    assert circuit.guess(padded) == circuit.guess(clean)
    assert circuit.residuals(params, padded) == circuit.residuals(params, clean)
    assert circuit.jacobian(params, padded) == circuit.jacobian(params, clean)


def test_all_padding_rows_error() -> None:
    """A frame with nothing measured says so rather than failing in the fit."""
    frame = pl.DataFrame(
        {
            PREF_COLUMNS["f"]: [0.0, 0.0],
            PREF_COLUMNS["real"]: [0.0, 0.0],
            PREF_COLUMNS["imag"]: [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="no measured points"):
        fasteis.Circuit(CIRCUIT).fit(frame)


def test_explicit_sequences_keep_every_point(spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    """Only the dataframe path drops padding, two sequences are taken as given."""
    freqs, z = spectrum
    padded_f = np.concatenate([freqs, np.zeros(3)])
    padded_z = np.concatenate([z, np.zeros(3, dtype=np.complex128)])

    residuals = fasteis.Circuit(CIRCUIT).residuals(list(TRUTH.values()), padded_f, padded_z)
    assert len(residuals) == 2 * len(padded_f)


def test_missing_frequency_error() -> None:
    """Error lists both accepted column names."""
    with pytest.raises(ValueError, match="Frequency / Hz.*frequency_hertz"):
        fasteis.Circuit(CIRCUIT).fit(pl.DataFrame({"a": [1.0], "b": [2.0]}))


def test_missing_impedance_error() -> None:
    """Error lists both accepted column names."""
    frame = pl.DataFrame({PREF_COLUMNS["f"]: [1.0, 2.0]})
    with pytest.raises(ValueError, match="no impedance columns"):
        fasteis.Circuit(CIRCUIT).fit(frame)


def test_non_numeric_column_error() -> None:
    """Non-numerical columns are specified."""
    frame = pl.DataFrame(
        {
            PREF_COLUMNS["f"]: ["a", "b"],
            PREF_COLUMNS["real"]: [1.0, 2.0],
            PREF_COLUMNS["imag"]: [-1.0, -2.0],
        }
    )
    with pytest.raises(ValueError, match='column "Frequency / Hz" is not numeric'):
        fasteis.Circuit(CIRCUIT).fit(frame)


def test_mismatched_column_error() -> None:
    """Different length columns errors."""
    columns = {
        PREF_COLUMNS["f"]: [1.0, 2.0, 3.0],
        PREF_COLUMNS["real"]: [1.0, 2.0],
        PREF_COLUMNS["imag"]: [-1.0, -2.0],
    }
    with pytest.raises(ValueError, match="rows"):
        fasteis.Circuit(CIRCUIT).fit(columns)
