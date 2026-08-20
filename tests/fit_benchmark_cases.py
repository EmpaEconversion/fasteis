# Copyright © 2026, Empa.
"""Benchmark for fitting against test data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray

DATA_DIR = Path(__file__).parent / "data"

# Skip datasets with too few points
MIN_POINTS = 10

DEFAULT_INDEX_COLUMN = "EIS Index / 1"


class FitBenchmarkCase:
    """A dataset + circuit topology + initial guess to fit."""

    def __init__(
        self,
        label: str,
        df: pl.DataFrame,
        circuit_string: str,
        eis_initial_guess: dict[str, float],
        ipy_initial_guess: list[float] | None = None,
        freq_column: str | None = None,
        real_column: str | None = None,
        imag_column: str | None = None,
        eis_index: int | None = None,
    ) -> None:
        self.label = label
        self.df = df
        self.freq_column = freq_column or "Frequency / Hz"
        self.real_column = real_column or "Real Impedance / ohm"
        self.imag_column = imag_column or "Imaginary Impedance / ohm"
        self.eis_index = eis_index
        self.circuit_string = circuit_string
        self.eis_initial_guess = eis_initial_guess
        self.ipy_initial_guess = ipy_initial_guess or list(eis_initial_guess.values())

    def load_data(self) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Load (frequencies, impedances) from the parquet file.

        If `eis_index` is set, filters to just that sweep's rows -- the file
        may hold many others (see module docstring).
        """
        freqs = self.df[self.freq_column].to_numpy()
        z = self.df[self.real_column].to_numpy() + 1j * self.df[self.imag_column].to_numpy()
        return freqs, z.astype(np.complex128)


def _cases_for_file(
    file: Path,
    circuit_string: str,
    eis_initial_guess: dict[str, float],
    ipy_initial_guess: list[float] | None = None,
) -> list[FitBenchmarkCase]:
    """One `FitBenchmarkCase` per `EIS Index / 1` group in `file`."""
    df = pl.read_parquet(file)
    if "EIS Index / 1" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("EIS Index / 1"))
    return [
        FitBenchmarkCase(
            label=f"{file.stem}_{eis_index[0]}",
            df=gdf,
            circuit_string=circuit_string,
            eis_initial_guess=eis_initial_guess,
            ipy_initial_guess=ipy_initial_guess,
        )
        for eis_index, gdf in df.group_by("EIS Index / 1")
        if len(gdf) > MIN_POINTS
    ]


FIT_BENCHMARK_CASES: list[FitBenchmarkCase] = []

for _file in sorted(DATA_DIR.glob("reda*.parquet")):
    FIT_BENCHMARK_CASES.extend(
        _cases_for_file(
            _file,
            circuit_string="L0-R0-p(R1,CPE1)-p(R2,CPE2)",
            eis_initial_guess={
                "L0.l": 1.00e-8,
                "R0.r": 0.2,
                "R1.r": 0.02,
                "CPE1.q": 1.00,
                "CPE1.alpha": 1.00,
                "R2.r": 0.1,
                "CPE2.q": 1.00,
                "CPE2.alpha": 1.00,
            },
        )
    )

for _file in sorted(DATA_DIR.glob("kiye*.parquet")):
    FIT_BENCHMARK_CASES.extend(
        _cases_for_file(
            _file,
            circuit_string="R0-p(R1,CPE1)-p(R2,CPE2)-Wo0",
            eis_initial_guess={
                "R0.r": 5,
                "R1.r": 10,
                "CPE1.q": 5e-4,
                "CPE1.alpha": 0.5,
                "R2.r": 30,
                "CPE2.q": 5e-3,
                "CPE2.alpha": 0.8,
                "Wo0.z0": 1e-3,
                "Wo0.tau": 1e-4,
            },
        )
    )
