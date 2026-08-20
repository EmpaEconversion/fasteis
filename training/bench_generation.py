# Copyright © 2026, Empa.
"""Measures how fast synthetic Randles spectra can be generated from Python.

Decides whether `fasteis` can feed a training loop directly or whether
generation needs to move to the batched torch implementation.
"""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis

N_FREQS = 64
BATCH = 512
N_REPEATS = 20

RANDLES = "R0-(CPE1,R1-W1)"


def _sample_params(rng: np.random.Generator, n: int) -> np.ndarray:
    """Crude log-uniform draws. Only the timing matters here."""
    return np.column_stack(
        [
            10 ** rng.uniform(-1, 1, n),  # R0.r
            10 ** rng.uniform(-5, -2, n),  # CPE1.q
            rng.uniform(0.6, 1.0, n),  # CPE1.alpha
            10 ** rng.uniform(0, 2, n),  # R1.r
            10 ** rng.uniform(0, 2, n),  # W1.aw
        ]
    )


def bench_shared_circuit(params: np.ndarray, freqs: list[float]) -> float:
    """One parsed Circuit reused, rebuilt per sample via with_values()."""
    circuit = fasteis.Circuit(RANDLES)

    def run() -> None:
        for row in params:
            circuit.with_values(list(row)).impedance(freqs)

    return timeit.timeit(run, number=N_REPEATS) / N_REPEATS * 1e3


def bench_reparsed(params: np.ndarray, freqs: list[float]) -> float:
    """Circuit string re-parsed every sample, to price the parse itself."""

    def run() -> None:
        for row in params:
            fasteis.Circuit(RANDLES).with_values(list(row)).impedance(freqs)

    return timeit.timeit(run, number=N_REPEATS) / N_REPEATS * 1e3


def bench_varying_freqs(params: np.ndarray, rng: np.random.Generator) -> float:
    """Realistic case: every sample gets its own frequency sweep."""
    circuit = fasteis.Circuit(RANDLES)
    sweeps = [
        list(np.logspace(lo, lo + w, N_FREQS))
        for lo, w in zip(rng.uniform(-2, 2, len(params)), rng.uniform(4, 8, len(params)))
    ]

    def run() -> None:
        for row, freqs in zip(params, sweeps):
            circuit.with_values(list(row)).impedance(freqs)

    return timeit.timeit(run, number=N_REPEATS) / N_REPEATS * 1e3


def main() -> None:
    """Run generation benchmarks."""
    rng = np.random.default_rng(0)
    params = _sample_params(rng, BATCH)
    freqs = list(np.logspace(-2, 6, N_FREQS))

    rows = [
        ("shared circuit, fixed sweep", bench_shared_circuit(params, freqs)),
        ("reparsed per sample", bench_reparsed(params, freqs)),
        ("shared circuit, varying sweep", bench_varying_freqs(params, rng)),
    ]

    print(f"batch={BATCH}, n_freqs={N_FREQS}, repeats={N_REPEATS}")
    print(f"{'variant':<32} {'ms/batch':>10} {'us/sample':>12} {'batches/s':>10}")
    for name, ms in rows:
        print(f"{name:<32} {ms:>10.2f} {ms / BATCH * 1e3:>12.1f} {1e3 / ms:>10.1f}")


if __name__ == "__main__":
    main()
