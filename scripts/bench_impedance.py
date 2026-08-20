# Copyright © 2026, Empa.
"""Benchmarks against impedance.py.

Benchmark across every circuit element and every composed circuit topology used
in the correctness regression suite (see tests/circuit_cases.py).
"""

from __future__ import annotations

import sys
import timeit
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import TYPE_CHECKING

from impedance.models.circuits import elements as ipy

import fasteis
from tests.circuit_cases import COMPOSITION_CASES, ELEMENT_CASES, FreqArray

if TYPE_CHECKING:
    from collections.abc import Callable

N_FREQS = 10_000
N_REPEATS = 50

BenchRow = tuple[str, float, float]


def _dense_freqs(freqs: FreqArray) -> list[float]:
    """Resample frequency array up to N_FREQS points across the same log10 range."""
    lo, hi = np.log10(freqs.min()), np.log10(freqs.max())
    return list(np.logspace(lo, hi, N_FREQS))


def _ms_per_call(fn: Callable[[], object]) -> float:
    with warnings.catch_warnings():
        # some impedance.py formulas warn on overflow
        warnings.simplefilter("ignore", RuntimeWarning)
        total = timeit.timeit(fn, number=N_REPEATS)
    return total / N_REPEATS * 1e3


def bench_elements() -> list[BenchRow]:
    """Benchmark single elements."""
    rows: list[BenchRow] = []
    for name, variations in ELEMENT_CASES.items():
        params, freqs = variations[0]  # magnitude doesn't affect timing
        freqs_list = _dense_freqs(freqs)

        circuit = getattr(fasteis.Circuit, name)(*params)
        eis_ms = _ms_per_call(
            lambda circuit=circuit, freqs_list=freqs_list: circuit.impedance(freqs_list)
        )
        ipy_ms = _ms_per_call(
            lambda name=name, params=params, freqs_list=freqs_list: ipy.circuit_elements[name](
                list(params), freqs_list
            )
        )
        rows.append((name, eis_ms, ipy_ms))
    return rows


def bench_compositions() -> list[BenchRow]:
    """Benchmark multi-element circuits."""
    rows: list[BenchRow] = []
    for case in COMPOSITION_CASES:
        freqs_list = _dense_freqs(case.freqs)
        eis_ms = _ms_per_call(
            lambda case=case, freqs_list=freqs_list: case.eis_circuit.impedance(freqs_list)
        )
        ipy_ms = _ms_per_call(lambda case=case, freqs_list=freqs_list: case.ipy_result(freqs_list))
        rows.append((case.label, eis_ms, ipy_ms))
    return rows


def _print_table(title: str, rows: list[BenchRow]) -> None:
    print(f"\n{title} ({N_FREQS} frequencies/call, {N_REPEATS} repeats)")
    print(f"{'name':<28} {'fasteis (ms)':>10} {'impedance.py (ms)':>20} {'speedup':>10}")
    for name, eis_ms, ipy_ms in rows:
        print(f"{name:<28} {eis_ms:>10.4f} {ipy_ms:>20.4f} {ipy_ms / eis_ms:>9.1f}x")


def main() -> None:
    """Run benchmarks."""
    _print_table("Elements", bench_elements())
    _print_table("Compositions", bench_compositions())


if __name__ == "__main__":
    main()
