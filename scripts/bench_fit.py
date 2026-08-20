# Copyright © 2026, Empa.
"""Benchmarks Circuit.fit() against impedance.py's circuit_fit.

By default runs a small sample of cases and skips the slow global/derivative-free
methods (and impedance.py's own basinhopping reference), since running those across
all of FIT_BENCHMARK_CASES can take hours. Pass --include-global and/or --all to
opt into the full, slow sweep.

Run: uv run python scripts/bench_fit.py [--cases N] [--all] [--include-global]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impedance.models.circuits import CustomCircuit

import fasteis
from tests.fit_benchmark_cases import FIT_BENCHMARK_CASES, FitBenchmarkCase

if TYPE_CHECKING:
    from collections.abc import Callable

# Always run: fast, bounded by design (no fixed-iteration global search).
FAST_EIS_CONFIGS: list[tuple[str, str, dict[str, object], int]] = [
    ("", "levenberg_marquardt", {}, 20),
]

# Only run with --include-global -- (label suffix, method, kwargs, repeats for
# timing). Add a new tuple here to compare a different configuration
# `seed` is fixed so repeated runs are reproducible
GLOBAL_EIS_CONFIGS: list[tuple[str, str, dict[str, object], int]] = [
    (
        "(60,300)",
        "particle_swarm",
        {"num_particles": 60, "generations": 300, "seed": 0},
        3,
    ),
    (
        "(200,1000)",
        "particle_swarm",
        {"num_particles": 200, "generations": 1000, "seed": 0},
        3,
    ),
    ("", "nelder_mead", {"nelder_mead_iterations": 2000}, 10),
    ("", "differential_evolution", {"de_evaluations": 20_000}, 3),
    ("", "simulated_annealing", {"sa_iterations": 5000, "seed": 0}, 3),
    ("", "basin_hopping", {"basin_hopping_hops": 20, "seed": 0}, 3),
]

# scipy basinhopping has no cap by default and can run ~100s/case
# Cap so it fails faster
IMPEDANCEPY_BASINHOPPING_NITER = 2

DEFAULT_CASE_COUNT = 10


def _weighted_cost(z_model: np.ndarray, z_measured: np.ndarray) -> float:
    """0.5 * sum(weighted residuals^2), modulus-weighted -- matches fit()'s own cost."""
    w = np.abs(z_measured)
    r_re = (z_model.real - z_measured.real) / w
    r_im = (z_model.imag - z_measured.imag) / w
    return 0.5 * float(np.sum(r_re**2) + np.sum(r_im**2))


def _max_rel_err(z_model: np.ndarray, z_measured: np.ndarray) -> float:
    return float(np.max(np.abs(z_model - z_measured) / np.abs(z_measured)))


def _rust_trf_fit(
    circuit: fasteis.Circuit, freqs_list: list[float], z_list: list[complex]
) -> np.ndarray:
    """Test with scipy.optimize.least_squares(method="trf") (same as impedance.py)."""
    bounds = circuit.param_bounds()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.array(circuit.param_values()), lo, hi)

    def fun(p: np.ndarray) -> np.ndarray:
        return np.asarray(circuit.residuals(p.tolist(), freqs_list, z_list, "modulus"))

    def jac(p: np.ndarray) -> np.ndarray:
        return np.asarray(circuit.jacobian(p.tolist(), freqs_list, z_list, "modulus"))

    result = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf")
    return result.x


def _median_ms(fn: Callable[[], object], number: int) -> float:
    fn()  # warm up -- first call pays one-time allocator/JIT costs
    times = []
    for _ in range(number):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


def _print_row(method: str, time_ms: float, cost: float, err: float) -> None:
    print(f"{method:<40} {time_ms:>12.3f} {cost:>14.6f} {err:>12.4f}")


def bench_case(case: FitBenchmarkCase, *, include_global: bool) -> None:
    freqs, z = case.load_data()
    freqs_list = list(freqs)

    print(f"\n=== {case.label} ({len(freqs)} points) ===")
    print(f"{'method':<40} {'time (ms)':>12} {'cost':>14} {'max rel err':>12}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        def ipy_curve_fit() -> CustomCircuit:
            c = CustomCircuit(case.circuit_string, initial_guess=case.ipy_initial_guess)
            c.fit(freqs, z, weight_by_modulus=True)
            return c

        ipy_ms = _median_ms(ipy_curve_fit, number=5)
        z_ipy = np.asarray(ipy_curve_fit().predict(freqs), dtype=np.complex128)
        _print_row(
            "impedance.py (curve_fit)",
            ipy_ms,
            _weighted_cost(z_ipy, z),
            _max_rel_err(z_ipy, z),
        )

        if include_global:

            def ipy_basinhopping() -> CustomCircuit:
                c = CustomCircuit(case.circuit_string, initial_guess=case.ipy_initial_guess)
                c.fit(
                    freqs,
                    z,
                    global_opt=True,
                    seed=0,
                    niter=IMPEDANCEPY_BASINHOPPING_NITER,
                )
                return c

            ipy_bh_ms = _median_ms(ipy_basinhopping, number=1)
            z_ipy_bh = np.asarray(ipy_basinhopping().predict(freqs), dtype=np.complex128)
            _print_row(
                "impedance.py (basinhopping)",
                ipy_bh_ms,
                _weighted_cost(z_ipy_bh, z),
                _max_rel_err(z_ipy_bh, z),
            )

    circuit = fasteis.Circuit.from_string(case.circuit_string).with_named_values(
        case.eis_initial_guess
    )
    z_list = list(z)

    def rust_trf() -> np.ndarray:
        return _rust_trf_fit(circuit, freqs_list, z_list)

    trf_ms = _median_ms(rust_trf, number=5)
    x_trf = rust_trf()
    z_trf = np.asarray(circuit.with_values(list(x_trf)).impedance(freqs_list), dtype=np.complex128)
    _print_row(
        "rust-math + scipy TRF",
        trf_ms,
        _weighted_cost(z_trf, z),
        _max_rel_err(z_trf, z),
    )

    configs = FAST_EIS_CONFIGS + GLOBAL_EIS_CONFIGS if include_global else FAST_EIS_CONFIGS
    for suffix, method, kwargs, repeats in configs:

        def eis_fit(method: str = method, kwargs: dict[str, object] = kwargs) -> fasteis.FitResult:
            return circuit.fit(freqs_list, list(z), method=method, **kwargs)

        fit_ms = _median_ms(eis_fit, number=repeats)
        result = eis_fit()
        z_fit = np.asarray(result.circuit.impedance(freqs_list), dtype=np.complex128)
        _print_row(f"eis {method}{suffix}", fit_ms, result.cost, _max_rel_err(z_fit, z))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=int,
        default=DEFAULT_CASE_COUNT,
        help=f"number of cases to run (default {DEFAULT_CASE_COUNT}); ignored with --all",
    )
    parser.add_argument("--all", action="store_true", help="run every case in FIT_BENCHMARK_CASES")
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="also run the slow global/derivative-free methods and impedance.py's "
        "own basinhopping reference (skipped by default)",
    )
    return parser.parse_args()


def main() -> None:
    """Run bench_case over a slice of FIT_BENCHMARK_CASES per CLI args."""
    args = parse_args()
    cases = FIT_BENCHMARK_CASES if args.all else FIT_BENCHMARK_CASES[: args.cases]
    for case in cases:
        bench_case(case, include_global=args.include_global)


if __name__ == "__main__":
    main()
