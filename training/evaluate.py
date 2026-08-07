"""Benchmark harness for fitting.

FLOOR is the number of fit iterations when starting from the correc parameters.
Objective is to minimize extra work beyond FLOOR with consistent convergence.

Fit first with plain LM. The library LM has some extra tricks (extra candidate
screening, restarts etc.) that would muddy the training.

`fit_library` is a secondary check with the "as shipped" but it is never used
to select checkpoints.

Magnitude parameters are fitted in log space, matching the coordinate transform
fit.rs uses and the space the model predicts in. alpha stays linear and
unconstrained.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits, priors

MAX_NFEV = 400
FTOL = 1e-10
XTOL = 1e-10

# Fit parameters
LIBRARY_FIT_KWARGS: dict[str, object] = {
    "weight": "modulus",
    "method": "levenberg_marquardt",
    "max_iterations": 200,
    "ftol": 1e-10,
    "xtol": 1e-10,
    "seed": 0,
}

# a fit counts as converged if it lands within this factor of FLOOR's cost
CONVERGENCE_TOL = 1.01

# a callable returning the parameters a fit should start from, given a spectrum
InitParams = Callable[[priors.Spectrum], NDArray[np.float64]]


@dataclass(frozen=True)
class Outcome:
    """Result of one fit. `evaluations` is the work measure being minimised."""

    success: bool
    evaluations: int
    cost: float
    params: NDArray[np.float64]
    seconds: float


def _to_free(params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Physical -> unconstrained coordinates (log for magnitudes, alpha linear)."""
    free = np.array(params, dtype=np.float64)
    free[list(circuits.LOG_PARAMS)] = np.log(
        np.clip(free[list(circuits.LOG_PARAMS)], 1e-300, None)
    )
    return free


def _to_params(free: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of `_to_free`."""
    params = np.array(free, dtype=np.float64)
    params[list(circuits.LOG_PARAMS)] = np.exp(
        np.clip(params[list(circuits.LOG_PARAMS)], -700, 700)
    )
    return params


def fit_plain_lm(
    spectrum: priors.Spectrum,
    init_params: NDArray[np.float64],
    circuit: fasteis.Circuit | None = None,
) -> Outcome:
    """Fit a single-start LM."""
    circuit = circuit if circuit is not None else fasteis.Circuit(circuits.CIRCUIT_STRING)
    freqs = list(spectrum.freqs)
    z = list(spectrum.z)

    def residuals(free: NDArray[np.float64]) -> NDArray[np.float64]:
        p = [float(v) for v in _to_params(free)]
        return np.asarray(circuit.residuals(p, freqs, z, "modulus"))

    def jacobian(free: NDArray[np.float64]) -> NDArray[np.float64]:
        params = _to_params(free)
        j = np.asarray(circuit.jacobian([float(v) for v in params], freqs, z, "modulus"))
        # chain rule for the log-coordinate parameters
        scale = np.ones(circuits.N_PARAMS)
        scale[list(circuits.LOG_PARAMS)] = params[list(circuits.LOG_PARAMS)]
        return j * scale

    t0 = time.perf_counter()
    result = least_squares(
        residuals,
        _to_free(init_params),
        jac=jacobian,
        method="lm",
        ftol=FTOL,
        xtol=XTOL,
        max_nfev=MAX_NFEV,
    )
    elapsed = time.perf_counter() - t0

    params = _to_params(result.x)
    params[circuits.ALPHA] = np.clip(params[circuits.ALPHA], 0.0, 1.0)
    return Outcome(
        success=bool(result.success),
        evaluations=int(result.nfev),
        cost=float(result.cost),
        params=params,
        seconds=elapsed,
    )


def fit_library(
    spectrum: priors.Spectrum,
    init_params: NDArray[np.float64],
    circuit: fasteis.Circuit | None = None,
) -> Outcome:
    """Fit the smarter LM included in Circuit.fit() path, which screens and restarts internally."""
    circuit = circuit if circuit is not None else fasteis.Circuit(circuits.CIRCUIT_STRING)
    start = circuit.with_values([float(v) for v in init_params])

    t0 = time.perf_counter()
    result = start.fit(list(spectrum.freqs), list(spectrum.z), **LIBRARY_FIT_KWARGS)
    elapsed = time.perf_counter() - t0

    return Outcome(
        success=result.success,
        evaluations=int(result.iterations),
        cost=result.cost,
        params=np.array([result.params[n] for n in circuits.PARAM_NAMES]),
        seconds=elapsed,
    )


def truth_init_params(spectrum: priors.Spectrum) -> NDArray[np.float64]:
    """FLOOR: start at the answer."""
    return spectrum.params


def default_init_params(_: priors.Spectrum) -> NDArray[np.float64]:
    """Default parameters with no starting values."""
    return np.array(fasteis.Circuit(circuits.CIRCUIT_STRING).param_values())


Fitter = Callable[[priors.Spectrum, NDArray[np.float64], fasteis.Circuit], Outcome]


def fit_all(
    spectra: Sequence[priors.Spectrum],
    init_params: InitParams,
    fitter: Fitter = fit_plain_lm,
) -> list[Outcome]:
    """Fit every spectrum from the parameters `init_params` produces."""
    circuit = fasteis.Circuit(circuits.CIRCUIT_STRING)
    return [fitter(s, init_params(s), circuit) for s in spectra]


@dataclass(frozen=True)
class Summary:
    """Aggregated metrics for one source of initial parameters."""

    name: str
    converged: float
    median_excess: float
    p90_excess: float
    p99_excess: float
    median_evaluations: float
    total_seconds: float


def summarise(
    name: str, outcomes: Sequence[Outcome], floor: Sequence[Outcome]
) -> Summary:
    """Score one source of initial parameters against FLOOR."""
    evaluations = np.array([o.evaluations for o in outcomes], dtype=float)
    floor_evaluations = np.array([o.evaluations for o in floor], dtype=float)

    cost = np.array([o.cost for o in outcomes])
    floor_cost = np.array([o.cost for o in floor])
    # tiny absolute floor so two effectively-perfect fits do not fail on rounding
    converged = cost <= CONVERGENCE_TOL * floor_cost + 1e-12

    # excess is only meaningful where the fit actually got there
    excess = (evaluations - floor_evaluations)[converged]
    if excess.size == 0:
        excess = np.array([np.nan])

    return Summary(
        name=name,
        converged=float(np.mean(converged)),
        median_excess=float(np.median(excess)),
        p90_excess=float(np.percentile(excess, 90)),
        p99_excess=float(np.percentile(excess, 99)),
        median_evaluations=float(np.median(evaluations)),
        total_seconds=float(sum(o.seconds for o in outcomes)),
    )


def print_table(summaries: Sequence[Summary]) -> None:
    """Print one row per source."""
    header = (
        f"{'source':<18} {'converged':>11} {'excess: med':>12} {'p90':>7} "
        f"{'p99':>7} {'med evals':>10} {'seconds':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.name:<18} {100 * s.converged:>9.2f}% {s.median_excess:>12.1f} "
            f"{s.p90_excess:>7.1f} {s.p99_excess:>7.1f} {s.median_evaluations:>10.1f} "
            f"{s.total_seconds:>9.2f}"
        )


def validation_set(n: int) -> list[priors.Spectrum]:
    """Spectra for choosing a checkpoint during training."""
    return priors.sample_many(priors.split_rng(priors.VALIDATION), n)


def benchmark_set(n: int) -> list[priors.Spectrum]:
    """Spectra for the final numbers. Disjoint from training and validation."""
    return priors.sample_many(priors.split_rng(priors.BENCHMARK), n)


def main() -> None:
    """Verify the harness separates FLOOR from library defaults."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    spectra = benchmark_set(n)

    floor = fit_all(spectra, truth_init_params)
    default = fit_all(spectra, default_init_params)

    print(f"{n} benchmark spectra, single-start LM, max_nfev={MAX_NFEV}\n")
    print_table(
        [
            summarise("FLOOR (truth)", floor, floor),
            summarise("A (defaults)", default, floor),
        ]
    )

    lib_floor = fit_all(spectra, truth_init_params, fit_library)
    lib_default = fit_all(spectra, default_init_params, fit_library)
    print("\nsecondary: shipped Circuit.fit(), which screens and restarts internally\n")
    print_table(
        [
            summarise("FLOOR (truth)", lib_floor, lib_floor),
            summarise("A (defaults)", lib_default, lib_floor),
        ]
    )


if __name__ == "__main__":
    main()
