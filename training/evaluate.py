# Copyright © 2026, Empa.
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
from training import priors
from training.circuits import TrainingCircuit

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
    """Result of one fit.

    `evaluations` counts full impedance sweeps, includes the normal evaluations
    and Jacobian, which is a central difference over every parameter.
    """

    success: bool
    evaluations: int
    cost: float
    params: NDArray[np.float64]
    seconds: float


def _to_free(circuit: TrainingCircuit, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Physical -> unconstrained coordinates (log for magnitudes, exponents linear)."""
    log = list(circuit.log_params)
    free = np.array(params, dtype=np.float64)
    free[log] = np.log(np.clip(free[log], 1e-300, None))
    return free


def _to_params(circuit: TrainingCircuit, free: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of `_to_free`."""
    log = list(circuit.log_params)
    params = np.array(free, dtype=np.float64)
    params[log] = np.exp(np.clip(params[log], -700, 700))
    return params


def fit_plain_lm(
    circuit: TrainingCircuit,
    spectrum: priors.Spectrum,
    init_params: NDArray[np.float64],
    built: fasteis.Circuit,
) -> Outcome:
    """Fit a single-start LM."""
    freqs = list(spectrum.freqs)
    z = list(spectrum.z)

    def residuals(free: NDArray[np.float64]) -> NDArray[np.float64]:
        p = [float(v) for v in _to_params(circuit, free)]
        return np.asarray(built.residuals(p, freqs, z, "modulus"))

    def jacobian(free: NDArray[np.float64]) -> NDArray[np.float64]:
        params = _to_params(circuit, free)
        j = np.asarray(built.jacobian([float(v) for v in params], freqs, z, "modulus"))
        # chain rule for the log-coordinate parameters
        scale = np.ones(circuit.n_params)
        scale[list(circuit.log_params)] = params[list(circuit.log_params)]
        return j * scale

    t0 = time.perf_counter()
    result = least_squares(
        residuals,
        _to_free(circuit, init_params),
        jac=jacobian,
        method="lm",
        ftol=FTOL,
        xtol=XTOL,
        max_nfev=MAX_NFEV,
    )
    elapsed = time.perf_counter() - t0

    params = _to_params(circuit, result.x)
    linear = list(circuit.linear_params)
    params[linear] = np.clip(params[linear], 0.0, 1.0)
    return Outcome(
        success=bool(result.success),
        # scipy counts residual and Jacobian calls separately; a Jacobian is
        # 2 sweeps per parameter inside Circuit.jacobian()
        evaluations=int(result.nfev + 2 * circuit.n_params * result.njev),
        cost=float(result.cost),
        params=params,
        seconds=elapsed,
    )


def fit_library(
    circuit: TrainingCircuit,
    spectrum: priors.Spectrum,
    init_params: NDArray[np.float64],
    built: fasteis.Circuit,
) -> Outcome:
    """Fit via Circuit.fit(), which screens candidate starts and restarts."""
    start = built.with_values([float(v) for v in init_params])

    t0 = time.perf_counter()
    result = start.fit(list(spectrum.freqs), list(spectrum.z), **LIBRARY_FIT_KWARGS)
    elapsed = time.perf_counter() - t0

    return Outcome(
        success=result.success,
        evaluations=int(result.impedance_evals),
        cost=result.cost,
        params=np.array([result.params[n] for n in circuit.param_names]),
        seconds=elapsed,
    )


def truth_init_params(_: TrainingCircuit) -> InitParams:
    """FLOOR: start at the answer."""

    def truth(spectrum: priors.Spectrum) -> NDArray[np.float64]:
        return spectrum.params

    return truth


def default_init_params(circuit: TrainingCircuit) -> InitParams:
    """The library's placeholder values, i.e. what a user gets with no guess."""
    values = np.array(fasteis.Circuit(circuit.circuit_str).param_values())

    def default(_: priors.Spectrum) -> NDArray[np.float64]:
        return values

    return default


# every magnitude out by this factor, up or down, and alpha out by this much
PERTURB_FACTOR = 5.0
PERTURB_ALPHA = 0.15


def make_perturbed_init_params(circuit: TrainingCircuit, seed: int = 0) -> InitParams:
    """Generate perturbed truth intial parameters.

    Each magnitude parameter is multiplied or divided by `PERTURB_FACTOR` at
    random and alpha shifted by `PERTURB_ALPHA`. Used to represent a reasonable
    guess at initial values, like may be expected in real world fitting.
    """
    rng = np.random.default_rng(seed)
    log = list(circuit.log_params)
    linear = list(circuit.linear_params)

    def perturbed(spectrum: priors.Spectrum) -> NDArray[np.float64]:
        params = np.array(spectrum.params, dtype=np.float64)
        signs = rng.choice([-1.0, 1.0], size=circuit.n_params)
        params[log] *= PERTURB_FACTOR ** signs[log]
        params[linear] = np.clip(
            params[linear] + signs[linear] * PERTURB_ALPHA, *circuit.alpha_range
        )
        return params

    return perturbed


Fitter = Callable[[TrainingCircuit, priors.Spectrum, NDArray[np.float64], fasteis.Circuit], Outcome]


def fit_all(
    circuit: TrainingCircuit,
    spectra: Sequence[priors.Spectrum],
    init_params: InitParams,
    fitter: Fitter = fit_plain_lm,
) -> list[Outcome]:
    """Fit every spectrum from the parameters `init_params` produces."""
    built = fasteis.Circuit(circuit.circuit_str)
    return [fitter(circuit, s, init_params(s), built) for s in spectra]


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


def summarise(name: str, outcomes: Sequence[Outcome], floor: Sequence[Outcome]) -> Summary:
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
        f"{'p99':>7} {'med sweeps':>11} {'seconds':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.name:<18} {100 * s.converged:>9.2f}% {s.median_excess:>12.1f} "
            f"{s.p90_excess:>7.1f} {s.p99_excess:>7.1f} {s.median_evaluations:>11.1f} "
            f"{s.total_seconds:>9.2f}"
        )


def validation_set(circuit: TrainingCircuit, n: int) -> list[priors.Spectrum]:
    """Spectra for choosing a checkpoint during training."""
    return priors.sample_many(priors.split_rng(priors.VALIDATION), circuit, n)


def benchmark_set(circuit: TrainingCircuit, n: int) -> list[priors.Spectrum]:
    """Spectra for the final numbers. Disjoint from training and validation."""
    return priors.sample_many(priors.split_rng(priors.BENCHMARK), circuit, n)


def main() -> None:
    """Check the harness separates FLOOR from library defaults, without a model."""
    from training import circuits  # noqa: PLC0415  (avoids an import cycle)

    args = sys.argv[1:]
    name = args[0] if args else "randles"
    n = int(args[1]) if len(args) > 1 else 500
    circuit = circuits.get(name)
    spectra = benchmark_set(circuit, n)

    floor = fit_all(circuit, spectra, truth_init_params(circuit))
    default = fit_all(circuit, spectra, default_init_params(circuit))

    print(f"{n} benchmark spectra, {name!r}, single-start LM, max_nfev={MAX_NFEV}\n")
    print_table(
        [
            summarise("floor (truth)", floor, floor),
            summarise("library defaults", default, floor),
        ]
    )


if __name__ == "__main__":
    main()
