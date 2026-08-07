"""Final evaluation: the shipped model against FLOOR and the library defaults.

Reports the convergence rate and the excess evaluations for fits.
Breaks down the excess evaluations by noise level and sweep width to show where
the model is weak.

Runs the embedded Rust model through `Circuit.guess`, so includes the extra
LM tricks for improving convergence, the same as what users will get.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits, evaluate, priors


def make_guess_init_params(name: str):
    """Wrap the embedded model as an evaluate.InitParams."""
    circuit = fasteis.Circuit(name)

    def guess_init_params(spectrum: priors.Spectrum) -> np.ndarray:
        return np.array(circuit.guess(list(spectrum.freqs), list(spectrum.z)))

    return guess_init_params


def _bucket_report(
    label: str, values: np.ndarray, excess: np.ndarray, converged: np.ndarray, edges
) -> None:
    print(f"\nexcess evaluations by {label}")
    print(f"{'bucket':<20} {'n':>6} {'converged':>10} {'median':>8} {'p90':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if mask.sum() == 0:
            continue
        ok = mask & converged
        med = np.median(excess[ok]) if ok.sum() else np.nan
        p90 = np.percentile(excess[ok], 90) if ok.sum() else np.nan
        print(
            f"{f'{lo:.3g} - {hi:.3g}':<20} {mask.sum():>6} "
            f"{100 * converged[mask].mean():>9.1f}% {med:>8.1f} {p90:>8.1f}"
        )


def main() -> None:
    """Run the benchmark."""
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="randles", help="registered model to benchmark")
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()

    guess = make_guess_init_params(args.name)
    spectra = evaluate.benchmark_set(args.n)

    floor = evaluate.fit_all(spectra, evaluate.truth_init_params)
    default = evaluate.fit_all(spectra, evaluate.default_init_params)
    model = evaluate.fit_all(spectra, guess)

    print(f"{args.n} benchmark spectra, basic LM, model {args.name!r}\n")
    evaluate.print_table(
        [
            evaluate.summarise("FLOOR (truth)", floor, floor),
            evaluate.summarise("A (defaults)", default, floor),
            evaluate.summarise("C (model)", model, floor),
        ]
    )

    # measured directly rather than inferred by subtraction
    timed = spectra[:200]
    t0 = time.perf_counter()
    for s in timed:
        guess(s)
    inference_ms = 1e3 * (time.perf_counter() - t0) / len(timed)
    floor_fit_ms = 1e3 * sum(o.seconds for o in floor) / len(floor)
    default_fit_ms = 1e3 * sum(o.seconds for o in default) / len(default)
    print(
        f"\ninference time {inference_ms:.2f} ms/spectrum"
        f"   (fit from truth {floor_fit_ms:.2f} ms, from defaults {default_fit_ms:.2f} ms)"
    )

    # secondary: what a user calling the shipped fit() actually sees
    lib_floor = evaluate.fit_all(spectra, evaluate.truth_init_params, evaluate.fit_library)
    lib_default = evaluate.fit_all(
        spectra, evaluate.default_init_params, evaluate.fit_library
    )
    lib_model = evaluate.fit_all(spectra, guess, evaluate.fit_library)
    print("\nsecondary: shipped Circuit.fit(), which screens and restarts internally\n")
    evaluate.print_table(
        [
            evaluate.summarise("FLOOR (truth)", lib_floor, lib_floor),
            evaluate.summarise("A (defaults)", lib_default, lib_floor),
            evaluate.summarise("C (model)", lib_model, lib_floor),
        ]
    )

    # where are the initial parameters weak?
    evals = np.array([o.evaluations for o in model], dtype=float)
    floor_evals = np.array([o.evaluations for o in floor], dtype=float)
    cost = np.array([o.cost for o in model])
    floor_cost = np.array([o.cost for o in floor])
    converged = cost <= evaluate.CONVERGENCE_TOL * floor_cost + 1e-12
    excess = evals - floor_evals

    _bucket_report(
        "noise sigma",
        np.array([s.noise for s in spectra]),
        excess,
        converged,
        [0.0, 0.005, 0.01, 0.02, 0.05, 1.0],
    )
    _bucket_report(
        "sweep width (decades)",
        np.array([np.log10(s.freqs[-1] / s.freqs[0]) for s in spectra]),
        excess,
        converged,
        [4.0, 5.0, 6.0, 7.0, 8.01],
    )

    # parameter recovery, diagnostic only: which parameter drags the count up
    truth = np.array([s.params for s in spectra])
    starts = np.array([guess(s) for s in spectra])
    print("\ninitial parameter error before fitting (decades, except alpha which is absolute)")
    print(f"{'param':<12} {'median':>9} {'p90':>9}")
    for j, name in enumerate(circuits.PARAM_NAMES):
        err = np.abs(starts[:, j] - truth[:, j]) if j == circuits.ALPHA else np.abs(np.log10(starts[:, j] / truth[:, j]))
        print(f"{name:<12} {np.median(err):>9.3f} {np.percentile(err, 90):>9.3f}")


if __name__ == "__main__":
    main()
