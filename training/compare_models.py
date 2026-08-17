"""Compares trained models for one circuit on synthetic and measured spectra.

Scores each `.eisnn` by what a fit costs after using its guess. Score is the
residual it reaches and the number of impedance calculations to get there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import benchmark_real, circuits, evaluate

TOL = evaluate.CONVERGENCE_TOL


def _fit_all(built: fasteis.Circuit, spectra, weights: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Fit every spectrum from one model's guess. Returns (chi_square, sweeps)."""
    chi, sweeps = [], []
    for m in spectra:
        freqs, z = list(m.freqs), list(m.z)
        try:
            r = built.fit(freqs, z, guess_init=True, weights=weights, **evaluate.LIBRARY_FIT_KWARGS)
            chi.append(r.chi_square if np.isfinite(r.chi_square) else np.inf)
            sweeps.append(r.impedance_evals)
        except ValueError:
            chi.append(np.inf)
            sweeps.append(0)
    return np.array(chi), np.array(sweeps, dtype=float)


def _report(label: str, models: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Markdown table with best chi-square and cost."""
    best = np.min(np.array([c for c, _ in models.values()]), axis=0)
    print(f"\n{label}\n")
    header = f"{'model':<28} {'reached best':>13} {'med chi2':>11} {'med sweeps':>11} {'p90':>9}"
    print(header)
    print("-" * len(header))
    for name, (chi, sweeps) in models.items():
        ok = chi <= TOL * best + 1e-30
        print(
            f"{name:<28} {100 * ok.mean():>12.2f}% {np.median(chi):>11.3e} "
            f"{np.median(sweeps):>11.0f} {np.percentile(sweeps, 90):>9.0f}"
        )


def main() -> None:
    """Compare every given model on synthetic and measured spectra."""
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="two_rq_l")
    p.add_argument(
        "weights",
        nargs="+",
        help="one or more .eisnn files; 'embedded' uses the model built into the crate",
    )
    p.add_argument("--data", type=Path, default=Path("tests/data/reda_20338409.parquet"))
    p.add_argument("--n", type=int, default=500, help="synthetic spectra")
    args = p.parse_args()

    circuit = circuits.get(args.circuit)
    built = fasteis.Circuit(circuit.name)
    paths = {w: (None if w == "embedded" else w) for w in args.weights}

    synthetic = evaluate.benchmark_set(circuit, args.n)
    measured = benchmark_real.load(args.data)

    for label, spectra in (
        (f"synthetic, {len(synthetic)} spectra", synthetic),
        (f"measured, {len(measured)} spectra from {args.data.name}", measured),
    ):
        results = {name: _fit_all(built, spectra, path) for name, path in paths.items()}
        _report(label, results)


if __name__ == "__main__":
    main()
