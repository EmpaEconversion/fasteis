# Copyright © 2026, Empa.
"""Sanity-checks for the priors of one circuit.

A prior is useless for training if it mostly produces curves whose features fall
outside the window, or whose parameters are swamped by noise. Feature visibility
is reported alongside a per-parameter identifiability estimate taken from the
circuit Jacobian at the true parameters. If a parameter's standard error there
is huge, no amount of training can recover it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits, priors

TAU = 2.0 * np.pi
N = 20_000
N_JACOBIAN = 2000  # jacobian work is heavier, so sample fewer

# above this relative standard error a parameter is not meaningfully constrained
UNIDENTIFIABLE = 1.0


def relative_stderr(
    circuit: circuits.TrainingCircuit,
    built: fasteis.Circuit,
    spectrum: priors.Spectrum,
) -> np.ndarray:
    """Per-parameter standard error at the true parameters, relative to the value.

    Modulus-weighted residuals are dimensionless multiples of the relative noise, so
    scaling by the spectrum's sigma turns the Jacobian into real uncertainties.
    """
    j = np.asarray(
        built.jacobian(list(spectrum.params), list(spectrum.freqs), list(spectrum.z), "modulus")
    )
    try:
        cov = np.linalg.inv(j.T @ j) * spectrum.noise**2
    except np.linalg.LinAlgError:
        return np.full(circuit.n_params, np.inf)
    var = np.diag(cov)
    var = np.where(var > 0, var, np.inf)
    return np.sqrt(var) / np.abs(spectrum.params)


def main() -> None:
    """Report observability statistics for one circuit's priors."""
    name = sys.argv[1] if len(sys.argv) > 1 else "randles"
    circuit = circuits.get(name)
    built = fasteis.Circuit(circuit.circuit_str)

    rng = priors.split_rng(priors.TRAINING, seed=999)
    spectra = priors.sample_many(rng, circuit, N)

    in_window = []
    phase_span = []
    for s in spectra:
        w_lo, w_hi = TAU * s.freqs[0], TAU * s.freqs[-1]
        # every circuit exposes its own characteristic frequency
        _, w_c = circuit.scales_from_params(s.params[None])
        in_window.append(w_lo <= float(w_c[0]) <= w_hi)
        phase_span.append(np.ptp(np.degrees(np.angle(s.z_clean))))

    def pct(x) -> str:
        return f"{100 * np.mean(x):.1f}%"

    n_points = [len(s.freqs) for s in spectra]
    decades = [np.log10(s.freqs[-1] / s.freqs[0]) for s in spectra]

    print(f"{circuit.name}: {N} spectra from the default priors\n")
    print(f"characteristic frequency inside window  {pct(in_window)}")
    print(
        f"phase span (deg)   median {np.median(phase_span):6.1f}"
        f"  p10 {np.percentile(phase_span, 10):6.1f}"
    )
    print(f"points per sweep   median {np.median(n_points):6.0f}")
    print(f"sweep width (dec)  median {np.median(decades):6.1f}")
    print(f"noise sigma        median {np.median([s.noise for s in spectra]):6.4f}")

    stderr = np.array([relative_stderr(circuit, built, s) for s in spectra[:N_JACOBIAN]])

    print(f"\nidentifiability at the true parameters ({N_JACOBIAN} spectra)")
    print(f"{'param':<12} {'median':>9} {'p90':>9} {'unidentifiable':>15}")
    for j, param in enumerate(circuit.param_names):
        col = stderr[:, j]
        print(
            f"{param:<12} {np.median(col):>9.3f} {np.percentile(col, 90):>9.3f} "
            f"{pct(col > UNIDENTIFIABLE):>15}"
        )
    worst = (stderr > UNIDENTIFIABLE).any(axis=1)
    print(f"\nspectra with >=1 unidentifiable parameter  {pct(worst)}")


if __name__ == "__main__":
    main()
