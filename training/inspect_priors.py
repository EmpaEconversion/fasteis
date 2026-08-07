"""Sanity-checks for priors used in training.

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


def relative_stderr(circuit: fasteis.Circuit, spectrum: priors.Spectrum) -> np.ndarray:
    """Per-parameter standard error at the true parameters, relative to the value.

    Modulus-weighted residuals are dimensionless multiples of the relative noise, so
    scaling by the spectrum's sigma turns the Jacobian into real uncertainties.
    """
    j = np.asarray(
        circuit.jacobian(
            list(spectrum.params), list(spectrum.freqs), list(spectrum.z), "modulus"
        )
    )
    try:
        cov = np.linalg.inv(j.T @ j) * spectrum.noise**2
    except np.linalg.LinAlgError:
        return np.full(circuits.N_PARAMS, np.inf)
    var = np.diag(cov)
    var = np.where(var > 0, var, np.inf)
    return np.sqrt(var) / np.abs(spectrum.params)


def main() -> None:
    """Report observability statistics for the default priors."""
    rng = np.random.default_rng(0)
    spectra = priors.sample_many(rng, N)

    arc_in_window = []
    warburg_in_window = []
    has_peak = []
    phase_span = []
    log_r_ratio = []

    for s in spectra:
        p = s.params
        w_lo, w_hi = TAU * s.freqs[0], TAU * s.freqs[-1]
        tau = (p[circuits.R1] * p[circuits.Q]) ** (1.0 / p[circuits.ALPHA])
        w_arc = 1.0 / tau
        w_warburg = 2.0 * p[circuits.AW] ** 2 / p[circuits.R1] ** 2

        arc_in_window.append(w_lo <= w_arc <= w_hi)
        warburg_in_window.append(w_lo <= w_warburg <= w_hi)

        neg_im = -s.z_clean.imag
        interior = np.argmax(neg_im)
        has_peak.append(0 < interior < len(neg_im) - 1)

        phase = np.angle(s.z_clean)
        phase_span.append(np.ptp(np.degrees(phase)))
        log_r_ratio.append(np.log10(p[circuits.R1] / p[circuits.R0]))

    def pct(x: list[bool]) -> str:
        return f"{100 * np.mean(x):.1f}%"

    print(f"{N} spectra from the default priors\n")
    print(f"CPE arc peak inside window      {pct(arc_in_window)}")
    print(f"Warburg onset inside window     {pct(warburg_in_window)}")
    print(f"interior max of -Im(Z) present  {pct(has_peak)}")
    print()
    print(f"phase span (deg)   median {np.median(phase_span):6.1f}  p10 {np.percentile(phase_span, 10):6.1f}")
    print(f"log10(R1/R0)       median {np.median(log_r_ratio):6.2f}  p10 {np.percentile(log_r_ratio, 10):6.2f}")
    print()

    n_points = [len(s.freqs) for s in spectra]
    decades = [np.log10(s.freqs[-1] / s.freqs[0]) for s in spectra]
    print(f"points per sweep   median {np.median(n_points):6.0f}")
    print(f"sweep width (dec)  median {np.median(decades):6.1f}")
    print(f"noise sigma        median {np.median([s.noise for s in spectra]):6.4f}")

    circuit = fasteis.Circuit(circuits.CIRCUIT_STRING)
    stderr = np.array([relative_stderr(circuit, s) for s in spectra[:N_JACOBIAN]])

    print(f"\nidentifiability at the true parameters ({N_JACOBIAN} spectra)")
    print(f"{'param':<12} {'median':>9} {'p90':>9} {'unidentifiable':>15}")
    for j, param in enumerate(circuits.PARAM_NAMES):
        col = stderr[:, j]
        print(
            f"{param:<12} {np.median(col):>9.3f} {np.percentile(col, 90):>9.3f} "
            f"{pct(col > UNIDENTIFIABLE):>15}"
        )
    worst = (stderr > UNIDENTIFIABLE).any(axis=1)
    print(f"\nspectra with >=1 unidentifiable parameter  {pct(worst)}")


if __name__ == "__main__":
    main()
