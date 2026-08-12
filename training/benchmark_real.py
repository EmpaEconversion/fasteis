"""Benchmarks initial-parameter sources against measured spectra.

Ground truth is unknown. The reference is the best chi-square any strategy
reached on that spectrum, and a fit counts as converged if it lands within
`CONVERGENCE_TOL` of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits, evaluate

FREQ = "Frequency / Hz"
RE = "Real Impedance / ohm"
IM = "Imaginary Impedance / ohm"
INDEX = "EIS Index / 1"


@dataclass(frozen=True)
class Measured:
    """One measured spectrum."""

    index: int
    freqs: np.ndarray
    z: np.ndarray


def load(path: Path) -> list[Measured]:
    """Split a parquet into one entry per EIS index, ascending in frequency."""
    df = pl.read_parquet(path)
    out = []
    for index in df[INDEX].unique().sort():
        g = df.filter(pl.col(INDEX) == index).sort(FREQ)
        out.append(
            Measured(
                index=int(index),
                freqs=g[FREQ].to_numpy(),
                z=g[RE].to_numpy() + 1j * g[IM].to_numpy(),
            )
        )
    return out


def _fit(built: fasteis.Circuit, m: Measured, **kwargs) -> tuple[float, int, float]:
    """Returns (chi_square, sweeps, seconds). Failures come back as inf chi-square."""
    freqs, z = list(m.freqs), list(m.z)
    t0 = time.perf_counter()
    try:
        result = built.fit(freqs, z, **{**evaluate.LIBRARY_FIT_KWARGS, **kwargs})
    except ValueError:
        return float("inf"), 0, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    chi = result.chi_square if np.isfinite(result.chi_square) else float("inf")
    return chi, int(result.impedance_evaluations), elapsed


def strategies(circuit: circuits.TrainingCircuit, *, with_global: bool) -> dict:
    """Name -> fit keyword arguments."""
    # the baselines are what you get without a model, so they opt out of the guess
    out = {
        "library defaults": {"guess_init": False},
        "ml guess": {"guess_init": True},
    }
    if with_global:
        out["differential_evolution"] = {
            "method": "differential_evolution",
            "seed": 0,
            "guess_init": False,
        }
    return out


def main() -> None:
    """Run every strategy over every measured spectrum."""
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="two_rq_l")
    p.add_argument("--data", type=Path, default=Path("tests/data/reda_20338409.parquet"))
    p.add_argument("--limit", type=int, default=None, help="first N spectra only")
    p.add_argument("--no-global", action="store_true", help="skip differential evolution")
    p.add_argument("--results", type=Path, default=Path("training/results"))
    args = p.parse_args()

    circuit = circuits.get(args.circuit)
    built = fasteis.Circuit(circuit.name)
    spectra = load(args.data)[: args.limit]
    names = strategies(circuit, with_global=not args.no_global)

    chi = {k: [] for k in names}
    sweeps = {k: [] for k in names}
    seconds = {k: [] for k in names}

    for m in spectra:
        for name, kwargs in names.items():
            c, s, t = _fit(built, m, **kwargs)
            chi[name].append(c)
            sweeps[name].append(s)
            seconds[name].append(t)

    best = np.min(np.array([chi[k] for k in names]), axis=0)

    print(f"{len(spectra)} measured spectra from {args.data.name}, circuit {circuit.name!r}\n")
    header = (
        f"{'source of initial parameters':<24} {'converged':>10} {'med sweeps':>11} "
        f"{'med ms':>8} {'med chi2':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        c = np.array(chi[name])
        converged = c <= evaluate.CONVERGENCE_TOL * best + 1e-30
        row = {
            "name": name,
            "converged": float(converged.mean()),
            "median_sweeps": float(np.median(sweeps[name])),
            "median_ms": float(1e3 * np.median(seconds[name])),
            "median_chi_square": float(np.median(c)),
        }
        rows.append(row)
        print(
            f"{name:<24} {100 * row['converged']:>9.2f}% {row['median_sweeps']:>11.0f} "
            f"{row['median_ms']:>8.2f} {row['median_chi_square']:>11.3e}"
        )

    ml = np.array(sweeps["ml guess"], dtype=float)
    other = np.array(sweeps["library defaults"], dtype=float)
    ratio = np.where(ml > 0, other / np.maximum(ml, 1), np.nan)
    print(
        f"\nsweeps saved against the defaults: median {np.nanmedian(ratio):.1f}x, "
        f"p10 {np.nanpercentile(ratio, 10):.1f}x, p90 {np.nanpercentile(ratio, 90):.1f}x"
    )

    results = {
        "circuit": circuit.name,
        "data": args.data.name,
        "n_spectra": len(spectra),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategies": rows,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{circuit.name}_real.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
