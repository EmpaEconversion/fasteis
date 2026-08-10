"""Final evaluation: the shipped model against FLOOR and the library defaults.

Reports the convergence rate and the excess evaluations for fits.
Breaks down the excess evaluations by noise level and sweep width to show where
the model is weak.

Runs the embedded Rust model through `Circuit.guess`, so includes the extra
LM tricks for improving convergence, the same as what users will get.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits, evaluate, priors


def make_guess_init_params(circuit_str: str, weights: str | None = None):
    """Wrap a model as an evaluate.InitParams.

    `weights` loads a `.eisnn` by path, so a circuit that is not yet registered in
    src/models.rs can still be benchmarked.
    """
    built = fasteis.Circuit(circuit_str)

    def guess_init_params(spectrum: priors.Spectrum) -> np.ndarray:
        return np.array(built.guess(list(spectrum.freqs), list(spectrum.z), weights=weights))

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
    p.add_argument("--circuit", default="randles", help="which trained circuit")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--weights", default=None, help="a .eisnn path, else the embedded model")
    p.add_argument(
        "--results",
        type=Path,
        default=Path("training/results"),
        help="where the json goes; update_readme.py renders it",
    )
    args = p.parse_args()

    circuit = circuits.get(args.circuit)
    guess = make_guess_init_params(
        circuit.circuit_str if args.weights else circuit.name, args.weights
    )
    spectra = evaluate.benchmark_set(circuit, args.n)

    def table(fitter) -> list[evaluate.Summary]:
        """Fit every source with one fitter, scored against that fitter's own floor."""
        sources = [
            ("floor (truth)", evaluate.truth_init_params(circuit)),
            ("library defaults", evaluate.default_init_params(circuit)),
            (
                f"truth x/div {evaluate.PERTURB_FACTOR:.0f}",
                evaluate.make_perturbed_init_params(circuit),
            ),
            ("ml guess", guess),
        ]
        outcomes = [(name, evaluate.fit_all(circuit, spectra, fn, fitter)) for name, fn in sources]
        floor = outcomes[0][1]
        return [evaluate.summarise(name, o, floor) for name, o in outcomes]

    print(f"{args.n} benchmark spectra, circuit {circuit.name!r}")
    print("\nplain single-start LM\n")
    plain = table(evaluate.fit_plain_lm)
    evaluate.print_table(plain)

    print("\nCircuit.fit(), which screens candidate starts and restarts\n")
    library = table(evaluate.fit_library)
    evaluate.print_table(library)

    floor = evaluate.fit_all(circuit, spectra, evaluate.truth_init_params(circuit))
    model = evaluate.fit_all(circuit, spectra, guess)

    # measured directly rather than inferred by subtraction
    timed = spectra[:200]
    t0 = time.perf_counter()
    for s in timed:
        guess(s)
    inference_ms = 1e3 * (time.perf_counter() - t0) / len(timed)
    floor_fit_ms = 1e3 * sum(o.seconds for o in floor) / len(floor)
    print(
        f"\ninference {inference_ms:.2f} ms/spectrum, against {floor_fit_ms:.2f} ms "
        f"for the fit it starts"
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
    print("\ninitial parameter error before fitting (relative, %)")
    print(f"{'param':<12} {'median':>9} {'p90':>9} {'p99':>9}")
    param_error = {}
    for j, name in enumerate(circuit.param_names):
        err = 100.0 * np.abs(starts[:, j] / truth[:, j] - 1.0)
        param_error[name] = {
            "median": float(np.median(err)),
            "p90": float(np.percentile(err, 90)),
            "p99": float(np.percentile(err, 99)),
        }
        print(
            f"{name:<12} {param_error[name]['median']:>9.2f} "
            f"{param_error[name]['p90']:>9.2f} {param_error[name]['p99']:>9.2f}"
        )

    weights = Path("src/models") / f"{circuit.name}.eisnn"
    results = {
        "circuit": circuit.name,
        "circuit_str": circuit.circuit_str,
        "n_spectra": args.n,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights_bytes": weights.stat().st_size if weights.exists() else None,
        "inference_ms": inference_ms,
        "floor_fit_ms": floor_fit_ms,
        "fitters": {
            "plain_lm": [asdict(row) for row in plain],
            "circuit_fit": [asdict(row) for row in library],
        },
        "param_error_pct": param_error,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{circuit.name}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
