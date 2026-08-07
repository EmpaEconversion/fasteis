"""Compares the (k, w_c) estimators in scales.py.

A good scale estimator collapses physically equivalent impedance curves on top
of each other, independent of the circuit parameters, measurement window, or
noise. This script compares the estimators by measuring the spread of the
target, i.e. the rescaled circuit parameters.

cells:  Spread of the target across different cells, i.e. different inputs to
        the circuit. A scale estimator should be robust against different scales
        and shapes of the of circuits. Compare the spread against the
        unnormalized 'target' i.e. spread of the input parameters.

sweep:  Spread of the target for one circuit measured with different frequency
        sweeps (number of points, start point, end point).

noise:  Spread for one fixed cell and one fixed sweep with different noise.

A robust scale estimator needs to score well in ALL 3 metrics.
Window scores artifically high in 'cells' becuase prior puts the feature
relative to the window centre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fasteis
from training import circuits, priors, scales

TAU = 2.0 * np.pi

N_CELLS = 4000
N_SWEEPS = 8  # sweep variations per cell
N_NOISE = 8  # noise draws per cell


def _targets(
    circuit: circuits.TrainingCircuit,
    freqs: NDArray[np.float64],
    z: NDArray[np.complex128],
    params: NDArray[np.float64],
    estimator: str,
) -> NDArray[np.float64]:
    k, w_c = scales.estimate(TAU * freqs, z, estimator)
    return circuit.to_targets(circuit.to_normalised(params[None], k, w_c))[0]


def _sweeps_around(
    rng: np.random.Generator, w_arc: float, n: int
) -> list[NDArray[np.float64]]:
    """Sweeps that all observe the arc, but differ in width and centre."""
    out = []
    for _ in range(n):
        decades = rng.uniform(4.0, 8.0)
        centre = np.log10(w_arc / TAU) + rng.uniform(-1.5, 1.5)
        n_points = int(rng.integers(20, 101))
        out.append(np.logspace(centre - decades / 2, centre + decades / 2, n_points))
    return out


def run(
    circuit: circuits.TrainingCircuit,
    seed: int = 0,
) -> tuple[NDArray[np.float64], dict[str, dict[str, NDArray[np.float64]]]]:
    """Collect all three metrics for every estimator."""
    rng = np.random.default_rng(seed)
    built = fasteis.Circuit(circuit.circuit_str)
    names = list(scales.ESTIMATORS)

    raw = np.empty((N_CELLS, circuit.n_params))
    cells = {n: np.empty((N_CELLS, circuit.n_params)) for n in names}
    sweep = {n: np.empty((N_CELLS, circuit.n_params)) for n in names}
    noise = {n: np.empty((N_CELLS, circuit.n_params)) for n in names}

    for i in range(N_CELLS):
        spectrum = priors.sample(rng, circuit, built=built)
        params = spectrum.params
        raw[i] = circuit.to_targets(params[None])[0]

        _, w_arc = circuit.scales_from_params(params[None])
        variants = _sweeps_around(rng, float(w_arc[0]), N_SWEEPS)
        clean = [
            np.asarray(
                built.with_values(list(params)).impedance(list(f)),
                dtype=np.complex128,
            )
            for f in variants
        ]

        base_f, base_z = spectrum.freqs, spectrum.z_clean
        draws = [
            priors._add_noise(rng, base_z, spectrum.noise) for _ in range(N_NOISE)
        ]

        for name in names:
            cells[name][i] = _targets(circuit, spectrum.freqs, spectrum.z, params, name)
            sweep[name][i] = np.std(
                [_targets(circuit, f, z, params, name) for f, z in zip(variants, clean)], axis=0
            )
            noise[name][i] = np.std(
                [_targets(circuit, base_f, z, params, name) for z in draws], axis=0
            )

    return raw, {"cells": cells, "sweep": sweep, "noise": noise}


def main() -> None:
    """Compare every estimator and print the table."""
    name = sys.argv[1] if len(sys.argv) > 1 else "randles"
    circuit = circuits.get(name)
    raw, results = run(circuit)

    print(
        f"{N_CELLS} cells, {N_SWEEPS} sweeps and {N_NOISE} noise draws each\n"
        "all figures in decades; lower is better\n"
    )
    header = f"{'estimator':<20} {'param':<12} {'cells':>9} {'sweep':>9} {'noise':>9}"
    print(header)
    print("-" * len(header))

    for j, param in enumerate(circuit.param_names):
        print(f"{'none (control)':<20} {param:<12} {np.std(raw[:, j]):>9.3f} {'-':>9} {'-':>9}")
    print()

    for name in scales.ESTIMATORS:
        for j, param in enumerate(circuit.param_names):
            print(
                f"{name:<20} {param:<12} "
                f"{np.std(results['cells'][name][:, j]):>9.3f} "
                f"{np.mean(results['sweep'][name][:, j]):>9.3f} "
                f"{np.mean(results['noise'][name][:, j]):>9.3f}"
            )
        print()


if __name__ == "__main__":
    main()
