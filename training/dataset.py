"""On-the-fly generation of training batches."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

import fasteis
from training import circuits, features, priors, scales


def encode(
    circuit: circuits.TrainingCircuit,
    spectrum: priors.Spectrum,
    estimator: str = scales.DEFAULT,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    """Spectrum -> (grid, scalars, normalised targets)."""
    f = features.extract(spectrum.freqs, spectrum.z, estimator)
    targets = circuit.to_targets(
        circuit.to_normalised(spectrum.params[None], f.k, f.w_c)
    )[0]
    return (
        f.grid.astype(np.float32),
        f.scalars.astype(np.float32),
        targets,
    )


class SpectrumStream(IterableDataset):
    """Endless stream of encoded synthetic impedance spectra."""

    def __init__(
        self,
        circuit: circuits.TrainingCircuit,
        seed: int = 0,
        cfg: priors.PriorConfig = priors.DEFAULT,
        estimator: str = scales.DEFAULT,
    ) -> None:
        self.circuit = circuit
        self.seed = seed
        self.cfg = cfg
        self.estimator = estimator

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        rng = priors.split_rng(priors.TRAINING, self.seed, worker)
        built = fasteis.Circuit(self.circuit.circuit_str)

        while True:
            spectrum = priors.sample(rng, self.circuit, self.cfg, built)
            grid, scalars, targets = encode(self.circuit, spectrum, self.estimator)
            yield (
                torch.from_numpy(grid),
                torch.from_numpy(scalars),
                torch.from_numpy(targets.astype(np.float32)),
            )


def target_statistics(
    circuit: circuits.TrainingCircuit,
    n: int = 50_000,
    seed: int = 999,
    estimator: str = scales.DEFAULT,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-target mean and std, used to standardise the regression targets."""
    rng = np.random.default_rng(seed)
    spectra = priors.sample_many(rng, circuit, n)
    targets = np.array([encode(circuit, s, estimator)[2] for s in spectra])
    return targets.mean(axis=0), targets.std(axis=0)
