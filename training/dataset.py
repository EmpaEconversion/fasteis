"""On-the-fly generation of training batches."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

import fasteis
from training import circuits, features, priors, scales


def encode(
    spectrum: priors.Spectrum, estimator: str = scales.DEFAULT
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float64]]:
    """Spectrum -> (grid, scalars, normalised targets)."""
    f = features.extract(spectrum.freqs, spectrum.z, estimator)
    targets = circuits.to_targets(
        circuits.to_normalised(spectrum.params[None], f.k, f.w_c)
    )[0]
    return (
        f.grid.astype(np.float32),
        f.scalars.astype(np.float32),
        targets,
    )


class RandlesStream(IterableDataset):
    """Endless stream of encoded synthetic impedance spectra."""

    def __init__(
        self,
        seed: int = 0,
        cfg: priors.PriorConfig = priors.DEFAULT,
        estimator: str = scales.DEFAULT,
    ) -> None:
        self.seed = seed
        self.cfg = cfg
        self.estimator = estimator

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        rng = priors.split_rng(priors.TRAINING, self.seed, worker)
        circuit = fasteis.Circuit(circuits.CIRCUIT_STRING)

        while True:
            spectrum = priors.sample(rng, self.cfg, circuit)
            grid, scalars, targets = encode(spectrum, self.estimator)
            yield (
                torch.from_numpy(grid),
                torch.from_numpy(scalars),
                torch.from_numpy(targets.astype(np.float32)),
            )


def target_statistics(
    n: int = 50_000, seed: int = 999, estimator: str = scales.DEFAULT
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-target mean and std, used to standardise the regression targets."""
    rng = np.random.default_rng(seed)
    spectra = priors.sample_many(rng, n)
    targets = np.array([encode(s, estimator)[2] for s in spectra])
    return targets.mean(axis=0), targets.std(axis=0)
