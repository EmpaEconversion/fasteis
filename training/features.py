# Copyright © 2026, Empa.
"""Interpolate and rescale the impedance spectrum for model inputs.

Interpolates the inputs to 64 points in log frequency.
Applies the scaling from scales.py, to pass the rescaled arrays to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from training import scales

if TYPE_CHECKING:
    from numpy.typing import NDArray

TAU = 2.0 * np.pi

N_GRID = 64
N_CHANNELS = 3
N_SCALARS = 2


@dataclass(frozen=True)
class Features:
    """Model input for one spectrum, plus the scales needed to undo normalisation."""

    grid: NDArray[np.float64]  # (3, N_GRID)
    scalars: NDArray[np.float64]  # (2,)
    k: float
    w_c: float


def _resample(
    freqs: NDArray[np.float64], z: NDArray[np.complex128]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate log|Z| and unwrapped phase onto N_GRID points across the sweep."""
    order = np.argsort(freqs)
    freqs, z = freqs[order], z[order]

    log_f = np.log10(freqs)
    grid_log_f = np.linspace(log_f[0], log_f[-1], N_GRID)

    log_mag = np.interp(grid_log_f, log_f, np.log10(np.abs(z)))
    phase = np.interp(grid_log_f, log_f, np.unwrap(np.angle(z)))
    return grid_log_f, log_mag, phase


def extract(
    freqs: NDArray[np.float64],
    z: NDArray[np.complex128],
    estimator: str = scales.DEFAULT,
    k: float | None = None,
    w_c: float | None = None,
) -> Features:
    """Build model input. Pass `k`/`w_c` explicitly for the second inference pass."""
    w = TAU * np.asarray(freqs, dtype=np.float64)
    z = np.asarray(z, dtype=np.complex128)

    if k is None or w_c is None:
        k, w_c = scales.estimate(w, z, estimator)

    grid_log_f, log_mag, phase = _resample(np.asarray(freqs, dtype=np.float64), z)

    # log10(w_i / w_c) == log10(f_i / f_c); the 2*pi cancels
    rel_log_w = grid_log_f - np.log10(w_c / TAU)

    grid = np.stack(
        [
            log_mag - np.log10(k),
            phase / (np.pi / 2.0),
            rel_log_w / 4.0,
        ]
    )
    scalars = np.array(
        [
            (grid_log_f[-1] - grid_log_f[0]) / 8.0,
            np.log10(len(freqs)) / 2.0,
        ]
    )
    return Features(grid=grid, scalars=scalars, k=float(k), w_c=float(w_c))
