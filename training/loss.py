# Copyright © 2026, Empa.
"""Residual loss against the observed curve.

Generic: the only circuit-specific part is `TrainingCircuit.impedance_torch`,
which tests/test_parity.py ensures is the same as `Circuit.impedance()`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from training.circuits import TrainingCircuit


def spectrum_from_grid(grid: Tensor) -> tuple[Tensor, Tensor]:
    """Recover normalised (w, z) from the model's own input channels.

    features.py packs log10|Z_hat|, phase/(pi/2) and log10(w_hat)/4 into the three
    channels, so the loss can read the spectrum straight back out rather than
    carrying variable-length raw arrays through the dataloader.
    """
    log_mag, scaled_phase, scaled_log_w = (
        grid[..., 0, :],
        grid[..., 1, :],
        grid[..., 2, :],
    )
    w = torch.pow(10.0, 4.0 * scaled_log_w)
    magnitude = torch.pow(10.0, log_mag)
    phase = scaled_phase * (math.pi / 2.0)
    z = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    return w, z


def modulus_residuals(circuit: TrainingCircuit, params: Tensor, w: Tensor, z: Tensor) -> Tensor:
    """Modulus-weighted residuals, matching fit.rs: (z_model - z) / |z|.

    Returned as (..., n, 2) real rather than interleaved, which is equivalent for a
    sum of squares and keeps the tensor shape obvious.
    """
    model = circuit.impedance_torch(params, w)
    delta = (model - z) / torch.abs(z).clamp_min(1e-30)
    return torch.stack([delta.real, delta.imag], dim=-1)


def residual_loss(circuit: TrainingCircuit, params: Tensor, w: Tensor, z: Tensor) -> Tensor:
    """Mean squared modulus-weighted residual, per spectrum."""
    return modulus_residuals(circuit, params, w, z).pow(2).mean(dim=(-2, -1))
