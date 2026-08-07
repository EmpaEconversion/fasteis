"""Differentiable Randles impedance, so the residual loss can backpropagate.

This duplicates the maths in src/elements.rs. tests/test_parity.py ensures they
are the same within numerical error.

Doubles as a batched generator for GPUs, where CPU generation via fasteis would
otherwise starve the model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from training import circuits


def impedance(params: Tensor, w: Tensor) -> Tensor:
    """Randles impedance for `R0-(CPE1,R1-W1)`.

    `params` is (..., 5) in PARAM_NAMES order, `w` is (..., n) angular frequency in
    rad/s. Returns complex (..., n).
    """
    r0 = params[..., circuits.R0].unsqueeze(-1)
    q = params[..., circuits.Q].unsqueeze(-1)
    alpha = params[..., circuits.ALPHA].unsqueeze(-1)
    r1 = params[..., circuits.R1].unsqueeze(-1)
    aw = params[..., circuits.AW].unsqueeze(-1)

    # (j*w)**alpha == w**alpha * exp(i*pi*alpha/2) for w > 0; avoids a complex power
    phase = math.pi / 2.0 * alpha
    y_cpe = q * w**alpha * torch.complex(torch.cos(phase), torch.sin(phase))

    z_w = aw * torch.complex(torch.ones_like(w), -torch.ones_like(w)) / torch.sqrt(w)
    z_branch = r1.to(z_w.dtype) + z_w

    return r0.to(z_w.dtype) + 1.0 / (y_cpe + 1.0 / z_branch)


def modulus_residuals(params: Tensor, w: Tensor, z: Tensor) -> Tensor:
    """Modulus-weighted residuals, matching fit.rs: (z_model - z) / |z|.

    Returned as (..., n, 2) real rather than interleaved, which is equivalent for a
    sum of squares and keeps the tensor shape obvious.
    """
    model = impedance(params, w)
    delta = (model - z) / torch.abs(z).clamp_min(1e-30)
    return torch.stack([delta.real, delta.imag], dim=-1)


def residual_loss(params: Tensor, w: Tensor, z: Tensor) -> Tensor:
    """Mean squared modulus-weighted residual, per spectrum."""
    return modulus_residuals(params, w, z).pow(2).mean(dim=(-2, -1))


def spectrum_from_grid(grid: Tensor) -> tuple[Tensor, Tensor]:
    """Recover normalised (w, z) from the model's own input channels.

    features.py packs log10|Z_hat|, phase/(pi/2) and log10(w_hat)/4 into the three
    channels, so the residual loss can read the spectrum straight back out rather
    than carrying variable-length raw arrays through the dataloader.
    """
    log_mag, scaled_phase, scaled_log_w = grid[..., 0, :], grid[..., 1, :], grid[..., 2, :]
    w = torch.pow(10.0, 4.0 * scaled_log_w)
    magnitude = torch.pow(10.0, log_mag)
    phase = scaled_phase * (math.pi / 2.0)
    z = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    return w, z
