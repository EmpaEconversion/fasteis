"""The parameter guessing network.

A 1-D CNN over the log-frequency axis. Shifting a time constant translates
features along log-frequency, so use translation equivariance inductive bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from training import features

# tanh so src/nn.rs can reproduce it
GELU_APPROXIMATE = "tanh"

KERNEL = 5
DILATIONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Config:
    """Architecture of the CNN, stored in the checkpoint."""

    channels: int = 32
    blocks: int = 4
    head_width: int = 128
    groups: int = 8

    def dilations(self) -> tuple[int, ...]:
        """One dilation per block, cycling if there are more blocks than entries."""
        return tuple(DILATIONS[i % len(DILATIONS)] for i in range(self.blocks))

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT = Config()


class Block(nn.Module):
    """Residual pair of dilated convolutions."""

    def __init__(self, channels: int, dilation: int, groups: int) -> None:
        super().__init__()
        pad = dilation * (KERNEL - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, KERNEL, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, KERNEL, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU(approximate=GELU_APPROXIMATE)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class GuessNet(nn.Module):
    """Predicts standardised normalised parameters, plus a log-variance each."""

    def __init__(self, n_params: int, config: Config = DEFAULT) -> None:
        super().__init__()
        self.config = config
        self.n_params = n_params
        c = config.channels

        self.stem = nn.Conv1d(features.N_CHANNELS, c, KERNEL, padding=KERNEL // 2)
        self.blocks = nn.ModuleList(
            Block(c, dilation, config.groups) for dilation in config.dilations()
        )
        self.head = nn.Sequential(
            nn.Linear(2 * c + features.N_SCALARS, config.head_width),
            nn.GELU(approximate=GELU_APPROXIMATE),
            nn.Linear(config.head_width, config.head_width),
            nn.GELU(approximate=GELU_APPROXIMATE),
            nn.Linear(config.head_width, 2 * n_params),
        )

    def forward(self, grid: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (mu, log_var), both (B, n_params) in standardised target space."""
        h = self.stem(grid)
        for block in self.blocks:
            h = block(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1), scalars], dim=-1)
        out = self.head(pooled)
        mu, log_var = out.chunk(2, dim=-1)
        # keeps sigma in a sane range; NLL otherwise drifts toward degenerate values
        return mu, log_var.clamp(-10.0, 6.0)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
