"""Sampling of synthetic Randles spectra.

Sampling parameters log-uniformly over wide ranges mostly produces curves whose
features sit outside the measured window, which is useless for the network.
These priors are used to sample near where features are observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import fasteis
from training import circuits

if TYPE_CHECKING:
    from numpy.typing import NDArray

TAU = 2.0 * np.pi


@dataclass(frozen=True)
class Spectrum:
    """One synthetic measurement plus the parameters that produced it."""

    params: NDArray[np.float64]  # (5,) physical, in PARAM_NAMES order
    freqs: NDArray[np.float64]  # (n,) Hz, ascending
    z: NDArray[np.complex128]  # (n,) noisy
    z_clean: NDArray[np.complex128]  # (n,) noise-free
    noise: float  # relative sigma used


@dataclass(frozen=True)
class PriorConfig:
    """Ranges for the shape-space sampler. Defaults are the v1 baseline."""

    log_r0_over_r1: tuple[float, float] = (-2.0, 0.5)
    alpha: tuple[float, float] = (0.5, 1.0)
    p_alpha_one: float = 0.1  # extra mass at exactly 1.0 (ideal capacitor)
    log_wc_tau: tuple[float, float] = (-2.0, 2.0)  # arc position vs window centre
    log_ww_over_wc: tuple[float, float] = (-4.0, 0.5)  # diffusion onset vs window
    decades: tuple[float, float] = (4.0, 8.0)
    log_f_centre: tuple[float, float] = (-1.0, 4.0)
    n_points: tuple[int, int] = (20, 100)
    log_impedance_scale: tuple[float, float] = (-2.0, 4.0)
    log_noise: tuple[float, float] = (-2.7, -1.3)  # ~0.2% to 5% relative

    # artifacts, off by default until there is a clean baseline to compare against
    p_dropout: float = 0.0
    p_outlier: float = 0.0
    p_inductance: float = 0.0


DEFAULT = PriorConfig()


def _sweep(
    rng: np.random.Generator, cfg: PriorConfig
) -> tuple[NDArray[np.float64], float]:
    """Draw a frequency sweep; returns (freqs in Hz, window centre in rad/s)."""
    decades = rng.uniform(*cfg.decades)
    log_centre = rng.uniform(*cfg.log_f_centre)
    n = int(rng.integers(cfg.n_points[0], cfg.n_points[1] + 1))
    freqs = np.logspace(log_centre - decades / 2, log_centre + decades / 2, n)
    return freqs, float(TAU * 10.0**log_centre)


def _params(
    rng: np.random.Generator, cfg: PriorConfig, w_window: float
) -> NDArray[np.float64]:
    """Draw parameters positioned relative to the window centre `w_window`."""
    alpha = (
        1.0
        if rng.random() < cfg.p_alpha_one
        else float(rng.uniform(*cfg.alpha))
    )
    r1 = 1.0
    r0 = r1 * 10.0 ** rng.uniform(*cfg.log_r0_over_r1)

    tau = 10.0 ** rng.uniform(*cfg.log_wc_tau) / w_window
    q = tau**alpha / r1

    w_warburg = w_window * 10.0 ** rng.uniform(*cfg.log_ww_over_wc)
    aw = r1 * np.sqrt(w_warburg / 2.0)

    scale = 10.0 ** rng.uniform(*cfg.log_impedance_scale)
    return np.array([r0 * scale, q / scale, alpha, r1 * scale, aw * scale])


def _add_noise(
    rng: np.random.Generator, z: NDArray[np.complex128], sigma: float
) -> NDArray[np.complex128]:
    """Proportional complex noise: EIS instrument error scales with |Z|."""
    g = rng.standard_normal((2, z.size))
    return z * (1.0 + sigma * (g[0] + 1j * g[1]) / np.sqrt(2.0))


def sample(
    rng: np.random.Generator,
    cfg: PriorConfig = DEFAULT,
    circuit: fasteis.Circuit | None = None,
) -> Spectrum:
    """Draw one synthetic spectrum."""
    circuit = circuit if circuit is not None else fasteis.Circuit(circuits.CIRCUIT_STRING)

    freqs, w_window = _sweep(rng, cfg)
    params = _params(rng, cfg, w_window)

    z_clean = np.asarray(
        circuit.with_values(list(params)).impedance(list(freqs)), dtype=np.complex128
    )

    if cfg.p_inductance and rng.random() < cfg.p_inductance:
        # cable inductance; targets deliberately stay the Randles parameters
        target = rng.uniform(0.0, 0.2) * params[circuits.R1]
        inductance = target / (TAU * freqs[-1])
        z_clean = z_clean + 1j * TAU * freqs * inductance

    sigma = float(10.0 ** rng.uniform(*cfg.log_noise))
    z = _add_noise(rng, z_clean, sigma)

    if cfg.p_outlier and rng.random() < cfg.p_outlier:
        idx = rng.integers(0, z.size, rng.integers(1, 4))
        z[idx] *= 1.0 + rng.uniform(0.1, 0.5, idx.size)

    if cfg.p_dropout:
        keep = rng.random(z.size) >= rng.uniform(0.0, cfg.p_dropout)
        if keep.sum() >= cfg.n_points[0]:
            freqs, z, z_clean = freqs[keep], z[keep], z_clean[keep]

    return Spectrum(params=params, freqs=freqs, z=z, z_clean=z_clean, noise=sigma)


TRAINING, VALIDATION, BENCHMARK = 0, 1, 2

def split_rng(split: int, seed: int = 0, worker: int = 0) -> np.random.Generator:
    """Ensure the RNG seed used to train/test/validate are always different."""
    return np.random.default_rng([split, seed, worker])


def sample_many(
    rng: np.random.Generator, n: int, cfg: PriorConfig = DEFAULT
) -> list[Spectrum]:
    """Draw `n` spectra, reusing one parsed circuit."""
    circuit = fasteis.Circuit(circuits.CIRCUIT_STRING)
    return [sample(rng, cfg, circuit) for _ in range(n)]
