"""Everything that varies from one trainable circuit to the next.

A `TrainingCircuit` contains the circuit topology, parameters, how those
parameters pick up the normalisation scales, how to draw plausible parameters,
and a differentiable impedance for the training loss. Every other module in
`training/` is generic and takes one of these as an argument.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from training.priors import PriorConfig


class TrainingCircuit:
    """Base class for a circuit to train."""

    # Should match the circuit and model name
    name: str

    # String used to build the circuit
    circuit_str: str

    # Order must match fasteis.Circuit(circuit_str).param_names()
    param_names: tuple[str, ...]

    # Indices of parameters that stay linear, i.e. the bounded CPE exponents.
    # Everything else is log-scaled; see `log_params`.
    linear_params: tuple[int, ...]

    # Exponents are predicted unbounded, so they are clamped before evaluating
    alpha_range: tuple[float, float] = (0.05, 1.0)

    # How each parameter picks up the impedance scale k and frequency scale w_c:
    #   physical = normalised * k**a * w_c**(b + c * params[i])
    # i = -1 means no parameter dependence
    # Each parameter has a row (a, b, c, i)
    # Resistances are (1, 0, 0, -1), capacitances (-1, -1, 0, -1)
    # inductances (1, -1, 0, -1), time constants (0, -1, 0, -1)
    scaling: NDArray[np.float64]

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return len(self.param_names)

    @property
    def log_params(self) -> tuple[int, ...]:
        """Indices held in log space: everything that is not in `linear_params`."""
        return tuple(i for i in range(self.n_params) if i not in self.linear_params)

    def _scale_factors(
        self,
        params: NDArray[np.float64],
        k: float | NDArray[np.float64],
        w_c: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """`k**a * w_c**(b + c*param)` per parameter, shape (batch, n_params)."""
        a = self.scaling[:, 0]
        b = self.scaling[:, 1]
        c = self.scaling[:, 2]
        idx = self.scaling[:, 3].astype(int)
        exponent = np.broadcast_to(b, params.shape).copy()
        dependent = idx >= 0
        if dependent.any():
            exponent[:, dependent] += c[dependent] * params[:, idx[dependent]]

        return k**a * w_c**exponent

    def to_normalised(
        self,
        params: NDArray[np.float64],
        k: float | NDArray[np.float64],
        w_c: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Physical parameters -> parameters of the same curve after Z/k and w/w_c."""
        params = np.atleast_2d(params)
        k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
        w_c = np.asarray(w_c, dtype=np.float64).reshape(-1, 1)
        # exponents depending on alpha use it from the parameters, and alpha is invariant
        return params / self._scale_factors(params, k, w_c)

    def to_physical(
        self,
        params_hat: NDArray[np.float64],
        k: float | NDArray[np.float64],
        w_c: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Inverse of `to_normalised`."""
        params_hat = np.atleast_2d(params_hat)
        k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
        w_c = np.asarray(w_c, dtype=np.float64).reshape(-1, 1)
        return params_hat * self._scale_factors(params_hat, k, w_c)

    def to_targets(self, params_hat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalised parameters -> regression targets (log10 except the exponents)."""
        out = np.array(params_hat, dtype=np.float64, copy=True)
        out[..., self.log_params] = np.log10(out[..., self.log_params])
        return out

    def from_targets(self, targets: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse of `to_targets`."""
        out = np.array(targets, dtype=np.float64, copy=True)
        out[..., self.log_params] = 10.0 ** out[..., self.log_params]
        return out

    def scales_from_params(
        self,
        params: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the (k, w_c) a parameter vector itself implies.

        Needs to be defined per-circuit.
        """
        raise NotImplementedError

    def sample_params(
        self, rng: np.random.Generator, cfg: PriorConfig, w_window: float
    ) -> NDArray[np.float64]:
        """Draw parameters with features placed relative to the window centre.

        Sampling log-uniformly over wide ranges mostly puts features outside the
        measured window, which teaches the network nothing, so each circuit
        samples in its own observable shape space instead.
        """
        raise NotImplementedError

    def impedance_torch(self, params: Tensor, w: Tensor) -> Tensor:
        """Differentiable impedance, so the residual loss can backpropagate.

        `params` is (..., n_params), `w` is (..., n) in rad/s. Duplicates the
        maths in src/elements.rs, tests/test_parity.py ensures they agree.
        """
        raise NotImplementedError


def _j_pow(alpha: Tensor, w: Tensor) -> Tensor:
    """`(j*w)**alpha` for w > 0, avoiding a complex power."""
    import torch  # noqa: PLC0415  (training-only dependency)

    phase = (math.pi / 2.0) * alpha
    return w**alpha * torch.complex(torch.cos(phase), torch.sin(phase))


class TrainingRandles(TrainingCircuit):
    """Randles circuit with constant phase element."""

    def __init__(self) -> None:
        """Init the object."""
        self.name = "randles"
        self.circuit_str = "R0-(CPE1,R1-W1)"
        self.param_names = ("R0.r", "CPE1.q", "CPE1.alpha", "R1.r", "W1.aw")
        self.linear_params = (2,)  # CPE1.alpha

        # shape ranges, all relative to the measured window
        self.log_r0_over_r1 = (-2.0, 0.5)
        self.log_wc_tau = (-2.0, 2.0)  # arc position vs window centre
        self.log_ww_over_wc = (-4.0, 0.5)  # diffusion onset vs window centre

        self.scaling = np.array(
            [
                [1.0, 0.0, 0.0, -1],  # R0.r
                [-1.0, 0.0, -1.0, 2],  # CPE1.q, exponent -alpha
                [0.0, 0.0, 0.0, -1],  # CPE1.alpha
                [1.0, 0.0, 0.0, -1],  # R1.r
                [1.0, 0.5, 0.0, -1],  # W1.aw
            ]
        )

    def scales_from_params(
        self,
        params: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Charge-transfer resistance and CPE relaxation rate."""
        params = np.atleast_2d(params)
        r1 = params[:, 3]
        alpha = params[:, 2]
        tau = (r1 * params[:, 1]) ** (1.0 / alpha)
        return r1, 1.0 / tau

    def sample_params(
        self, rng: np.random.Generator, cfg: PriorConfig, w_window: float
    ) -> NDArray[np.float64]:
        """Draw parameters positioned relative to the window centre `w_window`."""
        alpha = cfg.draw_alpha(rng)
        r1 = 1.0
        r0 = r1 * 10.0 ** rng.uniform(*self.log_r0_over_r1)

        tau = 10.0 ** rng.uniform(*self.log_wc_tau) / w_window
        q = tau**alpha / r1

        # the frequency where the Warburg magnitude reaches r1
        w_warburg = w_window * 10.0 ** rng.uniform(*self.log_ww_over_wc)
        aw = r1 * np.sqrt(w_warburg / 2.0)

        scale = 10.0 ** rng.uniform(*cfg.log_impedance_scale)
        return np.array([r0 * scale, q / scale, alpha, r1 * scale, aw * scale])

    def impedance_torch(self, params: Tensor, w: Tensor) -> Tensor:
        """`R0-(CPE1,R1-W1)`."""
        import torch  # noqa: PLC0415  (training-only dependency)

        r0 = params[..., 0].unsqueeze(-1)
        q = params[..., 1].unsqueeze(-1)
        alpha = params[..., 2].unsqueeze(-1)
        r1 = params[..., 3].unsqueeze(-1)
        aw = params[..., 4].unsqueeze(-1)

        y_cpe = q * _j_pow(alpha, w)
        z_w = aw * torch.complex(torch.ones_like(w), -torch.ones_like(w)) / torch.sqrt(w)
        branch = r1.to(z_w.dtype) + z_w
        return r0.to(z_w.dtype) + 1.0 / (y_cpe + 1.0 / branch)


class TrainingTwoRqL(TrainingCircuit):
    """Two depressed arcs in series, with cable inductance.

    `LR(RQ)(RQ)` in Boukamp circuit description code. Widely described as a
    two-time-constant or two-arc model; `RQ` is the usual shorthand for a
    parallel resistor and CPE, distinguishing it from an ideal-capacitor `RC`.
    The inductance is a measurement artifact rather than part of the cell, so it
    is the `_l` suffix rather than part of the base name.
    """

    def __init__(self) -> None:
        """Init the object."""
        self.name = "two_rq_l"
        self.circuit_str = "L0-R0-(R1,CPE1)-(R2,CPE2)"
        self.param_names = (
            "L0.l",
            "R0.r",
            "R1.r",
            "CPE1.q",
            "CPE1.alpha",
            "R2.r",
            "CPE2.q",
            "CPE2.alpha",
        )
        self.linear_params = (4, 7)  # CPE1.alpha, CPE2.alpha

        self.log_r0_over_r1 = (-2.0, 0.5)
        self.log_r2_over_r1 = (-1.0, 1.0)
        self.log_wc_tau1 = (-2.0, 1.0)  # faster arc vs window centre
        self.log_tau_ratio = (0.5, 3.0)  # slower arc below the faster one
        self.log_wl_over_wc = (1.0, 4.0)  # inductance onset above the window

        self.scaling = np.array(
            [
                [1.0, -1.0, 0.0, -1],  # L0.l
                [1.0, 0.0, 0.0, -1],  # R0.r
                [1.0, 0.0, 0.0, -1],  # R1.r
                [-1.0, 0.0, -1.0, 4],  # CPE1.q, exponent -alpha
                [0.0, 0.0, 0.0, -1],  # CPE1.alpha
                [1.0, 0.0, 0.0, -1],  # R2.r
                [-1.0, 0.0, -1.0, 7],  # CPE2.q, exponent -alpha
                [0.0, 0.0, 0.0, -1],  # CPE2.alpha
            ]
        )

    def scales_from_params(
        self,
        params: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Mean arc resistance and mean relaxation rate of the two arcs."""
        params = np.atleast_2d(params)
        r1 = params[:, 2]
        r2 = params[:, 5]
        tau1 = (r1 * params[:, 3]) ** (1.0 / params[:, 4])
        tau2 = (r2 * params[:, 6]) ** (1.0 / params[:, 7])
        return (r1 + r2) / 2.0, (1.0 / tau1 + 1.0 / tau2) / 2.0

    def sample_params(
        self, rng: np.random.Generator, cfg: PriorConfig, w_window: float
    ) -> NDArray[np.float64]:
        """Draw parameters positioned relative to the window centre `w_window`.

        The two arcs are drawn with a controlled separation; overlapping arcs are
        barely distinguishable, so sampling their time constants independently would
        spend most of the prior on unidentifiable cases.
        """
        alpha1 = cfg.draw_alpha(rng)
        alpha2 = cfg.draw_alpha(rng)

        r1 = 1.0
        r2 = r1 * 10.0 ** rng.uniform(*self.log_r2_over_r1)
        r0 = r1 * 10.0 ** rng.uniform(*self.log_r0_over_r1)

        tau1 = 10.0 ** rng.uniform(*self.log_wc_tau1) / w_window
        tau2 = tau1 * 10.0 ** rng.uniform(*self.log_tau_ratio)
        q1 = tau1**alpha1 / r1
        q2 = tau2**alpha2 / r2

        # the frequency where the inductive magnitude reaches r1
        w_l = w_window * 10.0 ** rng.uniform(*self.log_wl_over_wc)
        inductance = r1 / w_l

        scale = 10.0 ** rng.uniform(*cfg.log_impedance_scale)
        return np.array(
            [
                inductance * scale,
                r0 * scale,
                r1 * scale,
                q1 / scale,
                alpha1,
                r2 * scale,
                q2 / scale,
                alpha2,
            ]
        )

    def impedance_torch(self, params: Tensor, w: Tensor) -> Tensor:
        """`L0-R0-p(R1,CPE1)-p(R2,CPE2)`."""
        import torch  # noqa: PLC0415  (training-only dependency)

        def unsq(i: int) -> Tensor:
            return params[..., i].unsqueeze(-1)

        zero = torch.zeros_like(w)
        z_l = torch.complex(zero, unsq(0) * w)
        z = z_l + unsq(1).to(z_l.dtype)
        for r, q, alpha in ((unsq(2), unsq(3), unsq(4)), (unsq(5), unsq(6), unsq(7))):
            # parallel R and CPE: R / (1 + R*q*(j w)**alpha)
            z = z + r.to(z_l.dtype) / (1.0 + r * q * _j_pow(alpha, w))
        return z


# Registry for all training circuits
CIRCUITS: dict[str, TrainingCircuit] = {
    circuit.name: circuit for circuit in (TrainingRandles(), TrainingTwoRqL())
}


def get(name: str) -> TrainingCircuit:
    """Look up a trainable circuit by name."""
    try:
        return CIRCUITS[name]
    except KeyError:
        available = ", ".join(repr(n) for n in CIRCUITS)
        message = f"unknown circuit {name!r}, expected one of {available}"
        raise KeyError(message) from None
