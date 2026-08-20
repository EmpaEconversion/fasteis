# Copyright © 2026, Empa.
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


class ArcChain(TrainingCircuit):
    """`[L0-]R0` then R/C or R/CPE arcs, optionally with diffusion.

    Covers many equivalent circuits for batteries. Arcs are ordered by ascending
    relaxation time, and constrained to keep this ordering.
    """

    def __init__(
        self,
        name: str,
        n_arcs: int,
        *,
        cpe: bool = True,
        inductor: bool = False,
        diffusion: str | None = None,
        **ranges: tuple[float, float],
    ) -> None:
        """Build one circuit from its structure."""
        import fasteis  # noqa: PLC0415  (parser is the source of truth for names)

        if diffusion not in (None, "W", "Wo"):
            message = f"{name}: diffusion must be None, 'W' or 'Wo'"
            raise ValueError(message)

        self.name = name
        self.n_arcs = n_arcs
        self.cpe = cpe
        self.inductor = inductor
        self.diffusion = diffusion

        element = "CPE" if cpe else "C"
        parts = (["L0"] if inductor else []) + ["R0"]
        for i in range(1, n_arcs + 1):
            branch = f"R{i}-{diffusion}{i}" if diffusion and i == n_arcs else f"R{i}"
            parts.append(f"({branch},{element}{i})")
        self.circuit_str = "-".join(parts)

        self.param_names = tuple(fasteis.Circuit(self.circuit_str).param_names())
        index = {n: i for i, n in enumerate(self.param_names)}

        rows = {n: [1.0, 0.0, 0.0, -1] for n in self.param_names}  # resistances
        if inductor:
            rows["L0.l"] = [1.0, -1.0, 0.0, -1]
        for i in range(1, n_arcs + 1):
            if cpe:
                rows[f"CPE{i}.q"] = [-1.0, 0.0, -1.0, index[f"CPE{i}.alpha"]]
                rows[f"CPE{i}.alpha"] = [0.0, 0.0, 0.0, -1]
            else:
                rows[f"C{i}.c"] = [-1.0, -1.0, 0.0, -1]
        if diffusion == "W":
            rows[f"W{n_arcs}.aw"] = [1.0, 0.5, 0.0, -1]
        elif diffusion == "Wo":
            # finite-space diffusion: z0 is impedance-like, tau is a time constant
            rows[f"Wo{n_arcs}.z0"] = [1.0, 0.0, 0.0, -1]
            rows[f"Wo{n_arcs}.tau"] = [0.0, -1.0, 0.0, -1]

        self.scaling = np.array([rows[n] for n in self.param_names])
        self.linear_params = tuple(
            i for i, n in enumerate(self.param_names) if n.endswith(".alpha")
        )

        # shape ranges, all relative to the measured window
        self.log_r0_over_r1 = (-2.0, 0.5)
        self.log_r_ratio = (-1.0, 1.0)  # later arcs against the first
        self.log_wc_tau1 = (-2.0, 1.0)  # fastest arc vs window centre
        self.log_tau_ratio = (1.0, 2.5)  # each arc slower than the last
        self.log_wl_over_wc = (1.0, 4.0)  # inductance onset above the window
        self.log_ww_over_wc = (-4.0, 0.5)  # semi-infinite diffusion onset
        self.log_wd_over_wc = (-3.0, 0.5)  # finite diffusion corner vs window
        self.log_z0_over_r = (-1.0, 1.0)  # diffusion plateau vs its own arc
        for key, value in ranges.items():
            if not hasattr(self, key):
                message = f"{name}: unknown prior range {key!r}"
                raise AttributeError(message)
            setattr(self, key, value)

    def _arc(self, params: NDArray[np.float64], i: int) -> tuple[NDArray, NDArray]:
        """Resistance and relaxation rate of arc `i` (1-based), per row."""
        index = {n: j for j, n in enumerate(self.param_names)}
        r = params[:, index[f"R{i}.r"]]
        if self.cpe:
            q = params[:, index[f"CPE{i}.q"]]
            alpha = params[:, index[f"CPE{i}.alpha"]]
            tau = (r * q) ** (1.0 / alpha)
        else:
            tau = r * params[:, index[f"C{i}.c"]]
        return r, 1.0 / tau

    def scales_from_params(
        self,
        params: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Mean arc resistance and mean relaxation rate over every arc."""
        params = np.atleast_2d(params)
        arcs = [self._arc(params, i) for i in range(1, self.n_arcs + 1)]
        k = np.mean([r for r, _ in arcs], axis=0)
        w_c = np.mean([w for _, w in arcs], axis=0)
        return k, w_c

    def sample_params(
        self, rng: np.random.Generator, cfg: PriorConfig, w_window: float
    ) -> NDArray[np.float64]:
        """Draw parameters with every feature placed relative to the window."""
        values: dict[str, float] = {}
        r1 = 1.0
        values["R0.r"] = r1 * 10.0 ** rng.uniform(*self.log_r0_over_r1)

        tau = 10.0 ** rng.uniform(*self.log_wc_tau1) / w_window
        for i in range(1, self.n_arcs + 1):
            if i > 1:
                tau *= 10.0 ** rng.uniform(*self.log_tau_ratio)
            r = r1 if i == 1 else r1 * 10.0 ** rng.uniform(*self.log_r_ratio)
            values[f"R{i}.r"] = r
            if self.cpe:
                alpha = cfg.draw_alpha(rng)
                values[f"CPE{i}.alpha"] = alpha
                values[f"CPE{i}.q"] = tau**alpha / r
            else:
                values[f"C{i}.c"] = tau / r

        if self.inductor:
            # the frequency where the inductive magnitude reaches r1
            values["L0.l"] = r1 / (w_window * 10.0 ** rng.uniform(*self.log_wl_over_wc))
        r_last = values[f"R{self.n_arcs}.r"]
        if self.diffusion == "W":
            # the frequency where the Warburg magnitude reaches its own arc's r
            w_warburg = w_window * 10.0 ** rng.uniform(*self.log_ww_over_wc)
            values[f"W{self.n_arcs}.aw"] = r_last * np.sqrt(w_warburg / 2.0)
        elif self.diffusion == "Wo":
            # the corner where it turns from 45 degrees to a capacitive tail
            w_diff = w_window * 10.0 ** rng.uniform(*self.log_wd_over_wc)
            values[f"Wo{self.n_arcs}.tau"] = 1.0 / w_diff
            values[f"Wo{self.n_arcs}.z0"] = r_last * 10.0 ** rng.uniform(*self.log_z0_over_r)

        scale = 10.0 ** rng.uniform(*cfg.log_impedance_scale)
        out = np.array([values[n] for n in self.param_names])
        # k**a fixes how each parameter carries the overall impedance scale
        return out * scale ** self.scaling[:, 0]

    def impedance_torch(self, params: Tensor, w: Tensor) -> Tensor:
        """Series inductor and resistor, then one parallel arc at a time."""
        import torch  # noqa: PLC0415  (training-only dependency)

        index = {n: i for i, n in enumerate(self.param_names)}

        def value(name: str) -> Tensor:
            return params[..., index[name]].unsqueeze(-1)

        complex_dtype = torch.complex(w, w).dtype
        z = value("R0.r").to(complex_dtype)
        if self.inductor:
            z = z + torch.complex(torch.zeros_like(w), value("L0.l") * w)

        for i in range(1, self.n_arcs + 1):
            branch = value(f"R{i}.r").to(complex_dtype)
            if self.diffusion == "W" and i == self.n_arcs:
                ones = torch.ones_like(w)
                branch = branch + value(f"W{i}.aw") * torch.complex(ones, -ones) / torch.sqrt(w)
            elif self.diffusion == "Wo" and i == self.n_arcs:
                # z0 / (x tanh x) with x = sqrt(j w tau)
                x = _j_pow(torch.full_like(w, 0.5), w * value(f"Wo{i}.tau"))
                branch = branch + value(f"Wo{i}.z0").to(complex_dtype) / (x * torch.tanh(x))
            if self.cpe:
                admittance = value(f"CPE{i}.q") * _j_pow(value(f"CPE{i}.alpha"), w)
            else:
                admittance = torch.complex(torch.zeros_like(w), value(f"C{i}.c") * w)
            z = z + 1.0 / (admittance + 1.0 / branch)
        return z


# Registry for all training circuits
CIRCUITS: dict[str, TrainingCircuit] = {
    circuit.name: circuit
    for circuit in (
        # one arc
        ArcChain("rc", 1, cpe=False),
        ArcChain("rc_l", 1, cpe=False, inductor=True),
        ArcChain("rq", 1),
        ArcChain("rq_l", 1, inductor=True),
        # two arcs
        ArcChain("two_rc", 2, cpe=False),
        ArcChain("two_rc_l", 2, cpe=False, inductor=True),
        ArcChain("two_rq", 2),
        ArcChain("two_rq_l", 2, inductor=True),
        # one arc with diffusion
        ArcChain("randles", 1, diffusion="W", log_wc_tau1=(-2.0, 2.0)),
        # two arcs with semi-infinite diffusion
        ArcChain("sei_randles", 2, diffusion="W", log_ww_over_wc=(-5.0, -1.0)),
        # same, with finite-space diffusion: an intercalation particle has a
        # blocking core, so |Z| stays finite as f -> 0 rather than diverging
        ArcChain("sei_randles_wo", 2, diffusion="Wo"),
    )
}


def get(name: str) -> TrainingCircuit:
    """Look up a trainable circuit by name."""
    try:
        return CIRCUITS[name]
    except KeyError:
        available = ", ".join(repr(n) for n in CIRCUITS)
        message = f"unknown circuit {name!r}, expected one of {available}"
        raise KeyError(message) from None
