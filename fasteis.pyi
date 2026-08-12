from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

class FitResult:
    circuit: Circuit
    params: dict[str, float]
    stderr: dict[str, float] | None
    success: bool
    iterations: int
    impedance_evaluations: int
    cost: float
    chi_square: float

class Circuit:
    @staticmethod
    def R(r: float) -> Circuit: ...
    @staticmethod
    def C(c: float) -> Circuit: ...
    @staticmethod
    def L(l: float) -> Circuit: ...
    @staticmethod
    def La(l: float, alpha: float) -> Circuit: ...
    @staticmethod
    def CPE(q: float, alpha: float) -> Circuit: ...
    @staticmethod
    def W(aw: float) -> Circuit: ...
    @staticmethod
    def Wo(z0: float, tau: float) -> Circuit: ...
    @staticmethod
    def Ws(z0: float, tau: float) -> Circuit: ...
    @staticmethod
    def G(rg: float, tg: float) -> Circuit: ...
    @staticmethod
    def Gs(rg: float, tg: float, phi: float) -> Circuit: ...
    @staticmethod
    def K(r: float, tau_k: float) -> Circuit: ...
    @staticmethod
    def Zarc(r: float, tau_k: float, gamma: float) -> Circuit: ...
    @staticmethod
    def TLMQ(r_ion: float, qs: float, gamma: float) -> Circuit: ...
    @staticmethod
    def T(a_coeff: float, b_coeff: float, a_param: float, b_param: float) -> Circuit: ...
    @staticmethod
    def series(elements: Sequence[Circuit]) -> Circuit: ...
    @staticmethod
    def parallel(elements: Sequence[Circuit]) -> Circuit: ...
    def __init__(self, s: str) -> None: ...
    @staticmethod
    def ml_circuits() -> list[str]: ...
    def guess(
        self,
        frequencies: Sequence[float],
        impedances: Sequence[complex],
        weights: str | None = None,
    ) -> list[float]: ...
    def param_names(self) -> list[str]: ...
    def with_values(self, values: Sequence[float]) -> Circuit: ...
    def with_named_values(self, values: dict[str, float]) -> Circuit: ...
    def impedance(self, frequencies: Sequence[float]) -> npt.NDArray[np.complex128]: ...
    def param_values(self) -> list[float]: ...
    def param_bounds(self) -> list[tuple[float, float]]: ...
    def param_units(self) -> list[str]: ...
    def residuals(
        self,
        params: Sequence[float],
        frequencies: Sequence[float],
        impedances: Sequence[complex],
        weight: Literal["modulus", "unit"] = "modulus",
    ) -> list[float]: ...
    def jacobian(
        self,
        params: Sequence[float],
        frequencies: Sequence[float],
        impedances: Sequence[complex],
        weight: Literal["modulus", "unit"] = "modulus",
    ) -> list[list[float]]: ...
    def fit(
        self,
        frequencies: Sequence[float],
        impedances: Sequence[complex],
        guess_init: bool | None = None,
        weights: str | None = None,
        weight: Literal["modulus", "unit"] = "modulus",
        method: Literal[
            "levenberg_marquardt",
            "particle_swarm",
            "nelder_mead",
            "differential_evolution",
            "simulated_annealing",
            "basin_hopping",
        ] = "levenberg_marquardt",
        max_iterations: int = 200,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        num_particles: int = 200,
        generations: int = 1000,
        nelder_mead_iterations: int = 2000,
        de_evaluations: int = 20_000,
        sa_iterations: int = 5000,
        sa_initial_temperature: float = 2.0,
        basin_hopping_hops: int = 20,
        basin_hopping_step_size: float = 1.0,
        basin_hopping_temperature: float = 1.0,
        seed: int | None = None,
    ) -> FitResult: ...
