from collections.abc import Iterator, Sequence
from typing import Literal, Protocol, SupportsComplex, SupportsFloat

import numpy as np
import numpy.typing as npt

class _FloatArray(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[SupportsFloat]: ...

class _ComplexArray(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[SupportsComplex | SupportsFloat]: ...

class _DataFrame(Protocol):
    def __getitem__(self, key: str, /) -> _FloatArray: ...

class Element:
    """A single circuit element.

    Each variant is a subclass taking its fields as arguments.
    They are combined with `Series()` and `Parallel()`.
    """

    class R(Element):
        """Ideal resistor, `Z = r`.

        Args:
            r: Resistance in ohm.
        """

        r: float

        def __init__(self, r: float) -> None: ...

    class C(Element):
        """Ideal capacitor, `Z = 1 / (jwc)`.

        Args:
            c: Capacitance in F.
        """

        c: float

        def __init__(self, c: float) -> None: ...

    class L(Element):
        """Ideal inductor, `Z = jwl`.

        Args:
            l: Inductance in H.
        """

        l: float

        def __init__(self, l: float) -> None: ...

    class La(Element):
        """Modified inductance, `Z = (jwl) ** alpha`.

        Reduces to an ideal inductor at `alpha = 1`.

        Args:
            l: Inductance in `H*s`.
            alpha: Fractional exponent in [0, 1].
        """

        l: float
        alpha: float

        def __init__(self, l: float, alpha: float) -> None: ...

    class Cpe(Element):
        """Constant phase element, `Z = 1 / (q * (jw) ** alpha)`.

        Reduces to a capacitor of `q` farads at `alpha = 1`, and to a resistor
        of `1 / q` ohm at `alpha = 0`.

        Args:
            q: Admittance prefactor in `ohm^-1*s^alpha`.
            alpha: Fractional exponent in [0, 1].
        """

        q: float
        alpha: float

        def __init__(self, q: float, alpha: float) -> None: ...

    class W(Element):
        """Semi-infinite Warburg diffusion, `Z = aw * (1 - j) / sqrt(w)`.

        Args:
            aw: Warburg coefficient in `ohm*s^-0.5`.
        """

        aw: float

        def __init__(self, aw: float) -> None: ...

    class Wo(Element):
        """Finite-length open Warburg, `Z = z0 / (x * tanh(x))` with `x = sqrt(jw*tau)`.

        The reflecting boundary case, for diffusion into a blocked layer.

        Args:
            z0: Impedance scale in ohm.
            tau: Diffusion time constant in s.
        """

        z0: float
        tau: float

        def __init__(self, z0: float, tau: float) -> None: ...

    class Ws(Element):
        """Finite-length short Warburg, `Z = z0 * tanh(x) / x` with `x = sqrt(jw*tau)`.

        The transmissive boundary case, for diffusion into a reservoir.

        Args:
            z0: Impedance scale in ohm.
            tau: Diffusion time constant in s.
        """

        z0: float
        tau: float

        def __init__(self, z0: float, tau: float) -> None: ...

    class G(Element):
        """Gerischer element, `Z = rg / sqrt(1 + jw*tg)`.

        For diffusion coupled to a homogeneous chemical reaction. Reduces to a
        resistor at `tg = 0`.

        Args:
            rg: Gerischer resistance in ohm.
            tg: Reaction time constant in s.
        """

        rg: float
        tg: float

        def __init__(self, rg: float, tg: float) -> None: ...

    class Gs(Element):
        """Finite-length Gerischer, `Z = rg / (s * tanh(s*phi))` with `s = sqrt(1 + jw*tg)`.

        Args:
            rg: Gerischer resistance in ohm.
            tg: Reaction time constant in s.
            phi: Dimensionless layer-thickness factor.
        """

        rg: float
        tg: float
        phi: float

        def __init__(self, rg: float, tg: float, phi: float) -> None: ...

    class K(Element):
        """Debye relaxation, `Z = r / (1 + jw*tau_k)`.

        A resistor and capacitor in parallel, as a single element.

        Args:
            r: Polarisation resistance in ohm.
            tau_k: Relaxation time constant in s.
        """

        r: float
        tau_k: float

        def __init__(self, r: float, tau_k: float) -> None: ...

    class Zarc(Element):
        """Depressed semicircle, `Z = r / (1 + (jw*tau_k) ** gamma)`.

        A resistor and constant phase element in parallel, as a single element.
        Reduces to `K` at `gamma = 1`.

        Args:
            r: Polarisation resistance in ohm.
            tau_k: Relaxation time constant in s.
            gamma: Depression exponent in [0, 1].
        """

        r: float
        tau_k: float
        gamma: float

        def __init__(self, r: float, tau_k: float, gamma: float) -> None: ...

    class Tlmq(Element):
        """Transmission line with a constant phase element surface impedance.

        `Z = sqrt(r_ion*zs) / tanh(sqrt(r_ion/zs))` for a distributed pore of
        surface impedance `zs = 1 / (qs * (jw) ** gamma)`.

        Args:
            r_ion: Ionic pore resistance in ohm.
            qs: Surface admittance prefactor in `F*s^(gamma-1)`.
            gamma: Fractional exponent in [0, 1].
        """

        r_ion: float
        qs: float
        gamma: float

        def __init__(self, r_ion: float, qs: float, gamma: float) -> None: ...

    class T(Element):
        """General transmission line.

        `Z = a_coeff*coth(beta)/beta + b_coeff*cosech(beta)/beta`, where
        `beta = sqrt(a_param + jw*b_param)`.

        Args:
            a_coeff: Coefficient of the coth term, in `ohm*m^2`.
            b_coeff: Coefficient of the cosech term, in `ohm*m^2`.
            a_param: Dimensionless offset inside the square root.
            b_param: Time constant inside the square root, in s.
        """

        a_coeff: float
        b_coeff: float
        a_param: float
        b_param: float

        def __init__(
            self, a_coeff: float, b_coeff: float, a_param: float, b_param: float
        ) -> None: ...

class FitResult:
    """Outcome of `Circuit.fit()`."""

    circuit: Circuit
    """The fitted circuit, carrying the optimised parameter values."""
    params: dict[str, float]
    """Fitted values keyed by `Circuit.param_names()`."""
    stderr: dict[str, float] | None
    """Standard errors of parameters, or None when they cannot be estimated.

    Taken from the diagonal of the inverted Gauss-Newton matrix, scaled by
    `chi_square` for each degree of freedom. None when the fit has no spare
    degrees of freedom, or when that matrix is singular.
    """
    success: bool
    """Whether the optimiser reported convergence."""
    iterations: int
    """Optimiser iterations, counting residual calls only."""
    impedance_evaluations: int
    """Full impedance sweeps spent, including Jacobians and restarts."""
    cost: float
    """Half the sum of squared weighted residuals, the quantity minimised."""
    chi_square: float
    """Sum of squared weighted residuals, `2 * cost`."""

class Circuit:
    """An equivalent circuit: elements in series and parallel, with their values.

    Build one by parsing a topology string, or by composing elements with
    `Series()` and `Parallel()`.
    """

    def __init__(self, s: str) -> None:
        """Parse a circuit topology string, e.g. `"R0-(R1,CPE1)"`.

        The string carries no parameter values, so every element starts at a
        placeholder default. Set real values with `with_values()`,
        `with_named_values()`, or `fit(guess_init=True)`.

        Also accepts the name of a built-in circuit, e.g. `"randles"`; see
        `ml_circuits()`.

        Args:
            s: Topology string, or the name of a built-in circuit.

        Raises:
            ValueError: The string is not valid circuit syntax.
        """

    @staticmethod
    def ml_circuits() -> list[str]:
        """Built-in circuit names that have trained initial-parameter models.

        These get a guessed starting point from `fit()` by default.
        """

    def guess(
        self,
        frequencies: _FloatArray | _DataFrame,
        impedances: _ComplexArray | None = None,
        weights: str | None = None,
    ) -> list[float]:
        """Machine-learning guess of starting parameters for this topology.

        Series and parallel elements may be written in any order and with any
        labels: `(R1,C2)-R3` uses the same model as `R0-(R1,C1)`.

        Args:
            frequencies: Frequencies in Hz, or a battery data format dataframe
                with frequencies and impedances.
            impedances: Measured complex impedances in ohm, one per frequency.
                Leave as None when passing a dataframe.
            weights: Path to a custom `.eisnn` file to load, instead of looking
                for a bundled model.

        Returns:
            Starting values, in `param_names()` order.

        Raises:
            ValueError: No model has been trained for this topology, or
                `weights` was trained for a different one.
        """

    def param_names(self) -> list[str]:
        """Parameter names, in the order used throughout this class.

        `with_values()` consumes values in this order, and
        `with_named_values()` expects these as keys.
        """

    def with_values(self, values: _FloatArray) -> Circuit:
        """Rebuild this circuit with a new flat parameter vector.

        Args:
            values: One value per parameter, in `param_names()` order.

        Raises:
            ValueError: The wrong number of values was supplied.
        """

    def with_named_values(self, values: dict[str, float]) -> Circuit:
        """Rebuild this circuit with parameter values looked up by name.

        Args:
            values: Values keyed by `param_names()`. Every name must be
                present, and no unknown names may be supplied.

        Raises:
            ValueError: Names are missing or unrecognised.
        """

    def impedance(self, frequencies: _FloatArray | _DataFrame) -> npt.NDArray[np.complex128]:
        """Impedance of this circuit at each frequency.

        Args:
            frequencies: Frequencies in Hz, or a battery data format dataframe
                to take the frequency column from.

        Returns:
            Complex impedances in ohm, one per frequency.
        """

    def param_values(self) -> list[float]:
        """Current parameter values, in `param_names()` order."""

    def param_bounds(self) -> list[tuple[float, float]]:
        """Physical-validity `(lo, hi)` bounds, in `param_names()` order.

        Fractional exponents are bounded to `[0, 1]`; every other parameter is
        a positive magnitude with `hi` set to `inf`.
        """

    def param_units(self) -> list[str]:
        """Physical units, in `param_names()` order. `"-"` means dimensionless."""

    def residuals(
        self,
        params: _FloatArray,
        frequencies: _FloatArray | _DataFrame,
        impedances: _ComplexArray | None = None,
        weight: Literal["modulus", "unit"] = "modulus",
    ) -> list[float]:
        """Weighted residual vector for an arbitrary parameter vector.

        The same building block `fit()` uses internally, exposed so that an
        external optimisers such as `scipy.optimize.least_squares` can be used.

        Args:
            params: Parameter values, in `param_names()` order.
            frequencies: Frequencies in Hz, or a battery data format dataframe
                carrying the whole spectrum.
            impedances: Measured complex impedances in ohm. Leave as None when
                passing a dataframe.
            weight: Divide each residual by the modulus of the measured point,
                or leave it unweighted.

        Returns:
            Real and imaginary parts interleaved, `[re0, im0, re1, im1, ...]`.
        """

    def jacobian(
        self,
        params: _FloatArray,
        frequencies: _FloatArray | _DataFrame,
        impedances: _ComplexArray | None = None,
        weight: Literal["modulus", "unit"] = "modulus",
    ) -> list[list[float]]:
        """Central-difference Jacobian of `residuals()` at `params`.

        Args:
            params: Parameter values, in `param_names()` order.
            frequencies: Frequencies in Hz, or a battery data format dataframe
                with frequencies and impedances.
            impedances: Measured complex impedances in ohm. Leave as None when
                passing a dataframe.
            weight: Weighting scheme, matching `residuals()`.

        Returns:
            Shape `(2 * len(frequencies), len(params))`, where rows are
            residuals and columns are parameters, as
            `scipy.optimize.least_squares(jac=...)` expects.
        """

    def fit(
        self,
        frequencies: _FloatArray | _DataFrame,
        impedances: _ComplexArray | None = None,
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
    ) -> FitResult:
        """Fit this circuit's parameters to a measured spectrum.

        Args:
            frequencies: Frequencies in Hz, or a battery data format dataframe
                with frequencies and impedances.
            impedances: Measured complex impedances in ohm. Leave as None when
                passing a dataframe.
            guess_init: Left unset, start from a machine-learning guess
                whenever the circuit was not given explicit starting values,
                warning and falling back to placeholders if no model has been
                trained. True always guesses, and raises rather than warns when
                it cannot. False never guesses.
            weights: Path to a custom `.eisnn` file to guess with, instead of
                looking for a bundled model.
            weight: Divide each residual by the modulus of the measured point,
                or leave it unweighted.
            method: Optimiser to run.
            max_iterations: Iteration cap for `levenberg_marquardt`.
            ftol: Cost-change convergence tolerance for `levenberg_marquardt`.
            xtol: Parameter-change convergence tolerance for
                `levenberg_marquardt`.
            num_particles: Swarm size for `particle_swarm`.
            generations: Generation cap for `particle_swarm`.
            nelder_mead_iterations: Iteration cap for `nelder_mead`.
            de_evaluations: Evaluation budget for `differential_evolution`.
            sa_iterations: Iteration cap for `simulated_annealing`.
            sa_initial_temperature: Starting temperature for
                `simulated_annealing`.
            basin_hopping_hops: Number of hops for `basin_hopping`.
            basin_hopping_step_size: Perturbation size per hop.
            basin_hopping_temperature: Acceptance temperature for
                `basin_hopping`.
            seed: Fixes the random draws of the stochastic methods.

        Returns:
            The fitted parameters, their uncertainties, and fit diagnostics.
        """

# Element variants, re-exported so they can be written as `fasteis.R(100.0)`.
R = Element.R
C = Element.C
L = Element.L
La = Element.La
Cpe = Element.Cpe
W = Element.W
Wo = Element.Wo
Ws = Element.Ws
G = Element.G
Gs = Element.Gs
K = Element.K
Zarc = Element.Zarc
Tlmq = Element.Tlmq
T = Element.T

def Series(parts: Sequence[Element | Circuit]) -> Circuit:
    """Connect elements and circuits in series, summing their impedances.

    Args:
        parts: Elements, or circuits to nest, in order.
    """

def Parallel(parts: Sequence[Element | Circuit]) -> Circuit:
    """Connect elements and circuits in parallel, summing their admittances.

    Args:
        parts: Elements, or circuits to nest, one per branch.
    """
