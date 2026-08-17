"""Circuit element/circuit definitions used in tests and benchmarks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from impedance.models.circuits import elements as ipy
from numpy.typing import NDArray

import fasteis

FreqArray = NDArray[np.float64]
ElementParams = tuple[float, ...]
ElementCase = tuple[ElementParams, FreqArray]
IpyResultFn = Callable[[list[float]], NDArray[np.complex128]]

FREQS: FreqArray = np.logspace(-2, 6, 50)

# impedance.py spells these two differently from our Element variants
ELEMENT_ALIASES = {"CPE": "Cpe", "TLMQ": "Tlmq"}


def element(name: str, params: ElementParams) -> fasteis.Element:
    """Build an element from its impedance.py name."""
    return getattr(fasteis, ELEMENT_ALIASES.get(name, name))(*params)


# numpys tanh/sinh/cosh overflows with large numbers, and result becomes NaN
# the rust implementation is stable
# NARROW frequency is used wherever a full 1e-2..1e6 Hz sweep would not be safe
NARROW: FreqArray = np.logspace(-2, 3, 50)

# name -> list of (params, freqs) variations to check. `params` is the one
# element(name, params) unpacks it positionally
# impedance.py's functions take the same values as a list
ELEMENT_CASES: dict[str, list[ElementCase]] = {
    "R": [
        ((100.0,), FREQS),
        ((0.001,), FREQS),
        ((1e6,), FREQS),
    ],
    "C": [
        ((1e-6,), FREQS),
        ((1e-9,), FREQS),
        ((1.0,), FREQS),
    ],
    "L": [
        ((2e-3,), FREQS),
        ((1e-9,), FREQS),
        ((5.0,), FREQS),
    ],
    "La": [
        ((1e-3, 0.9), FREQS),
        ((1.0, 0.5), FREQS),
        ((1e-6, 1.2), FREQS),
    ],
    "CPE": [
        ((1e-5, 0.8), FREQS),
        ((1e-3, 0.5), FREQS),
        ((1e-6, 1.0), FREQS),
    ],
    "W": [
        ((50.0,), FREQS),
        ((1.0,), FREQS),
        ((1000.0,), FREQS),
    ],
    "Wo": [
        ((100.0, 1.0), NARROW),
        ((10.0, 0.01), NARROW),
        ((500.0, 5.0), NARROW),
    ],
    "Ws": [
        ((100.0, 1.0), NARROW),
        ((10.0, 0.01), NARROW),
        ((500.0, 5.0), NARROW),
    ],
    "G": [
        ((50.0, 0.1), FREQS),
        ((10.0, 1.0), FREQS),
        ((200.0, 0.001), FREQS),
    ],
    "Gs": [
        ((50.0, 0.1, 0.5), FREQS),
        ((10.0, 1.0, 2.0), NARROW),
        ((200.0, 0.001, 0.1), FREQS),
    ],
    "K": [
        ((50.0, 0.01), FREQS),
        ((10.0, 1.0), FREQS),
        ((500.0, 1e-4), FREQS),
    ],
    "Zarc": [
        ((50.0, 0.01, 0.8), FREQS),
        ((10.0, 1.0, 0.5), FREQS),
        ((500.0, 1e-4, 1.0), FREQS),
    ],
    "TLMQ": [
        ((10.0, 1e-4, 0.9), FREQS),
        ((5.0, 1e-6, 0.5), FREQS),
        ((20.0, 1e-3, 1.0), FREQS),
    ],
    "T": [
        ((1.0, 2.0, 0.5, 0.1), NARROW),
        ((0.5, 0.5, 1.0, 0.01), NARROW),
        ((2.0, 1.0, 0.1, 0.05), NARROW),
    ],
}

# This is used om test_against_impedance to ensure every element is tested
IMPEDANCEPY_ELEMENT_NAMES: set[str] = {
    name for name in ipy.circuit_elements if name not in ("s", "p")
}


class CompositionCase:
    """Circuit with both an eis and impedance representations."""

    def __init__(
        self,
        label: str,
        eis_circuit: fasteis.Circuit,
        ipy_result: IpyResultFn,
        freqs: FreqArray,
    ) -> None:
        self.label = label
        self.eis_circuit = eis_circuit
        self.ipy_result = ipy_result
        self.freqs = freqs
        self.freqs_list: list[float] = list(freqs)


def _make_series_r_parallel_r_cpe() -> CompositionCase:
    r0, r1, q, alpha = 50.0, 200.0, 1e-5, 0.85
    circuit = fasteis.Series(
        [
            fasteis.R(r0),
            fasteis.Parallel([fasteis.R(r1), fasteis.Cpe(q, alpha)]),
        ]
    )

    def ipy_result(freqs_list: list[float]) -> NDArray[np.complex128]:
        return np.asarray(
            ipy.s(
                [
                    ipy.R([r0], freqs_list),
                    ipy.p([ipy.R([r1], freqs_list), ipy.CPE([q, alpha], freqs_list)]),
                ]
            ),
            dtype=np.complex128,
        )

    return CompositionCase("series_r_parallel_r_cpe", circuit, ipy_result, FREQS)


def _make_nested_parallel_of_series() -> CompositionCase:
    ra, ca, rb, aw = 10.0, 1e-5, 20.0, 30.0
    branch_a = fasteis.Series([fasteis.R(ra), fasteis.C(ca)])
    branch_b = fasteis.Series([fasteis.R(rb), fasteis.W(aw)])
    circuit = fasteis.Parallel([branch_a, branch_b])

    def ipy_result(freqs_list: list[float]) -> NDArray[np.complex128]:
        return np.asarray(
            ipy.p(
                [
                    ipy.s([ipy.R([ra], freqs_list), ipy.C([ca], freqs_list)]),
                    ipy.s([ipy.R([rb], freqs_list), ipy.W([aw], freqs_list)]),
                ]
            ),
            dtype=np.complex128,
        )

    return CompositionCase("nested_parallel_of_series", circuit, ipy_result, FREQS)


def _make_three_branch_parallel() -> CompositionCase:
    """Resistors and capacitor in parallel."""
    r0, r1, c = 10.0, 20.0, 1e-6
    circuit = fasteis.Parallel(
        [
            fasteis.R(r0),
            fasteis.R(r1),
            fasteis.C(c),
        ]
    )

    def ipy_result(freqs_list: list[float]) -> NDArray[np.complex128]:
        return np.asarray(
            ipy.p(
                [
                    ipy.R([r0], freqs_list),
                    ipy.R([r1], freqs_list),
                    ipy.C([c], freqs_list),
                ]
            ),
            dtype=np.complex128,
        )

    return CompositionCase("three_branch_parallel", circuit, ipy_result, FREQS)


def _make_randles() -> CompositionCase:
    """Define a randles circuit."""
    rs, rct, cdl_val, aw = 20.0, 150.0, 20e-6, 60.0
    circuit = fasteis.Series(
        [
            fasteis.R(rs),
            fasteis.Parallel(
                [
                    fasteis.Series([fasteis.R(rct), fasteis.W(aw)]),
                    fasteis.C(cdl_val),
                ]
            ),
        ]
    )

    def ipy_result(freqs_list: list[float]) -> NDArray[np.complex128]:
        return np.asarray(
            ipy.s(
                [
                    ipy.R([rs], freqs_list),
                    ipy.p(
                        [
                            ipy.s([ipy.R([rct], freqs_list), ipy.W([aw], freqs_list)]),
                            ipy.C([cdl_val], freqs_list),
                        ]
                    ),
                ]
            ),
            dtype=np.complex128,
        )

    return CompositionCase("randles", circuit, ipy_result, FREQS)


def _make_randles_cpe() -> CompositionCase:
    """Randles circuit with CPE instead of double layer capacitor."""
    rs, rct, q, alpha, aw = 15.0, 300.0, 5e-5, 0.9, 45.0
    circuit = fasteis.Series(
        [
            fasteis.R(rs),
            fasteis.Parallel(
                [
                    fasteis.Series([fasteis.R(rct), fasteis.W(aw)]),
                    fasteis.Cpe(q, alpha),
                ]
            ),
        ]
    )

    def ipy_result(freqs_list: list[float]) -> NDArray[np.complex128]:
        return np.asarray(
            ipy.s(
                [
                    ipy.R([rs], freqs_list),
                    ipy.p(
                        [
                            ipy.s([ipy.R([rct], freqs_list), ipy.W([aw], freqs_list)]),
                            ipy.CPE([q, alpha], freqs_list),
                        ]
                    ),
                ]
            ),
            dtype=np.complex128,
        )

    return CompositionCase("randles_cpe", circuit, ipy_result, FREQS)


COMPOSITION_CASES: list[CompositionCase] = [
    _make_series_r_parallel_r_cpe(),
    _make_nested_parallel_of_series(),
    _make_three_branch_parallel(),
    _make_randles(),
    _make_randles_cpe(),
]
