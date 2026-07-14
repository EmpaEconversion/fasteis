"""Regression tests.

Compare against reference implementations in impedance.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from impedance.models.circuits import elements as ipy

import eis
from tests.circuit_cases import (
    COMPOSITION_CASES,
    ELEMENT_CASES,
    IMPEDANCEPY_ELEMENT_NAMES,
    CompositionCase,
    ElementParams,
    FreqArray,
)


def test_all_impedancepy_elements_are_covered() -> None:
    """Check all elements are in tests."""
    assert set(ELEMENT_CASES) == IMPEDANCEPY_ELEMENT_NAMES


FLAT_CASES: list[tuple[str, ElementParams, FreqArray]] = [
    (name, params, freqs)
    for name, variations in ELEMENT_CASES.items()
    for params, freqs in variations
]


@pytest.mark.parametrize(
    ("name", "params", "freqs"),
    FLAT_CASES,
    ids=[f"{name}{params}" for name, params, _ in FLAT_CASES],
)
def test_element_matches_impedancepy(
    name: str, params: ElementParams, freqs: FreqArray
) -> None:
    """Test individual elements."""
    circuit = getattr(eis.Circuit, name)(*params)
    got = np.array(circuit.impedance(list(freqs)))
    want = ipy.circuit_elements[name](list(params), list(freqs))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "case", COMPOSITION_CASES, ids=[case.label for case in COMPOSITION_CASES]
)
def test_composition_matches_impedancepy(case: CompositionCase) -> None:
    """Test circuits made of several elements."""
    got = np.array(case.eis_circuit.impedance(case.freqs_list))
    want = case.ipy_result(case.freqs_list)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
