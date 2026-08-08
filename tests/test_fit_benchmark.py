"""Regression tests for Circuit.fit() against a few known real datasets.

Update KNOWN_GOOD_CASES with new costs if they improve.
"""

from __future__ import annotations

import pytest

import fasteis
from tests.fit_benchmark_cases import FIT_BENCHMARK_CASES

_CASES_BY_LABEL = {c.label: c for c in FIT_BENCHMARK_CASES}

# Known datasets and their best-fit cost
KNOWN_GOOD_CASES: list[tuple[str, float]] = [
    ("reda_20338409_1", 0.0010679934002571462),
    ("reda_20338409_31", 0.0010071989300080302),
    ("kiye_19107066_60", 0.08223),
    ("kiye_19107066_130", 0.04061),
]


@pytest.mark.parametrize(
    "label,expected_cost",
    KNOWN_GOOD_CASES,
    ids=[label for label, _ in KNOWN_GOOD_CASES],
)
def test_fit_matches_recorded_cost(label: str, expected_cost: float) -> None:
    case = _CASES_BY_LABEL[label]
    freqs, z = case.load_data()

    circuit = fasteis.Circuit(case.circuit_string).with_named_values(case.eis_initial_guess)
    result = circuit.fit(list(freqs), list(z))

    assert result.success
    assert result.cost == pytest.approx(expected_cost, rel=1e-3)
