# Copyright © 2026, Empa.
"""The public guessing API: circuit lookup, guess_init, and the weight container."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fasteis
from training import circuits, priors, serialize_weights

RANDLES = circuits.get("randles")


def test_randles_is_registered() -> None:
    assert "randles" in fasteis.Circuit.ml_circuits()


def test_alias_builds_the_trained_topology() -> None:
    assert fasteis.Circuit("randles").param_names() == list(RANDLES.param_names)


@pytest.mark.parametrize("name", fasteis.Circuit.ml_circuits())
def test_every_registered_circuit_is_trainable_and_guesses(name: str) -> None:
    """Each registry row must name a training circuit and guess within its bounds."""
    circuit = circuits.get(name)
    built = fasteis.Circuit(name)
    assert built.param_names() == list(circuit.param_names)

    spectrum = priors.sample(np.random.default_rng(21), circuit)
    guess = built.guess(list(spectrum.freqs), list(spectrum.z))

    assert len(guess) == circuit.n_params
    assert np.all(np.isfinite(guess))
    for value, (lo, hi) in zip(guess, built.param_bounds(), strict=True):
        assert lo <= value <= hi


@pytest.mark.parametrize("name", fasteis.Circuit.ml_circuits())
def test_every_registered_circuit_fits_from_its_guess(name: str) -> None:
    """guess_init must reach the noise floor on a clean spectrum."""
    circuit = circuits.get(name)
    spectrum = priors.sample(np.random.default_rng(5), circuit)
    f, z = list(spectrum.freqs), list(spectrum.z_clean)

    result = fasteis.Circuit(name).fit(f, z, guess_init=True)
    assert result.success
    assert result.chi_square < 1e-12


def test_guess_init_starts_the_fit_from_the_guess() -> None:
    spectrum = priors.sample(np.random.default_rng(99), RANDLES)
    f, z = list(spectrum.freqs), list(spectrum.z)
    circuit = fasteis.Circuit("randles")

    from_guess = circuit.fit(f, z, guess_init=True)
    assert from_guess.success

    # same answer as passing guess() in by hand
    by_hand = circuit.with_values(circuit.guess(f, z)).fit(f, z)
    assert from_guess.iterations == by_hand.iterations
    assert from_guess.chi_square == pytest.approx(by_hand.chi_square, rel=1e-12)


def test_guess_init_recovers_the_true_parameters() -> None:
    spectrum = priors.sample(np.random.default_rng(3), RANDLES)
    result = fasteis.Circuit("randles").fit(list(spectrum.freqs), list(spectrum.z), guess_init=True)

    assert result.success
    for name, truth in zip(RANDLES.param_names, spectrum.params, strict=True):
        assert result.params[name] == pytest.approx(truth, rel=0.25)


def test_topology_matches_regardless_of_labels() -> None:
    """The registry matches on structure, so element labels are irrelevant."""
    spectrum = priors.sample(np.random.default_rng(11), RANDLES)
    f, z = list(spectrum.freqs), list(spectrum.z)

    named = fasteis.Circuit("randles").guess(f, z)
    relabelled = fasteis.Circuit("R7-(R9-W4,CPE2)").guess(f, z)

    assert relabelled == pytest.approx(named, rel=1e-12)


@pytest.mark.parametrize(
    "topology",
    ["R0-(CPE1,R1-W1)", "(CPE1,W1-R1)-R0", "(R1-W1,CPE1)-R0"],
)
def test_topology_matches_regardless_of_element_order(topology: str) -> None:
    """Series and parallel elements commute, and the guess is reordered to suit."""
    spectrum = priors.sample(np.random.default_rng(11), RANDLES)
    f, z = list(spectrum.freqs), list(spectrum.z)

    reference = fasteis.Circuit("randles")
    expected = dict(zip(reference.param_names(), reference.guess(f, z), strict=True))

    circuit = fasteis.Circuit(topology)
    guess = dict(zip(circuit.param_names(), circuit.guess(f, z), strict=True))

    assert guess == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "topology",
    [
        "R0-C1",  # nothing like it
        "R0-(C1,R1-W1)",  # C where the model wants CPE
        "R0-CPE1-R1-W1",  # same elements, no parallel
        "(R0-R1-W1,CPE1)",  # same elements, R0 moved inside the branch
    ],
)
def test_untrained_topologies_report_what_is_available(topology: str) -> None:
    spectrum = priors.sample(np.random.default_rng(1), RANDLES)
    circuit = fasteis.Circuit(topology)

    with pytest.raises(ValueError) as excinfo:
        circuit.fit(list(spectrum.freqs), list(spectrum.z), guess_init=True)

    message = str(excinfo.value)
    assert message.startswith("No training data on this circuit.")
    assert "'randles'" in message


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        fasteis.Circuit("randles").guess([1.0, 2.0, 3.0], [1 + 1j, 2 + 2j])


def test_too_few_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fasteis.Circuit("randles").guess([1.0], [1 + 1j])


def test_unsorted_frequencies_give_the_same_guess() -> None:
    """Resampling sorts internally; callers should not have to."""
    spectrum = priors.sample(np.random.default_rng(4), RANDLES)
    order = np.random.default_rng(0).permutation(len(spectrum.freqs))
    circuit = fasteis.Circuit("randles")

    ascending = circuit.guess(list(spectrum.freqs), list(spectrum.z))
    shuffled = circuit.guess(list(spectrum.freqs[order]), list(spectrum.z[order]))

    assert shuffled == pytest.approx(ascending, rel=1e-12)


def test_format_round_trips(tmp_path) -> None:
    path = tmp_path / "round.eisnn"
    tensors = {
        "a": np.arange(6, dtype=np.float64).reshape(2, 3),
        "b": np.array([1.5, -2.5]),
    }
    metadata = {"circuit": "R0-C1", "note": "unicode ok: µΩ"}
    serialize_weights.write(path, metadata, tensors)

    got_meta, got_tensors = serialize_weights.read(path)
    assert got_meta == metadata
    assert set(got_tensors) == set(tensors)
    for name, array in tensors.items():
        assert got_tensors[name].shape == array.shape
        assert got_tensors[name] == pytest.approx(array, rel=1e-6)


@pytest.mark.parametrize("dtype", ["f32", "f16", "int8"])
def test_every_dtype_round_trips(tmp_path, dtype: str) -> None:
    """The writer and its reader must agree for each encoding."""
    rng = np.random.default_rng(0)
    tensors = {
        "small": rng.normal(size=(4, 3, 5)) * 0.1,
        "wide": rng.normal(size=64) * 8.0,
        "zeros": np.zeros(7),  # int8 scale must not divide by zero
    }
    path = tmp_path / f"{dtype}.eisnn"
    serialize_weights.write(path, {"circuit": "R0-C1"}, tensors, dtype=dtype)

    _, got = serialize_weights.read(path)
    # int8 keeps ~2 significant digits relative to each tensor's peak
    tol = {"f32": 1e-6, "f16": 1e-3, "int8": 5e-3}[dtype]
    for name, array in tensors.items():
        peak = max(float(np.max(np.abs(array))), 1e-30)
        assert np.max(np.abs(got[name] - array)) <= tol * peak


def test_unknown_dtype_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown dtype"):
        serialize_weights.write(tmp_path / "x.eisnn", {}, {"a": np.zeros(2)}, dtype="f8")


@pytest.mark.parametrize("dtype", ["f16", "int8"])
def test_only_weights_are_quantised(tmp_path, dtype: str) -> None:
    """When saving a model, --dtype only affects weights.

    No point reducing precision on scales, stats etc. as it costs a lot of
    accuracy, at the weights themselves take up most of the file size.
    """
    stats = np.array([-0.805, -0.416, 0.775, -0.063, -0.784])
    tensors = {"w.layer.weight": np.linspace(-1.0, 1.0, 64), "target_mean": stats}

    path = tmp_path / f"{dtype}.eisnn"
    serialize_weights.write(path, {"circuit": "R0-C1"}, tensors, dtype=dtype)
    _, got = serialize_weights.read(path)

    assert got["target_mean"].tolist() == stats.astype(np.float32).tolist()
    # the weights did go through the lossy encoding
    assert not np.array_equal(got["w.layer.weight"], tensors["w.layer.weight"])


def test_evaluation_sets_are_reproducible() -> None:
    """Same calls give the same spectra so that runs are comparable."""
    from training import evaluate

    a, b = evaluate.benchmark_set(RANDLES, 20), evaluate.benchmark_set(RANDLES, 20)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.params, y.params)
        assert np.array_equal(x.z, y.z)


def test_a_longer_evaluation_set_extends_the_shorter_one() -> None:
    """benchmark_set(2000) must start with benchmark_set(300).

    Small runs contain the same tests as large runs, so they stay comparable.
    """
    from training import evaluate

    short, long = (
        evaluate.benchmark_set(RANDLES, 10),
        evaluate.benchmark_set(RANDLES, 40),
    )
    for x, y in zip(short, long[:10], strict=True):
        assert np.array_equal(x.params, y.params)
        assert np.array_equal(x.freqs, y.freqs)
        assert np.array_equal(x.z, y.z)


def test_weights_path_reproduces_the_embedded_model() -> None:
    """`weights=` must give the same answer as the model built into the crate."""
    weights = Path("src/models/randles.eisnn")
    if not weights.exists():
        pytest.skip("no exported weights")

    circuit = fasteis.Circuit("randles")
    for spectrum in priors.sample_many(np.random.default_rng(8), RANDLES, 20):
        f, z = list(spectrum.freqs), list(spectrum.z)
        assert circuit.guess(f, z, weights=str(weights)) == pytest.approx(
            circuit.guess(f, z), rel=1e-12
        )


def test_weights_trained_for_another_circuit_are_rejected() -> None:
    """Loading two_rq_l weights into a randles circuit must not silently work."""
    other = Path("src/models/two_rq_l.eisnn")
    if not other.exists():
        pytest.skip("no two_rq_l weights")

    spectrum = priors.sample(np.random.default_rng(6), RANDLES)
    with pytest.raises(ValueError, match="different circuit"):
        fasteis.Circuit("randles").guess(list(spectrum.freqs), list(spectrum.z), weights=str(other))
