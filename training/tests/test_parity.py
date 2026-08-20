# Copyright © 2026, Empa.
"""Ensure equivalent Python and Rust implementations agree.

- `TrainingCircuit.impedance_torch` reimplements the circuit maths from
  `elements.rs`/`circuit.rs` so the residual loss can backpropagate.
- `src/nn.rs` reimplements the forward pass, resampling and denormalisation
  so inference can run on pure Rust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import fasteis
from training import circuits, loss, priors, serialize_weights

TAU = 2.0 * np.pi
RTOL = 1e-10


ALL = pytest.mark.parametrize("circuit", list(circuits.CIRCUITS.values()), ids=lambda c: c.name)


def _reference(
    circuit: circuits.TrainingCircuit, params: np.ndarray, freqs: np.ndarray
) -> np.ndarray:
    built = fasteis.Circuit(circuit.circuit_str).with_values(list(params))
    return np.asarray(built.impedance(list(freqs)), dtype=np.complex128)


@ALL
@pytest.mark.parametrize("seed", range(5))
def test_torch_impedance_matches_fasteis_over_the_priors(
    circuit: circuits.TrainingCircuit, seed: int
) -> None:
    """Sample from the real priors rather than hand-picked values."""
    rng = np.random.default_rng(seed)
    spectra = priors.sample_many(rng, circuit, 40)

    for spectrum in spectra:
        w = torch.tensor(TAU * spectrum.freqs, dtype=torch.float64)
        params = torch.tensor(spectrum.params, dtype=torch.float64)
        got = circuit.impedance_torch(params, w).numpy()

        assert got.real == pytest.approx(spectrum.z_clean.real, rel=RTOL, abs=1e-300)
        assert got.imag == pytest.approx(spectrum.z_clean.imag, rel=RTOL, abs=1e-300)


@ALL
def test_batched_impedance_matches_per_sample(
    circuit: circuits.TrainingCircuit,
) -> None:
    rng = np.random.default_rng(7)
    spectra = priors.sample_many(rng, circuit, 16)
    freqs = np.logspace(-2, 6, 64)

    params = torch.tensor(np.array([s.params for s in spectra]), dtype=torch.float64)
    w = torch.tensor(TAU * freqs, dtype=torch.float64).expand(len(spectra), -1)
    batched = circuit.impedance_torch(params, w).numpy()

    for i, spectrum in enumerate(spectra):
        expected = _reference(circuit, spectrum.params, freqs)
        assert batched[i].real == pytest.approx(expected.real, rel=RTOL)
        assert batched[i].imag == pytest.approx(expected.imag, rel=RTOL)


@ALL
def test_modulus_residuals_match_fasteis(circuit: circuits.TrainingCircuit) -> None:
    """The loss must agree with the residual the optimiser itself minimises."""
    rng = np.random.default_rng(3)
    spectrum = priors.sample(rng, circuit)
    built = fasteis.Circuit(circuit.circuit_str)

    # deliberately off the truth, so the residuals are not all ~0
    nudge = np.full(circuit.n_params, 1.2)
    nudge[list(circuit.linear_params)] = 1.0  # keep exponents in range
    guess = spectrum.params * nudge
    expected = np.asarray(
        built.residuals(list(guess), list(spectrum.freqs), list(spectrum.z), "modulus")
    )

    got = loss.modulus_residuals(
        circuit,
        torch.tensor(guess, dtype=torch.float64),
        torch.tensor(TAU * spectrum.freqs, dtype=torch.float64),
        torch.tensor(spectrum.z, dtype=torch.complex128),
    ).numpy()

    assert got.reshape(-1) == pytest.approx(expected, rel=1e-9, abs=1e-12)


@ALL
def test_residual_loss_is_differentiable(circuit: circuits.TrainingCircuit) -> None:
    rng = np.random.default_rng(11)
    spectrum = priors.sample(rng, circuit)

    params = torch.tensor(spectrum.params, dtype=torch.float64, requires_grad=True)
    value = loss.residual_loss(
        circuit,
        params,
        torch.tensor(TAU * spectrum.freqs, dtype=torch.float64),
        torch.tensor(spectrum.z, dtype=torch.complex128),
    )
    value.backward()

    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


def _torch_from_weights(name: str, path: Path):
    """Rebuild the torch network and its target statistics from an exported file."""
    from training import model, train

    circuit = circuits.get(name)
    metadata, tensors = serialize_weights.read(path)
    config = model.Config(
        channels=int(metadata["channels"]),
        blocks=len(metadata["dilations"].split(",")),
        head_width=tensors["w.head.0.weight"].shape[0],
        groups=int(metadata["groups"]),
    )
    net = model.GuessNet(circuit.n_params, config)
    net.load_state_dict(
        {
            key.removeprefix("w."): torch.tensor(value, dtype=torch.float32)
            for key, value in tensors.items()
            if key.startswith("w.")
        }
    )
    net.eval()
    device = torch.device("cpu")
    std = train.Standardiser(tensors["target_mean"], tensors["target_std"], device)
    return circuit, net, std, device


@pytest.mark.parametrize("name", list(circuits.CIRCUITS))
def test_rust_guess_matches_the_torch_network(name: str) -> None:
    """src/nn.rs reimplements the forward pass, resampling and denormalisation.

    Compares the whole guess to also catch any drift in the scaling rules.
    """
    from training import train

    weights = Path("src/models") / f"{name}.eisnn"
    if not weights.exists():
        pytest.skip(f"{name} has no exported weights")

    circuit, net, std, device = _torch_from_weights(name, weights)

    built = fasteis.Circuit(circuit.circuit_str)
    for spectrum in priors.sample_many(np.random.default_rng(21), circuit, 200):
        expected = train.guess(circuit, net, std, device, spectrum.freqs, spectrum.z)
        got = np.array(built.guess(list(spectrum.freqs), list(spectrum.z)))
        # remaining gap is torch computing in f32 against rust's f64
        assert got == pytest.approx(expected, rel=1e-4)
