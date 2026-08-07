"""Ensure equivalent Python and Rust implementations agree.

- `randles_torch.py` reimplements the circuit maths from `elements.rs`/
  `circuits.rs`so the residual loss can backpropagate.
- `src/nn.rs` reimplements the forward pass, resampling and denormalisation
  so inference can run on pure Rust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import fasteis
from training import circuits, priors, randles_torch, serialize_weights

TAU = 2.0 * np.pi
RTOL = 1e-10


def _reference(params: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    circuit = fasteis.Circuit(circuits.CIRCUIT_STRING).with_values(list(params))
    return np.asarray(circuit.impedance(list(freqs)), dtype=np.complex128)


@pytest.mark.parametrize("seed", range(5))
def test_torch_impedance_matches_fasteis_over_the_priors(seed: int) -> None:
    """Sample from the real priors rather than hand-picked values."""
    rng = np.random.default_rng(seed)
    spectra = priors.sample_many(rng, 40)

    for spectrum in spectra:
        w = torch.tensor(TAU * spectrum.freqs, dtype=torch.float64)
        params = torch.tensor(spectrum.params, dtype=torch.float64)
        got = randles_torch.impedance(params, w).numpy()

        assert got.real == pytest.approx(spectrum.z_clean.real, rel=RTOL, abs=1e-300)
        assert got.imag == pytest.approx(spectrum.z_clean.imag, rel=RTOL, abs=1e-300)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 0.999, 0.2, 0.87])
def test_alpha_edge_cases(alpha: float) -> None:
    """0.5 and 1.0 take shortcut paths in elements.rs complex_powf."""
    params = np.array([2.0, 3e-4, alpha, 40.0, 7.5])
    freqs = np.logspace(-2, 6, 64)

    got = randles_torch.impedance(
        torch.tensor(params, dtype=torch.float64),
        torch.tensor(TAU * freqs, dtype=torch.float64),
    ).numpy()
    expected = _reference(params, freqs)

    assert got.real == pytest.approx(expected.real, rel=RTOL)
    assert got.imag == pytest.approx(expected.imag, rel=RTOL)


def test_batched_impedance_matches_per_sample() -> None:
    rng = np.random.default_rng(7)
    spectra = priors.sample_many(rng, 16)
    freqs = np.logspace(-2, 6, 64)

    params = torch.tensor(np.array([s.params for s in spectra]), dtype=torch.float64)
    w = torch.tensor(TAU * freqs, dtype=torch.float64).expand(len(spectra), -1)
    batched = randles_torch.impedance(params, w).numpy()

    for i, spectrum in enumerate(spectra):
        expected = _reference(spectrum.params, freqs)
        assert batched[i].real == pytest.approx(expected.real, rel=RTOL)
        assert batched[i].imag == pytest.approx(expected.imag, rel=RTOL)


def test_modulus_residuals_match_fasteis() -> None:
    """The loss must agree with the residual the optimiser itself minimises."""
    rng = np.random.default_rng(3)
    spectrum = priors.sample(rng)
    circuit = fasteis.Circuit(circuits.CIRCUIT_STRING)

    # deliberately off the truth, so the residuals are not all ~0
    guess = spectrum.params * np.array([1.3, 0.7, 1.0, 1.2, 0.9])
    expected = np.asarray(
        circuit.residuals(
            list(guess), list(spectrum.freqs), list(spectrum.z), "modulus"
        )
    )

    got = randles_torch.modulus_residuals(
        torch.tensor(guess, dtype=torch.float64),
        torch.tensor(TAU * spectrum.freqs, dtype=torch.float64),
        torch.tensor(spectrum.z, dtype=torch.complex128),
    ).numpy()

    assert got.reshape(-1) == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_residual_loss_is_differentiable() -> None:
    rng = np.random.default_rng(11)
    spectrum = priors.sample(rng)

    params = torch.tensor(spectrum.params, dtype=torch.float64, requires_grad=True)
    loss = randles_torch.residual_loss(
        params,
        torch.tensor(TAU * spectrum.freqs, dtype=torch.float64),
        torch.tensor(spectrum.z, dtype=torch.complex128),
    )
    loss.backward()

    assert params.grad is not None
    assert torch.isfinite(params.grad).all()


CHECKPOINT = Path("training/checkpoints/randles/best.pt")
WEIGHTS = Path("src/models/randles.eisnn")


@pytest.mark.skipif(
    not CHECKPOINT.exists() or not WEIGHTS.exists(),
    reason="no trained checkpoint and exported weights",
)
def test_rust_guess_matches_the_torch_network() -> None:
    """src/nn.rs reimplements the forward pass, resampling and denormalisation.

    Loads the weights back out of the weights file the crate uses.
    Compares the whole guess to also catch any drift in the scaling rules.
    """
    from training import train

    net, std, device = train.load_checkpoint(CHECKPOINT)
    _, tensors = serialize_weights.read(WEIGHTS)
    net.load_state_dict(
        {
            name.removeprefix("w."): torch.tensor(value, dtype=torch.float32)
            for name, value in tensors.items()
            if name.startswith("w.")
        }
    )

    circuit = fasteis.Circuit(circuits.CIRCUIT_STRING)
    for spectrum in priors.sample_many(np.random.default_rng(21), 200):
        expected = train.guess(net, std, device, spectrum.freqs, spectrum.z)
        got = np.array(circuit.guess(list(spectrum.freqs), list(spectrum.z)))
        # remaining gap is torch computing in f32 against rust's f64
        assert got == pytest.approx(expected, rel=1e-4)
