# Training

Convolutional neural networks of particular circuits are trained to be used
for initial guesses in `Circuit.fit()`, to (hopefully) converge reliably
and quickly without any manual initial guess.

The `fasteis` comes bundled with pre-trained models. The `fasteis` PyPI package
does not come with all the training infrastructure. Training new circuits
requires a clone of the repository, and extra dependencies like torch.

Trained models and their benchmark results are listed under [Models](models/index.md).

The `fasteis` repository contains scripts to train circuit models.

## Model

All models use a 1D convolutional neural network over log-frequency, starting
with a 3x64 matrix of normalized |Z|, phase, and frequency.

Shifting a time constant translates features in frequency, so 1D
convolution along the frequency axis works well (translation equivariance
inductive bias).

Convolution starts a stem from the 3 input channels to `x` 'feature' channels,
followed by four residual blocks with dilations 1/2/4/8, then mean+max pooling
over the frequency axis to get a `2x` length vector (plus 2 scaling constants).

Then a 3-layer 'head' multiplies to a width `y`, then emits a mean and
log-variance per parameter.

`model.Config` sets the widths `x` and `y` and is stored in the checkpoint. The
width is a compromise between having fast/small model vs accuracy. `rc` uses
16 / 64, the two `sei_randles` circuits use 64 / 256, and the rest 32 / 128.

| channels / head | weights | file | inference |
|---|---|---|---|
| 16 / 64 | 17.7k | 36 KiB | 0.23 ms |
| 32 / 128 | 68.6k | 136 KiB | 0.70 ms |
| 64 / 256 | 269k | 529 KiB | 2.41 ms |

## Input

The input frequencies and impedances are resampled onto 64 log-spaced points
across the measured range. There are three arrays in the input: `log10|Z_hat|`,
`phase/(pi/2)`, `log10(w_hat)/4`, and two scalars: sweep width in decades and
point count.

## Scaling symmetries

The frequencies, impedance, and the parameters are all renomalised so the model
only needs to learn the curve shape, and not the scale.

Impedance is invariant under `Z -> k*Z` and `w -> w/w_c` when parameters are
transformed to match.

Each parameter picks up the scales as

```
physical = normalised * k**a * w_c**(b + c * params[i])
```

with `i = -1` when the exponent has no parameter dependence. `circuits.SCALING`
holds one `(a, b, c, i)` row per parameter and is written into the weight file,
so the Rust reader needs no per-circuit code. Resistances are `(1, 0, 0, -1)`,
capacitances `(-1, -1, 0, -1)`, inductances `(1, -1, 0, -1)`, time constants
`(0, -1, 0, -1)`, and a CPE `q` is `(-1, 0, -1, alpha_index)`.

## Choosing k and w_c

Several methods have been tried, with `reactive_centroid` seeming the best.
Here, each point is weighted by `max(-sin(phase), 0)`, bounded in `[0,1]`
and scale-free, then takes log-space weighted means of `|Z|` and `w`. This means
featureless parts of the curve carry little weight, so widening the sweet beyond
the features does not move the scaling estimates.

See `compare_scales.py` for details of the comparison. The numbers here are how
much the target (scaled) parameters shift between different conditions. A good
scale estimator minimizes all three of these:

| estimator | across systems | across sweeps | across noise |
|---|---|---|---|
| none (control) | 1.7 – 2.3 | — | — |
| `window` | 0.44 – 0.77 | 0.19 – 0.36 | 0.003 |
| **`reactive_centroid`** | **0.42 – 0.74** | **0.13 – 0.27** | 0.006 |
| `imag_weighted` | 0.45 – 0.76 | 0.19 – 0.42 | 0.004 |
| `imag_peak` | 0.54 – 1.15 | 0.22 – 0.52 | 0.030 |

## Synthetic training data

See `priors.py`.

Training data is generated only where features of the circuit are observable,
otherwise it is impossible to guess parameters. Overall impedance and frequency
scale are randomised to exercise the normalisation. Noise is proportional to
`|Z|`, and log-uniform over 0.2%–5%.

Point dropout, outliers and a series inductance term exist behind flags in
`PriorConfig`, all off by default.

## Loss

```
L = residual + lambda * nll,   lambda: 1.0 -> 0.02
```

The residual is the same as the fit - a modulus-weighted residual of the guessed curve vs observed.

The negative log-likelihood (NLL) term keeps gradient available where the
residual has plateaus. E.g. a time constant several decades off gives a curve
with no observable arc in the window, where `d(residual)/d(log tau)` vanishes -
and keeps the log-variance head trained. It never decays to zero, so an
out-of-range alpha always has a gradient pulling it back.

## Storing the model

The model is stored in `src/models/*.eisnn`, written by `serialize_weights.py`.

The weights themselves are 99.8% of the bytes, changing dtype can reduce size
at the cost of accuracy. For randles, dropping to f16 is reasonable:

| dtype | file | init params error | converged | excess med | p99 |
|---|---|---|---|---|---|
| f32 | 270 KiB | 1.8% | 100% | 1 | 7 |
| **f16** (default) | **136 KiB** | **1.8%** | 100% | 1 | 7 |
| int8 | 69 KiB | 3.6% | 100% | 1 | 9 |

## Adding a circuit

1. Clone the repo: `git clone https://github.com/empaeconversion/fasteis`
2. Install the package with dev dependencies `uv sync --all-extras`
3. Create a `TrainingCircuit` class for the circuit in `circuits.py` and add it to the registry
4. Check it is identifiable with `training/inspect_priors.py <name>`
5. Train with e.g. `training/train.py --circuit <name> --steps 10000 --batch 4096 --workers 12`
6. Export with `training/export.py --circuit <name>`, which writes `src/models/<name>.eisnn`
7. Add a row to `MODELS` in `src/models.rs` and rebuild

    *(optional, for devs)*

8. Benchmark with `training/benchmark.py --circuit <name> --n 2000`
9. Add a page for it under `docs/models/` and list it in `zensical.toml`'s nav
10. Regenerate the tables with `training/update_docs.py`
