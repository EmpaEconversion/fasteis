# ML parameter guessing

Scripts to trains a neural network which guess circuit parameters from measured
EIS data.

Used as an initial guess for `Circuit.fit()`, to (hopefully) converge reliably
and quickly without any manual initial guess.

Trained circuits:

| name | circuit | fit parameters | weights | file | trained on |
|---|---|---|---|---|---|
| `randles` | `R0-(CPE1,R1-W1)` | 5 | 68.6k | 136 KiB | synthetic |

Training needs torch (`uv sync --all-extras`).

## Usage

```python
import fasteis

circuit = fasteis.Circuit("randles")
result = circuit.fit(freqs, z, guess_init=True)
```

`guess_init=True` looks up a model by the circuit's topology, if there is an
associated neural network, it can do one inference to guess reasonable intial
parameters, then normal least-squares fitting goes from there.
`Circuit.guess(freqs, z)` returns the parameter vector alone, in
`param_names()` order.

`Circuit.ml_circuits()` lists what is currently available.

## Results

Currently only using synthetic tests. The benchmarks compare fits starting from
different initial values:
- 'Floor' starts from already correct values (i.e. impossible to beat).
- Library defaults are fixed, physically sensible numbers, so should be easy to beat.
- 'Truth x/div 5' are the true values off by a factor 5, meant to represent
a reasonable real world guess.
- 'ML guess' is the initial parameter guess given by the ML model.

There are benchmarks for both standard LM, and for the LM with restarting tricks
used in `Circuit,fit()`, which can make bad initial guesses more robust.

### `randles`

From 2000 synthetic tests, the model inference costs ~0.7 ms.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med evals |
|---|---|---|---|---|---|
| floor (true values) | 100.00% | 0 | 0 | 0 | 5 |
| library defaults | 36.55% | 28 | 62 | 93 | 28 |
| truth x/div 5 | 70.20% | 9 | 23 | 44 | 14 |
| **ml guess** | **99.95%** | **0** | **1** | **6** | **5** |

`Circuit.fit()` / smart LM:

| source of initial parameters | converged | excess med | p90 | p99 | med evals |
|---|---|---|---|---|---|
| floor (true values) | 100.00% | 0 | 0 | 0 | 5 |
| library defaults | 75.35% | 79 | 302 | 1147 | 118 |
| truth x/div 5 | 88.45% | 13 | 76 | 393 | 21 |
| **ml guess** | **99.95%** | **0** | **1** | **6** | **5** |

Relative error of ML guessed parameters:

| | `R0.r` | `CPE1.q` | `CPE1.alpha` | `R1.r` | `W1.aw` |
|---|---|---|---|---|---|
| median | 1.1 | 3.3 | 0.6 | 2.9 | 1.4 |
| p90 | 3.6 | 15.8 | 3.5 | 27.4 | 10.5 |
| p99 | 44.2 | 58.8 | 12.1 | 157.2 | 42.5 |

## Training

### Model

All models use a 1D convolutional neural network over log-frequency.
Shifting a time constant translates features along that axis, so we use a
translation equivariance inductive bias.
Stem convolution, four residual blocks at dilations 1/2/4/8, mean+max pooling, a
three-layer head emitting a mean and log-variance per parameter.

`model.Config` sets the width and is stored in the checkpoint.
For the `randles` circuit, the 68k parameter model is used, which is smaller,
faster, and performs just as well.

| channels / head | weights | converged | excess med | p99 | inference |
|---|---|---|---|---|---|
| **32 / 128** | **68.6k** | **100%** | **0** | **4** | **0.71 ms** |
| 64 / 256 | 268k | 100% | 0 | 7 | 2.27 ms |

### Input

The input frequencies and impedances are resampled onto 64 log-spaced points
across the measured range. There are three arrays in the input: `log10|Z_hat|`,
`phase/(pi/2)`, `log10(w_hat)/4`, and two scalars: sweep width in decades and
point count.

### Scaling symmetries

The frequencies, impedance, and the parameters are all renomalised so the model
only needs to learn the curve shape, and not the scale.

Impedance is invariant under `Z -> k*Z` and `w -> w/s` when parameters are
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

### Choosing k and w_c

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

### Synthetic training data

See `priors.py`.

Training data is generated only where features of the circuit are observable,
otherwise it is impossible to guess parameters. Overall impedance and frequency
scale are randomised to exercise the normalisation. Noise is proportional to 
`|Z|`, and log-uniform over 0.2%–5%.

Point dropout, outliers and a series inductance term exist behind flags in
`PriorConfig`, all off by default.

From `inspect_priors.py`: the CPE arc lies inside the window for 100% of samples
and the Warburg onset for 78%. Relative standard error at the true parameters,
from the circuit Jacobian, is 0.005–0.039 median, and 6.4% of spectra have at
least one parameter that is not meaningfully constrained.

### Loss

```
L = residual + lambda * nll,   lambda: 1.0 -> 0.02
```

The residual term is the modulus-weighted residual of the guessed curve against the
observed one, matching what the optimiser itself minimises. `randles_torch.py`
provides a differentiable Randles for this.

The negative log-likelihood (NLL) term keeps gradient available where the
residual has plateaus. E.g. a time constant several decades off gives a curve
with no observable arc in the window, where `d(residual)/d(log tau)` vanishes —
and keeps the log-variance head trained. It never decays to zero, so an
out-of-range alpha always has a gradient pulling it back.

### Storing the model

The model is stored in `src/models/*.eisnn`, written by `serialize_weights.py`.

The weights themselves are 99.8% of the bytes, changing dtype can reduce size
at the cost of accuracy. For randles, dropping to f16 is reasonable:

| dtype | file | init params error | converged | excess med | p99 |
|---|---|---|---|---|---|
| f32 | 270 KiB | 1.8% | 100% | 1 | 7 |
| **f16** (default) | **136 KiB** | **1.8%** | 100% | 1 | 7 |
| int8 | 69 KiB | 3.6% | 100% | 1 | 9 |

## Adding a circuit

1. Add the circuit string, parameter names and `SCALING` rows to `circuits.py`.
2. Add a prior sampler in `priors.py` placing features relative to the sweep.
3. Add a differentiable implementation alongside `randles_torch.py`, with a parity
   test against `Circuit.impedance()`.
4. Train, then `export.py --name <name>`, which writes `src/models/<name>.eisnn`.
5. Add a row to `MODELS` in `src/models.rs` and rebuild.
6. Add a results table above.

Model details (e.g. scaling rules) are already stored in the .eisnn for `nn.rs`.

## Training a model

```sh
uv sync --group ml
uv run pytest training/tests/
uv run python training/train.py --steps 20000 --workers 4
uv run python training/export.py
uv run python training/benchmark.py --n 2000
```
