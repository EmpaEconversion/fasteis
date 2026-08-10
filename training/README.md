# ML parameter guessing

Scripts to trains a neural network which guess circuit parameters from measured
EIS data.

Used as an initial guess for `Circuit.fit()`, to (hopefully) converge reliably
and quickly without any manual initial guess.

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

## Models

Circuits are trained on synthetic data, and summarised in the table below.
- Percentages are rates of convergence starting from parameters multiplied or divided by 5 from real values vs the model intial guess.
- 'floor' is the median impedance calculation count starting from the true parameters.
- 'excess' is extra calculations beyond 'floor', median and 90th percentile are shown.

<!-- results:library -->
| name | circuit | params | params * / 5 | ml guess | floor | ml excess med | p90 |
|---|---|---|---|---|---|---|---|
| `rc` | `R0-(R1,C1)` | 3 | 98.3% | **100.0%** | 22| **6** | 7 |
| `rc_l` | `L0-R0-(R1,C1)` | 4 | 94.8% | **100.0%** | 28| **0** | 9 |
| `rq` | `R0-(R1,CPE1)` | 4 | 88.0% | **100.0%** | 28| **0** | 9 |
| `rq_l` | `L0-R0-(R1,CPE1)` | 5 | 87.4% | **100.0%** | 34| **0** | 11 |
| `two_rc` | `R0-(R1,C1)-(R2,C2)` | 5 | 75.5% | **100.0%** | 44| **0** | 11 |
| `two_rc_l` | `L0-R0-(R1,C1)-(R2,C2)` | 6 | 71.5% | **100.0%** | 53| **12** | 13 |
| `two_rq` | `R0-(R1,CPE1)-(R2,CPE2)` | 7 | 55.1% | **99.6%** | 76| **15** | 62 |
| `two_rq_l` | `L0-R0-(R1,CPE1)-(R2,CPE2)` | 8 | 50.3% | **98.8%** | 86| **17** | 188 |
| `randles` | `R0-(CPE1,R1-W1)` | 5 | 69.8% | **99.9%** | 45| **10** | 11 |
| `sei_randles` | `R0-(R1,CPE1)-(R2-W2,CPE2)` | 8 | 41.6% | **99.1%** | 103| **0** | 52 |
| `sei_randles_wo` | `R0-(R1,CPE1)-(R2-Wo2,CPE2)` | 9 | 26.0% | **91.3%** | 276| **16** | 252 |
<!-- /results:library -->

Expand to details of more challenging models:

### `two_rq`

<!-- results:two_rq -->
<details>
<summary>Show details</summary>

`R0-(R1,CPE1)-(R2,CPE2)`, 1000 synthetic spectra. Inference costs 0.72 ms/spectrum against 1.99 ms for the fit it starts.

Plain LM:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 76 |
| library defaults | 51.20% | 368 | 717 | 1641 | 366 |
| truth x/div 5 | 55.10% | 171 | 458 | 1476 | 276 |
| **ml guess** | **99.60%** | **15** | **62** | **246** | **91** |

`Circuit.fit()` / smart LM, which screens candidate starts:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 101 |
| library defaults | 77.90% | 415 | 5348 | 44383 | 685 |
| truth x/div 5 | 77.30% | 352 | 2717 | 24850 | 652 |
| **ml guess** | **99.70%** | **15** | **62** | **294** | **131** |

Relative error of the ml guess, before fitting (%):
| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|
| median | 1.2 | 5.4 | 7.6 | 1.7 | 8.6 | 36.7 | 9.6 |
| p90 | 10.5 | 24.3 | 34.6 | 7.5 | 35.5 | 113.7 | 27.6 |
| p99 | 96.6 | 94.2 | 145.8 | 16.8 | 115.9 | 380.5 | 50.0 |
</details>
<!-- /results:two_rq -->

### `two_rq_l`

<!-- results:two_rq_l -->
<details>
<summary>Show details</summary>

`L0-R0-(R1,CPE1)-(R2,CPE2)`, 1000 synthetic spectra. Inference costs 0.72 ms/spectrum against 3.23 ms for the fit it starts.

Plain LM:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 86 |
| library defaults | 57.10% | 587 | 1180 | 2826 | 604 |
| truth x/div 5 | 50.30% | 260 | 687 | 2509 | 366 |
| **ml guess** | **98.80%** | **17** | **188** | **668** | **120** |

`Circuit.fit()` / smart LM, which screens candidate starts:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 130 |
| library defaults | 83.90% | 633 | 11359 | 81777 | 945 |
| truth x/div 5 | 80.00% | 507 | 4160 | 36863 | 928 |
| **ml guess** | **99.00%** | **17** | **192** | **1131** | **148** |

Relative error of the ml guess, before fitting (%):
| | `L0.l` | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|
| median | 4.8 | 7.6 | 8.1 | 13.6 | 2.7 | 13.5 | 30.9 | 6.6 |
| p90 | 13.6 | 47.7 | 50.2 | 61.1 | 12.5 | 54.6 | 145.0 | 24.1 |
| p99 | 85.9 | 237.1 | 209.8 | 268.5 | 31.5 | 165.9 | 711.8 | 46.1 |
</details>
<!-- /results:two_rq_l -->

### `randles`

<!-- results:randles -->
<details>
<summary>Show details</summary>

`R0-(CPE1,R1-W1)`, 1000 synthetic spectra. Inference costs 1.72 ms/spectrum against 2.64 ms for the fit it starts.

Plain LM:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 45 |
| library defaults | 55.20% | 102 | 284 | 575 | 199 |
| truth x/div 5 | 69.80% | 79 | 206 | 522 | 124 |
| **ml guess** | **99.90%** | **10** | **11** | **66** | **45** |

`Circuit.fit()` / smart LM, which screens candidate starts:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 66 |
| library defaults | 78.40% | 169 | 1795 | 13150 | 325 |
| truth x/div 5 | 89.10% | 102 | 671 | 3637 | 190 |
| **ml guess** | **99.90%** | **10** | **11** | **66** | **66** |

Relative error of the ml guess, before fitting (%):
| | `R0.r` | `CPE1.q` | `CPE1.alpha` | `R1.r` | `W1.aw` |
|---|---|---|---|---|---|
| median | 1.2 | 3.3 | 0.7 | 2.9 | 1.4 |
| p90 | 4.0 | 16.4 | 3.5 | 24.9 | 9.6 |
| p99 | 55.1 | 81.8 | 12.5 | 119.2 | 39.2 |
</details>
<!-- /results:randles -->

### `sei_randles`

<!-- results:sei_randles -->
<details>
<summary>Show details</summary>

`R0-(R1,CPE1)-(R2-W2,CPE2)`, 1000 synthetic spectra. Inference costs 2.35 ms/spectrum against 3.26 ms for the fit it starts.

Plain LM:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 103 |
| library defaults | 30.40% | 400 | 1180 | 5233 | 726 |
| truth x/div 5 | 41.60% | 228 | 672 | 2386 | 416 |
| **ml guess** | **99.10%** | **0** | **52** | **334** | **103** |

`Circuit.fit()` / smart LM, which screens candidate starts:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 148 |
| library defaults | 53.60% | 692 | 16594 | 80818 | 2018 |
| truth x/div 5 | 66.10% | 554 | 5597 | 37704 | 1148 |
| **ml guess** | **99.00%** | **0** | **66** | **1797** | **147** |

Relative error of the ml guess, before fitting (%):
| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `W2.aw` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|
| median | 1.3 | 2.5 | 4.7 | 0.9 | 8.1 | 5.7 | 10.6 | 3.4 |
| p90 | 4.7 | 15.6 | 23.9 | 4.7 | 42.9 | 41.6 | 58.6 | 16.2 |
| p99 | 58.1 | 66.7 | 71.1 | 14.1 | 149.4 | 171.9 | 194.2 | 36.9 |
</details>
<!-- /results:sei_randles -->

### `sei_randles_wo`

<!-- results:sei_randles_wo -->
<details>
<summary>Show details</summary>

`R0-(R1,CPE1)-(R2-Wo2,CPE2)`, 1000 synthetic spectra. Inference costs 2.48 ms/spectrum against 14.71 ms for the fit it starts.

Plain LM:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 276 |
| library defaults | 23.20% | 322 | 1504 | 7004 | 676 |
| truth x/div 5 | 26.00% | 192 | 730 | 5683 | 401 |
| **ml guess** | **91.30%** | **16** | **252** | **5904** | **257** |

`Circuit.fit()` / smart LM, which screens candidate starts:
| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 397 |
| library defaults | 41.40% | 928 | 43062 | 140937 | 4856 |
| truth x/div 5 | 52.60% | 746 | 17284 | 54826 | 3144 |
| **ml guess** | **91.00%** | **19** | **1094** | **38252** | **414** |

Relative error of the ml guess, before fitting (%):
| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `Wo2.z0` | `Wo2.tau` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|---|
| median | 0.6 | 1.7 | 3.2 | 0.6 | 23.2 | 46.9 | 49.0 | 5.6 | 1.2 |
| p90 | 4.8 | 11.9 | 20.4 | 4.1 | 84.2 | 255.1 | 306.0 | 43.9 | 11.6 |
| p99 | 63.9 | 50.1 | 97.2 | 16.0 | 250.3 | 1419.4 | 5491.2 | 226.0 | 32.2 |
</details>
<!-- /results:sei_randles_wo -->


### Benchmarks against real data

### `two_rq_l`

<!-- results:two_rq_l_real -->
<details>
<summary>Show details</summary>

`two_rq_l` against 201 measured spectra. Ground truth is not known, so 'converged' means within tolerance of the best chi-square reached.

| source of initial parameters | converged | med sweeps | med ms | med chi2 |
|---|---|---|---|---|
| library defaults | 93.03% | 5038 | 27.87 | 2.081e-03 |
| **ml guess** | **95.52%** | **2961** | **16.98** | **2.021e-03** |
| differential_evolution | 67.16% | 203285 | 1288.50 | 3.321e-03 |
</details>
<!-- /results:two_rq_l_real -->

## Training

### Model

All models use a 1D convolutional neural network over log-frequency, starting
with a 3x64 matrix of normalized |Z|, phase, and frequency.

Shifting a time constant translates features in frequency, so 1D
convolution along the frequency axis works well (translation equivariance
inductive bias).

Convolution starts a stem from the 3 input channels to $x$ 'feature' channels,
followed by four residual blocks with dilations 1/2/4/8, then mean+max pooling
over the frequency axis to get a $2x$ length vector (plus 2 scaling constants).

Then a 3-layer 'head' multiplies to a width $y$, then emits a mean and
log-variance per parameter.

`model.Config` sets the widths $x$ and $y$ and is stored in the checkpoint. The
width is a compromise between having fast/small model vs accuracy. `rc` uses
16 / 64, the two `sei_randles` circuits use 64 / 256, and the rest 32 / 128. 

| channels / head | weights | file | inference |
|---|---|---|---|
| 16 / 64 | 17.7k | 36 KiB | 0.23 ms |
| 32 / 128 | 68.6k | 136 KiB | 0.70 ms |
| 64 / 256 | 269k | 529 KiB | 2.41 ms |

### Input

The input frequencies and impedances are resampled onto 64 log-spaced points
across the measured range. There are three arrays in the input: `log10|Z_hat|`,
`phase/(pi/2)`, `log10(w_hat)/4`, and two scalars: sweep width in decades and
point count.

### Scaling symmetries

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

### Loss

```
L = residual + lambda * nll,   lambda: 1.0 -> 0.02
```

The residual is the same as the fit - a modulus-weighted residual of the guessed curve vs observed.

The negative log-likelihood (NLL) term keeps gradient available where the
residual has plateaus. E.g. a time constant several decades off gives a curve
with no observable arc in the window, where `d(residual)/d(log tau)` vanishes -
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

1. Create a `TrainingCircuit` class for the circuit in `circuits.py` and add it to the registry
2. Check it is identifiable with `training/inspect_priors.py <name>`
3. Train with e.g. `training/train.py --circuit <name> --steps 10000 --batch 4096 --workers 12`
4. Export with `training/export.py --circuit <name>`, which writes `src/models/<name>.eisnn`
5. Add a row to `MODELS` in `src/models.rs` and rebuild
6. Benchmark with `training/benchmark.py --circuit <name> --n 2000`
7. Regenerate the tables with `training/update_readme.py`
