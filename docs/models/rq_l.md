# `rq_l`

`L0-R0-(R1,CPE1)`

![rq_l circuit diagram](../assets/circuits/rq_l.svg#only-light)
![rq_l circuit diagram](../assets/circuits/rq_l-dark.svg#only-dark)
{: style="text-align:center" }

![rq_l Nyquist plot](../assets/circuits/rq_l-nyquist.svg#only-light)
![rq_l Nyquist plot](../assets/circuits/rq_l-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

As [`rq`](rq.md), with an inductor. A series inductor `L0` accounts for cable and cell inductance, which appears as a tail below the real axis at high frequency.

<!-- results:rq_l_model -->
ML model: 69k parameter 1D CNN, trained on synthetic data, 1.6 ms per guess. See [Training](../training.md).
<!-- /results:rq_l_model -->

## Benchmarks

<!-- results:rq_l -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 34 |
| library defaults | 79.80% | 135 | 317 | 504 | 182 |
| truth x/div 5 | 87.40% | 79 | 214 | 377 | 123 |
| **ml guess** | **100.00%** | **0** | **11** | **33** | **45** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 55 |
| library defaults | 90.00% | 160 | 1883 | 16233 | 241 |
| truth x/div 5 | 98.20% | 96 | 330 | 6952 | 161 |
| **ml guess** | **100.00%** | **0** | **11** | **69** | **66** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `L0.l` | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` |
|---|---|---|---|---|---|
| median | 1.6 | 1.5 | 0.8 | 2.8 | 0.6 |
| p90 | 6.9 | 29.1 | 3.3 | 14.6 | 2.8 |
| p99 | 42.5 | 245.1 | 11.9 | 48.5 | 12.1 |
<!-- /results:rq_l -->

## Benchmark notes

<!-- results:rq_l_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random, CPE exponents ±0.15.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:rq_l_method -->
