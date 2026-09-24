# `sei_randles`

`R0-(R1,CPE1)-(R2-W2,CPE2)`

![sei_randles circuit diagram](../assets/circuits/sei_randles.svg#only-light)
![sei_randles circuit diagram](../assets/circuits/sei_randles-dark.svg#only-dark)
{: style="text-align:center" }

![sei_randles Nyquist plot](../assets/circuits/sei_randles-nyquist.svg#only-light)
![sei_randles Nyquist plot](../assets/circuits/sei_randles-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

A [`randles`](rq.md) circuit with an additional RQ element, often attributed to a solid electrolyte interphase (SEI) or other surface film.

<!-- results:sei_randles_model -->
ML model: 270k parameter 1D CNN, trained on synthetic data, 2.3 ms per guess. See [Training](../training.md).
<!-- /results:sei_randles_model -->

## Benchmarks

<!-- results:sei_randles -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 103 |
| library defaults | 30.40% | 400 | 1180 | 5233 | 726 |
| truth x/div 5 | 41.60% | 228 | 672 | 2386 | 416 |
| **ml guess** | **99.10%** | **0** | **52** | **334** | **103** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 148 |
| library defaults | 53.60% | 692 | 16594 | 80818 | 2018 |
| truth x/div 5 | 66.10% | 554 | 5597 | 37704 | 1148 |
| **ml guess** | **99.00%** | **0** | **66** | **1797** | **147** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `W2.aw` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|
| median | 1.3 | 2.5 | 4.7 | 0.9 | 8.1 | 5.7 | 10.6 | 3.4 |
| p90 | 4.7 | 15.6 | 23.9 | 4.7 | 42.9 | 41.6 | 58.6 | 16.2 |
| p99 | 58.1 | 66.7 | 71.1 | 14.1 | 149.4 | 171.9 | 194.2 | 36.9 |
<!-- /results:sei_randles -->

## Benchmark notes

<!-- results:sei_randles_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random, CPE exponents ±0.15.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:sei_randles_method -->
