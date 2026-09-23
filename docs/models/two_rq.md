# `two_rq`

`R0-(R1,CPE1)-(R2,CPE2)`

![two_rq circuit diagram](../assets/circuits/two_rq.svg#only-light)
![two_rq circuit diagram](../assets/circuits/two_rq-dark.svg#only-dark)
{: style="text-align:center" }

![two_rq Nyquist plot](../assets/circuits/two_rq-nyquist.svg#only-light)
![two_rq Nyquist plot](../assets/circuits/two_rq-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

A series resistance `R0` and two RQ elements, giving two flattened semicircles. The CPE version of [`two_rc`](two_rc.md).

<!-- results:two_rq_model -->
ML model: 69k parameter 1D CNN, trained on synthetic data, 0.7 ms per guess. See [Training](../training.md).
<!-- /results:two_rq_model -->

## Benchmarks

<!-- results:two_rq -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 76 |
| library defaults | 51.20% | 368 | 717 | 1641 | 366 |
| truth x/div 5 | 55.10% | 171 | 458 | 1476 | 276 |
| **ml guess** | **99.60%** | **15** | **62** | **246** | **91** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 101 |
| library defaults | 77.90% | 415 | 5348 | 44383 | 685 |
| truth x/div 5 | 77.30% | 352 | 2717 | 24850 | 652 |
| **ml guess** | **99.70%** | **15** | **62** | **294** | **131** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|
| median | 1.2 | 5.4 | 7.6 | 1.7 | 8.6 | 36.7 | 9.6 |
| p90 | 10.5 | 24.3 | 34.6 | 7.5 | 35.5 | 113.7 | 27.6 |
| p99 | 96.6 | 94.2 | 145.8 | 16.8 | 115.9 | 380.5 | 50.0 |
<!-- /results:two_rq -->

## Benchmark notes

<!-- results:two_rq_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random, CPE exponents ±0.15.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:two_rq_method -->
