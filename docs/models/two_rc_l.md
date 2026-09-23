# `two_rc_l`

`L0-R0-(R1,C1)-(R2,C2)`

![two_rc_l circuit diagram](../assets/circuits/two_rc_l.svg#only-light)
![two_rc_l circuit diagram](../assets/circuits/two_rc_l-dark.svg#only-dark)
{: style="text-align:center" }

![two_rc_l Nyquist plot](../assets/circuits/two_rc_l-nyquist.svg#only-light)
![two_rc_l Nyquist plot](../assets/circuits/two_rc_l-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

As [`two_rc`](two_rc.md), with an inductor. A series inductor `L0` accounts for cable and cell inductance, which appears as a tail below the real axis at high frequency.

<!-- results:two_rc_l_model -->
ML model: 69k parameter 1D CNN, trained on synthetic data, 1.5 ms per guess. See [Training](../training.md).
<!-- /results:two_rc_l_model -->

## Benchmarks

<!-- results:two_rc_l -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 53 |
| library defaults | 69.30% | 254 | 528 | 1330 | 285 |
| truth x/div 5 | 71.50% | 118 | 317 | 738 | 171 |
| **ml guess** | **100.00%** | **12** | **13** | **39** | **53** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 76 |
| library defaults | 90.90% | 295 | 2880 | 10628 | 400 |
| truth x/div 5 | 91.40% | 203 | 1258 | 7074 | 312 |
| **ml guess** | **100.00%** | **1** | **13** | **52** | **76** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `L0.l` | `R0.r` | `R1.r` | `C1.c` | `R2.r` | `C2.c` |
|---|---|---|---|---|---|---|
| median | 1.8 | 0.9 | 0.8 | 1.4 | 1.6 | 3.2 |
| p90 | 8.3 | 16.9 | 3.5 | 6.8 | 11.8 | 24.5 |
| p99 | 62.8 | 269.3 | 13.5 | 35.5 | 57.1 | 137.5 |
<!-- /results:two_rc_l -->

## Benchmark notes

<!-- results:two_rc_l_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:two_rc_l_method -->
