# `two_rc`

`R0-(R1,C1)-(R2,C2)`

![two_rc circuit diagram](../assets/circuits/two_rc.svg#only-light)
![two_rc circuit diagram](../assets/circuits/two_rc-dark.svg#only-dark)
{: style="text-align:center" }

![two_rc Nyquist plot](../assets/circuits/two_rc-nyquist.svg#only-light)
![two_rc Nyquist plot](../assets/circuits/two_rc-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

A series resistance `R0` and two RC elements, giving two ideal semicircles. Used when two processes have well separated time constants, for example a surface film and charge transfer.

<!-- results:two_rc_model -->
ML model: 69k parameter 1D CNN, trained on synthetic data, 1.6 ms per guess. See [Training](../training.md).
<!-- /results:two_rc_model -->

## Benchmarks

<!-- results:two_rc -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 44 |
| library defaults | 67.40% | 168 | 374 | 1025 | 182 |
| truth x/div 5 | 75.50% | 79 | 215 | 475 | 124 |
| **ml guess** | **100.00%** | **0** | **11** | **22** | **45** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 65 |
| library defaults | 91.60% | 176 | 702 | 5782 | 258 |
| truth x/div 5 | 92.80% | 125 | 712 | 4519 | 207 |
| **ml guess** | **100.00%** | **0** | **11** | **22** | **66** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `C1.c` | `R2.r` | `C2.c` |
|---|---|---|---|---|---|
| median | 0.4 | 0.7 | 0.9 | 1.3 | 2.4 |
| p90 | 1.8 | 2.3 | 4.0 | 10.5 | 16.0 |
| p99 | 26.7 | 5.8 | 10.8 | 57.6 | 46.1 |
<!-- /results:two_rc -->

## Benchmark notes

<!-- results:two_rc_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:two_rc_method -->
