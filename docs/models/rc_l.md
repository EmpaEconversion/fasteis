# `rc_l`

`L0-R0-(R1,C1)`

![rc_l circuit diagram](../assets/circuits/rc_l.svg#only-light)
![rc_l circuit diagram](../assets/circuits/rc_l-dark.svg#only-dark)
{: style="text-align:center" }

![rc_l Nyquist plot](../assets/circuits/rc_l-nyquist.svg#only-light)
![rc_l Nyquist plot](../assets/circuits/rc_l-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

As [`rc`](rc.md), plus a series inductor `L0` for cable and cell inductance, which appears as a tail below the real axis at high frequency.

<!-- results:rc_l_model -->
ML model: 68k parameter 1D CNN, trained on synthetic data, 1.7 ms per guess. See [Training](../training.md).
<!-- /results:rc_l_model -->

## Benchmarks

<!-- results:rc_l -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 28 |
| library defaults | 96.40% | 99 | 251 | 425 | 130 |
| truth x/div 5 | 94.80% | 55 | 111 | 258 | 83 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **36** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 47 |
| library defaults | 99.20% | 94 | 407 | 3787 | 148 |
| truth x/div 5 | 99.90% | 63 | 185 | 882 | 111 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **55** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `L0.l` | `R0.r` | `R1.r` | `C1.c` |
|---|---|---|---|---|
| median | 1.7 | 0.9 | 0.5 | 1.1 |
| p90 | 5.9 | 15.9 | 2.2 | 5.2 |
| p99 | 50.1 | 253.4 | 8.3 | 18.0 |
<!-- /results:rc_l -->

## Benchmark notes

<!-- results:rc_l_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:rc_l_method -->
