# `rc`

`R0-(R1,C1)`

![rc circuit diagram](../assets/circuits/rc.svg#only-light)
![rc circuit diagram](../assets/circuits/rc-dark.svg#only-dark)
{: style="text-align:center" }

![rc Nyquist plot](../assets/circuits/rc-nyquist.svg#only-light)
![rc Nyquist plot](../assets/circuits/rc-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

A series resistance `R0` and one RC element, giving a single ideal semicircle offset from the origin by `R0`. In batteries, `R0` is usually the electrolyte and contact resistance, `R1` the charge-transfer resistance, and `C1` the double layer capacitance.

<!-- results:rc_model -->
ML model: 18k parameter 1D CNN, trained on synthetic data, 0.2 ms per guess. See [Training](../training.md).
<!-- /results:rc_model -->

## Benchmarks

<!-- results:rc -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 22 |
| library defaults | 99.20% | 56 | 99 | 311 | 78 |
| truth x/div 5 | 98.30% | 36 | 51 | 121 | 58 |
| **ml guess** | **100.00%** | **6** | **7** | **14** | **29** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 39 |
| library defaults | 100.00% | 50 | 114 | 994 | 89 |
| truth x/div 5 | 100.00% | 30 | 80 | 280 | 74 |
| **ml guess** | **100.00%** | **6** | **7** | **14** | **46** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `C1.c` |
|---|---|---|---|
| median | 0.7 | 0.7 | 3.1 |
| p90 | 3.3 | 2.7 | 16.5 |
| p99 | 51.3 | 5.8 | 44.3 |
<!-- /results:rc -->

## Benchmark notes

<!-- results:rc_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:rc_method -->
