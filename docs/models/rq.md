# `rq`

`R0-(R1,CPE1)`

![rq circuit diagram](../assets/circuits/rq.svg#only-light)
![rq circuit diagram](../assets/circuits/rq-dark.svg#only-dark)
{: style="text-align:center" }

![rq Nyquist plot](../assets/circuits/rq-nyquist.svg#only-light)
![rq Nyquist plot](../assets/circuits/rq-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

A series resistance `R0` and one RQ element, a resistor in parallel with a constant phase element (CPE). In batteries, `R0` is usually the electrolyte and contact resistance, `R1` the charge-transfer resistance, and `CPE1` the double layer capacitance. The CPE flattens the semicircle, which is common for real electrodes with rough or inhomogeneous surfaces.

<!-- results:rq_model -->
ML model: 68k parameter 1D CNN, trained on synthetic data, 1.6 ms per guess. See [Training](../training.md).
<!-- /results:rq_model -->

## Benchmarks

<!-- results:rq -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 28 |
| library defaults | 81.20% | 75 | 186 | 393 | 112 |
| truth x/div 5 | 88.00% | 55 | 121 | 222 | 83 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **37** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 47 |
| library defaults | 88.70% | 84 | 453 | 8964 | 148 |
| truth x/div 5 | 99.50% | 55 | 182 | 1356 | 106 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **56** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` |
|---|---|---|---|---|
| median | 0.6 | 0.7 | 3.3 | 0.5 |
| p90 | 4.2 | 2.5 | 15.8 | 2.8 |
| p99 | 86.4 | 6.1 | 71.1 | 8.0 |
<!-- /results:rq -->

## Benchmark notes

<!-- results:rq_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random, CPE exponents ±0.15.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:rq_method -->
