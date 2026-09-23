# `randles`

`R0-(R1-W1,CPE1)`

![randles circuit diagram](../assets/circuits/randles.svg#only-light)
![randles circuit diagram](../assets/circuits/randles-dark.svg#only-dark)
{: style="text-align:center" }

![randles Nyquist plot](../assets/circuits/randles-nyquist.svg#only-light)
![randles Nyquist plot](../assets/circuits/randles-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

The Randles circuit. One RQ element, usually attributed to charge transfer and the double layer, with a semi-infinite Warburg element `W1` in series with the charge transfer resistance for diffusion. The arc is followed by a 45° diffusion tail at low frequency.

<!-- results:randles_model -->
ML model: 69k parameter 1D CNN, trained on synthetic data, 0.7 ms per guess. See [Training](../training.md).
<!-- /results:randles_model -->

## Benchmarks

<!-- results:randles -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 45 |
| library defaults | 57.55% | 102 | 318 | 726 | 204 |
| truth x/div 5 | 70.75% | 78 | 205 | 452 | 124 |
| **ml guess** | **99.85%** | **0** | **11** | **57** | **45** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 66 |
| library defaults | 78.00% | 170 | 1884 | 19384 | 333 |
| truth x/div 5 | 88.15% | 103 | 720 | 8770 | 200 |
| **ml guess** | **99.85%** | **0** | **11** | **65** | **66** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `W1.aw` | `CPE1.q` | `CPE1.alpha` |
|---|---|---|---|---|---|
| median | 1.4 | 1.5 | 1.3 | 3.5 | 0.6 |
| p90 | 3.7 | 25.6 | 9.7 | 16.5 | 3.4 |
| p99 | 34.1 | 101.6 | 46.7 | 54.0 | 10.9 |
<!-- /results:randles -->

## Benchmark notes

<!-- results:randles_method -->
Synthetic benchmarks use 2000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random, CPE exponents ±0.15.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:randles_method -->
