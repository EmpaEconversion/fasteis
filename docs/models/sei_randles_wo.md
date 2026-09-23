# `sei_randles_wo`

`R0-(R1,CPE1)-(R2-Wo2,CPE2)`

![sei_randles_wo circuit diagram](../assets/circuits/sei_randles_wo.svg#only-light)
![sei_randles_wo circuit diagram](../assets/circuits/sei_randles_wo-dark.svg#only-dark)
{: style="text-align:center" }

![sei_randles_wo Nyquist plot](../assets/circuits/sei_randles_wo-nyquist.svg#only-light)
![sei_randles_wo Nyquist plot](../assets/circuits/sei_randles_wo-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

As [`sei_randles`](sei_randles.md), with a finite-length open Warburg element instead of semi-infinite, for diffusion into a layer with a blocking boundary like an intercalation particle. The 45° diffusion tail turns vertical at low frequency.

<!-- results:sei_randles_wo_model -->
ML model: 270k parameter 1D CNN, trained on synthetic data, 2.5 ms per guess. See [Training](../training.md).
<!-- /results:sei_randles_wo_model -->

## Benchmarks

<!-- results:sei_randles_wo -->
### Plain LM

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 276 |
| library defaults | 23.20% | 322 | 1504 | 7004 | 676 |
| truth x/div 5 | 26.00% | 192 | 730 | 5683 | 401 |
| **ml guess** | **91.30%** | **16** | **252** | **5904** | **257** |

### `Circuit.fit()`

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 397 |
| library defaults | 41.40% | 928 | 43062 | 140937 | 4856 |
| truth x/div 5 | 52.60% | 746 | 17284 | 54826 | 3144 |
| **ml guess** | **91.00%** | **19** | **1094** | **38252** | **414** |

### Error of the guess

Relative error of each guessed parameter before fitting, in %.

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `Wo2.z0` | `Wo2.tau` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|---|
| median | 0.6 | 1.7 | 3.2 | 0.6 | 23.2 | 46.9 | 49.0 | 5.6 | 1.2 |
| p90 | 4.8 | 11.9 | 20.4 | 4.1 | 84.2 | 255.1 | 306.0 | 43.9 | 11.6 |
| p99 | 63.9 | 50.1 | 97.2 | 16.0 | 250.3 | 1419.4 | 5491.2 | 226.0 | 32.2 |
<!-- /results:sei_randles_wo -->

## Benchmark notes

<!-- results:sei_randles_wo_method -->
Synthetic benchmarks use 1000 spectra drawn from the same distribution as the training data, with a different seed: 4 to 8 decade sweeps of 20 to 100 points, with 0.2% to 5% noise.

- **floor (truth)**: start from the true parameters.
- **library defaults**: start from the `Circuit()` placeholder values.
- **truth x/div 5**: true magnitudes multiplied or divided by 5 at random, CPE exponents ±0.15.
- **ml guess**: start from `Circuit.guess()`.

Plain LM is a single-start Levenberg-Marquardt. `Circuit.fit()` is LM that also screens candidate starts and restarts on bad fits.

**Converged**: final cost within 1% of the fit from the truth. **Sweeps**: impedance evaluations of the whole spectrum, including Jacobians. **Excess**: sweeps beyond the fit from the truth, for converged fits only.
<!-- /results:sei_randles_wo_method -->
