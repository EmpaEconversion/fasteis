# `sei_randles_wo`

<!-- results:sei_randles_wo -->
`R0-(R1,CPE1)-(R2-Wo2,CPE2)`, 1000 synthetic spectra. Inference costs 2.48 ms/spectrum against 14.71 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 276 |
| library defaults | 23.20% | 322 | 1504 | 7004 | 676 |
| truth x/div 5 | 26.00% | 192 | 730 | 5683 | 401 |
| **ml guess** | **91.30%** | **16** | **252** | **5904** | **257** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 397 |
| library defaults | 41.40% | 928 | 43062 | 140937 | 4856 |
| truth x/div 5 | 52.60% | 746 | 17284 | 54826 | 3144 |
| **ml guess** | **91.00%** | **19** | **1094** | **38252** | **414** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `Wo2.z0` | `Wo2.tau` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|---|
| median | 0.6 | 1.7 | 3.2 | 0.6 | 23.2 | 46.9 | 49.0 | 5.6 | 1.2 |
| p90 | 4.8 | 11.9 | 20.4 | 4.1 | 84.2 | 255.1 | 306.0 | 43.9 | 11.6 |
| p99 | 63.9 | 50.1 | 97.2 | 16.0 | 250.3 | 1419.4 | 5491.2 | 226.0 | 32.2 |
<!-- /results:sei_randles_wo -->
