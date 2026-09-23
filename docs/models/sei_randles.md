# `sei_randles`

![sei_randles circuit diagram](../assets/circuits/sei_randles.svg#only-light)
![sei_randles circuit diagram](../assets/circuits/sei_randles-dark.svg#only-dark)
{: style="text-align:center" }

![sei_randles Nyquist plot](../assets/circuits/sei_randles-nyquist.svg#only-light)
![sei_randles Nyquist plot](../assets/circuits/sei_randles-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:sei_randles -->
`R0-(R1,CPE1)-(R2-W2,CPE2)`, 1000 synthetic spectra. Inference costs 2.35 ms/spectrum against 3.26 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 103 |
| library defaults | 30.40% | 400 | 1180 | 5233 | 726 |
| truth x/div 5 | 41.60% | 228 | 672 | 2386 | 416 |
| **ml guess** | **99.10%** | **0** | **52** | **334** | **103** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 148 |
| library defaults | 53.60% | 692 | 16594 | 80818 | 2018 |
| truth x/div 5 | 66.10% | 554 | 5597 | 37704 | 1148 |
| **ml guess** | **99.00%** | **0** | **66** | **1797** | **147** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `W2.aw` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|
| median | 1.3 | 2.5 | 4.7 | 0.9 | 8.1 | 5.7 | 10.6 | 3.4 |
| p90 | 4.7 | 15.6 | 23.9 | 4.7 | 42.9 | 41.6 | 58.6 | 16.2 |
| p99 | 58.1 | 66.7 | 71.1 | 14.1 | 149.4 | 171.9 | 194.2 | 36.9 |
<!-- /results:sei_randles -->
