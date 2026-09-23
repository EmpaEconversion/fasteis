# `rc`

![rc circuit diagram](../assets/circuits/rc.svg#only-light)
![rc circuit diagram](../assets/circuits/rc-dark.svg#only-dark)
{: style="text-align:center" }

![rc Nyquist plot](../assets/circuits/rc-nyquist.svg#only-light)
![rc Nyquist plot](../assets/circuits/rc-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:rc -->
`R0-(R1,C1)`, 1000 synthetic spectra. Inference costs 0.24 ms/spectrum against 0.71 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 22 |
| library defaults | 99.20% | 56 | 99 | 311 | 78 |
| truth x/div 5 | 98.30% | 36 | 51 | 121 | 58 |
| **ml guess** | **100.00%** | **6** | **7** | **14** | **29** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 39 |
| library defaults | 100.00% | 50 | 114 | 994 | 89 |
| truth x/div 5 | 100.00% | 30 | 80 | 280 | 74 |
| **ml guess** | **100.00%** | **6** | **7** | **14** | **46** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `C1.c` |
|---|---|---|---|
| median | 0.7 | 0.7 | 3.1 |
| p90 | 3.3 | 2.7 | 16.5 |
| p99 | 51.3 | 5.8 | 44.3 |
<!-- /results:rc -->
