# `two_rc_l`

![two_rc_l circuit diagram](../assets/circuits/two_rc_l.svg#only-light)
![two_rc_l circuit diagram](../assets/circuits/two_rc_l-dark.svg#only-dark)
{: style="text-align:center" }

![two_rc_l Nyquist plot](../assets/circuits/two_rc_l-nyquist.svg#only-light)
![two_rc_l Nyquist plot](../assets/circuits/two_rc_l-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:two_rc_l -->
`L0-R0-(R1,C1)-(R2,C2)`, 1000 synthetic spectra. Inference costs 1.52 ms/spectrum against 2.06 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 53 |
| library defaults | 69.30% | 254 | 528 | 1330 | 285 |
| truth x/div 5 | 71.50% | 118 | 317 | 738 | 171 |
| **ml guess** | **100.00%** | **12** | **13** | **39** | **53** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 76 |
| library defaults | 90.90% | 295 | 2880 | 10628 | 400 |
| truth x/div 5 | 91.40% | 203 | 1258 | 7074 | 312 |
| **ml guess** | **100.00%** | **1** | **13** | **52** | **76** |

Relative error of the ml guess, before fitting (%):

| | `L0.l` | `R0.r` | `R1.r` | `C1.c` | `R2.r` | `C2.c` |
|---|---|---|---|---|---|---|
| median | 1.8 | 0.9 | 0.8 | 1.4 | 1.6 | 3.2 |
| p90 | 8.3 | 16.9 | 3.5 | 6.8 | 11.8 | 24.5 |
| p99 | 62.8 | 269.3 | 13.5 | 35.5 | 57.1 | 137.5 |
<!-- /results:two_rc_l -->
