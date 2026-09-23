# `rq`

![rq circuit diagram](../assets/circuits/rq.svg#only-light)
![rq circuit diagram](../assets/circuits/rq-dark.svg#only-dark)
{: style="text-align:center" }

![rq Nyquist plot](../assets/circuits/rq-nyquist.svg#only-light)
![rq Nyquist plot](../assets/circuits/rq-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:rq -->
`R0-(R1,CPE1)`, 1000 synthetic spectra. Inference costs 1.58 ms/spectrum against 1.63 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 28 |
| library defaults | 81.20% | 75 | 186 | 393 | 112 |
| truth x/div 5 | 88.00% | 55 | 121 | 222 | 83 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **37** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 47 |
| library defaults | 88.70% | 84 | 453 | 8964 | 148 |
| truth x/div 5 | 99.50% | 55 | 182 | 1356 | 106 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **56** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` |
|---|---|---|---|---|
| median | 0.6 | 0.7 | 3.3 | 0.5 |
| p90 | 4.2 | 2.5 | 15.8 | 2.8 |
| p99 | 86.4 | 6.1 | 71.1 | 8.0 |
<!-- /results:rq -->
