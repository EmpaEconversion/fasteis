# `rq_l`

![rq_l circuit diagram](../assets/circuits/rq_l.svg#only-light)
![rq_l circuit diagram](../assets/circuits/rq_l-dark.svg#only-dark)
{: style="text-align:center" }

![rq_l Nyquist plot](../assets/circuits/rq_l-nyquist.svg#only-light)
![rq_l Nyquist plot](../assets/circuits/rq_l-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:rq_l -->
`L0-R0-(R1,CPE1)`, 1000 synthetic spectra. Inference costs 1.55 ms/spectrum against 1.99 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 34 |
| library defaults | 79.80% | 135 | 317 | 504 | 182 |
| truth x/div 5 | 87.40% | 79 | 214 | 377 | 123 |
| **ml guess** | **100.00%** | **0** | **11** | **33** | **45** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 55 |
| library defaults | 90.00% | 160 | 1883 | 16233 | 241 |
| truth x/div 5 | 98.20% | 96 | 330 | 6952 | 161 |
| **ml guess** | **100.00%** | **0** | **11** | **69** | **66** |

Relative error of the ml guess, before fitting (%):

| | `L0.l` | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` |
|---|---|---|---|---|---|
| median | 1.6 | 1.5 | 0.8 | 2.8 | 0.6 |
| p90 | 6.9 | 29.1 | 3.3 | 14.6 | 2.8 |
| p99 | 42.5 | 245.1 | 11.9 | 48.5 | 12.1 |
<!-- /results:rq_l -->
