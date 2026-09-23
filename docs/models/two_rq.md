# `two_rq`

![two_rq circuit diagram](../assets/circuits/two_rq.svg#only-light)
![two_rq circuit diagram](../assets/circuits/two_rq-dark.svg#only-dark)
{: style="text-align:center" }

![two_rq Nyquist plot](../assets/circuits/two_rq-nyquist.svg#only-light)
![two_rq Nyquist plot](../assets/circuits/two_rq-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:two_rq -->
`R0-(R1,CPE1)-(R2,CPE2)`, 1000 synthetic spectra. Inference costs 0.72 ms/spectrum against 1.99 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 76 |
| library defaults | 51.20% | 368 | 717 | 1641 | 366 |
| truth x/div 5 | 55.10% | 171 | 458 | 1476 | 276 |
| **ml guess** | **99.60%** | **15** | **62** | **246** | **91** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 101 |
| library defaults | 77.90% | 415 | 5348 | 44383 | 685 |
| truth x/div 5 | 77.30% | 352 | 2717 | 24850 | 652 |
| **ml guess** | **99.70%** | **15** | **62** | **294** | **131** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|
| median | 1.2 | 5.4 | 7.6 | 1.7 | 8.6 | 36.7 | 9.6 |
| p90 | 10.5 | 24.3 | 34.6 | 7.5 | 35.5 | 113.7 | 27.6 |
| p99 | 96.6 | 94.2 | 145.8 | 16.8 | 115.9 | 380.5 | 50.0 |
<!-- /results:two_rq -->
