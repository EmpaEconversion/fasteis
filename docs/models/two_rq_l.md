# `two_rq_l`

![two_rq_l circuit diagram](../assets/circuits/two_rq_l.svg#only-light)
![two_rq_l circuit diagram](../assets/circuits/two_rq_l-dark.svg#only-dark)
{: style="text-align:center" }

![two_rq_l Nyquist plot](../assets/circuits/two_rq_l-nyquist.svg#only-light)
![two_rq_l Nyquist plot](../assets/circuits/two_rq_l-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:two_rq_l -->
`L0-R0-(R1,CPE1)-(R2,CPE2)`, 1000 synthetic spectra. Inference costs 0.72 ms/spectrum against 3.23 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 86 |
| library defaults | 57.10% | 587 | 1180 | 2826 | 604 |
| truth x/div 5 | 50.30% | 260 | 687 | 2509 | 366 |
| **ml guess** | **98.80%** | **17** | **188** | **668** | **120** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 130 |
| library defaults | 83.90% | 633 | 11359 | 81777 | 945 |
| truth x/div 5 | 80.00% | 507 | 4160 | 36863 | 928 |
| **ml guess** | **99.00%** | **17** | **192** | **1131** | **148** |

Relative error of the ml guess, before fitting (%):

| | `L0.l` | `R0.r` | `R1.r` | `CPE1.q` | `CPE1.alpha` | `R2.r` | `CPE2.q` | `CPE2.alpha` |
|---|---|---|---|---|---|---|---|---|
| median | 4.8 | 7.6 | 8.1 | 13.6 | 2.7 | 13.5 | 30.9 | 6.6 |
| p90 | 13.6 | 47.7 | 50.2 | 61.1 | 12.5 | 54.6 | 145.0 | 24.1 |
| p99 | 85.9 | 237.1 | 209.8 | 268.5 | 31.5 | 165.9 | 711.8 | 46.1 |
<!-- /results:two_rq_l -->

## Benchmarks against real data

<!-- results:two_rq_l_real -->
`two_rq_l` against 201 measured spectra. Ground truth is not known, so 'converged' means within tolerance of the best chi-square reached.

| source of initial parameters | converged | med sweeps | med ms | med chi2 |
|---|---|---|---|---|
| library defaults | 93.03% | 5038 | 27.87 | 2.081e-03 |
| **ml guess** | **95.52%** | **2961** | **16.98** | **2.021e-03** |
| differential_evolution | 67.16% | 203285 | 1288.50 | 3.321e-03 |
<!-- /results:two_rq_l_real -->
