# `randles`

![randles circuit diagram](../assets/circuits/randles.svg#only-light)
![randles circuit diagram](../assets/circuits/randles-dark.svg#only-dark)
{: style="text-align:center" }

![randles Nyquist plot](../assets/circuits/randles-nyquist.svg#only-light)
![randles Nyquist plot](../assets/circuits/randles-nyquist-dark.svg#only-dark)
{: style="text-align:center" }

<!-- results:randles -->
`R0-(R1-W1,CPE1)`, 2000 synthetic spectra. Inference costs 0.73 ms/spectrum against 1.25 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 45 |
| library defaults | 57.55% | 102 | 318 | 726 | 204 |
| truth x/div 5 | 70.75% | 78 | 205 | 452 | 124 |
| **ml guess** | **99.85%** | **0** | **11** | **57** | **45** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 66 |
| library defaults | 78.00% | 170 | 1884 | 19384 | 333 |
| truth x/div 5 | 88.15% | 103 | 720 | 8770 | 200 |
| **ml guess** | **99.85%** | **0** | **11** | **65** | **66** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `W1.aw` | `CPE1.q` | `CPE1.alpha` |
|---|---|---|---|---|---|
| median | 1.4 | 1.5 | 1.3 | 3.5 | 0.6 |
| p90 | 3.7 | 25.6 | 9.7 | 16.5 | 3.4 |
| p99 | 34.1 | 101.6 | 46.7 | 54.0 | 10.9 |
<!-- /results:randles -->
