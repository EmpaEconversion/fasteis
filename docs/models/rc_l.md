# `rc_l`

<!-- results:rc_l -->
`L0-R0-(R1,C1)`, 1000 synthetic spectra. Inference costs 1.67 ms/spectrum against 1.47 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 28 |
| library defaults | 96.40% | 99 | 251 | 425 | 130 |
| truth x/div 5 | 94.80% | 55 | 111 | 258 | 83 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **36** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 47 |
| library defaults | 99.20% | 94 | 407 | 3787 | 148 |
| truth x/div 5 | 99.90% | 63 | 185 | 882 | 111 |
| **ml guess** | **100.00%** | **0** | **9** | **18** | **55** |

Relative error of the ml guess, before fitting (%):

| | `L0.l` | `R0.r` | `R1.r` | `C1.c` |
|---|---|---|---|---|
| median | 1.7 | 0.9 | 0.5 | 1.1 |
| p90 | 5.9 | 15.9 | 2.2 | 5.2 |
| p99 | 50.1 | 253.4 | 8.3 | 18.0 |
<!-- /results:rc_l -->
