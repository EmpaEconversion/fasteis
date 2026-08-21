# `two_rc`

<!-- results:two_rc -->
`R0-(R1,C1)-(R2,C2)`, 1000 synthetic spectra. Inference costs 1.59 ms/spectrum against 2.20 ms for the fit it starts.

Plain LM:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 44 |
| library defaults | 67.40% | 168 | 374 | 1025 | 182 |
| truth x/div 5 | 75.50% | 79 | 215 | 475 | 124 |
| **ml guess** | **100.00%** | **0** | **11** | **22** | **45** |

`Circuit.fit()` / smart LM, which screens candidate starts:

| source of initial parameters | converged | excess med | p90 | p99 | med sweeps |
|---|---|---|---|---|---|
| floor (truth) | 100.00% | 0 | 0 | 0 | 65 |
| library defaults | 91.60% | 176 | 702 | 5782 | 258 |
| truth x/div 5 | 92.80% | 125 | 712 | 4519 | 207 |
| **ml guess** | **100.00%** | **0** | **11** | **22** | **66** |

Relative error of the ml guess, before fitting (%):

| | `R0.r` | `R1.r` | `C1.c` | `R2.r` | `C2.c` |
|---|---|---|---|---|---|
| median | 0.4 | 0.7 | 0.9 | 1.3 | 2.4 |
| p90 | 1.8 | 2.3 | 4.0 | 10.5 | 16.0 |
| p99 | 26.7 | 5.8 | 10.8 | 57.6 | 46.1 |
<!-- /results:two_rc -->
