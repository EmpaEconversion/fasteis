# Models

Circuits are trained on synthetic data, and summarised in the table below.
See [Training](../training.md) for how these models work and are built.

- Percentages are rates of convergence starting from parameters multiplied or divided by 5 from real values vs the model intial guess.
- 'floor' is the median impedance calculation count starting from the true parameters.
- 'excess' is extra calculations beyond 'floor', median and 90th percentile are shown.

<!-- results:library -->
| name | circuit | params | params * / 5 | ml guess | floor | ml excess med | p90 |
|---|---|---|---|---|---|---|---|
| [`rc`](rc.md) | `R0-(R1,C1)` | 3 | 98.3% | **100.0%** | 22| **6** | 7 |
| [`rc_l`](rc_l.md) | `L0-R0-(R1,C1)` | 4 | 94.8% | **100.0%** | 28| **0** | 9 |
| [`rq`](rq.md) | `R0-(R1,CPE1)` | 4 | 88.0% | **100.0%** | 28| **0** | 9 |
| [`rq_l`](rq_l.md) | `L0-R0-(R1,CPE1)` | 5 | 87.4% | **100.0%** | 34| **0** | 11 |
| [`two_rc`](two_rc.md) | `R0-(R1,C1)-(R2,C2)` | 5 | 75.5% | **100.0%** | 44| **0** | 11 |
| [`two_rc_l`](two_rc_l.md) | `L0-R0-(R1,C1)-(R2,C2)` | 6 | 71.5% | **100.0%** | 53| **12** | 13 |
| [`two_rq`](two_rq.md) | `R0-(R1,CPE1)-(R2,CPE2)` | 7 | 55.1% | **99.6%** | 76| **15** | 62 |
| [`two_rq_l`](two_rq_l.md) | `L0-R0-(R1,CPE1)-(R2,CPE2)` | 8 | 50.3% | **98.8%** | 86| **17** | 188 |
| [`randles`](randles.md) | `R0-(R1-W1,CPE1)` | 5 | 70.8% | **99.9%** | 45| **0** | 11 |
| [`sei_randles`](sei_randles.md) | `R0-(R1,CPE1)-(R2-W2,CPE2)` | 8 | 41.6% | **99.1%** | 103| **0** | 52 |
| [`sei_randles_wo`](sei_randles_wo.md) | `R0-(R1,CPE1)-(R2-Wo2,CPE2)` | 9 | 26.0% | **91.3%** | 276| **16** | 252 |
<!-- /results:library -->

Each circuit's page has more detailed benchmarks.
