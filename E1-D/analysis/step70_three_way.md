# E1-D step70: three-way comparison (matches E1-D_task.md §14)

E1-D step70 evaluation: `E1-D/eval_results/step70_20260913_024024` (original 7-benchmark protocol, 51,201 samples, greedy).

| run | filter | reward | macro EM | micro EM | search rounds | queries | parallel factor | no-answer |
|---|---|---|---|---|---|---|---|---|
| **E1-D step70** | **EM** | **shaped** | **0.3976** | **0.4333** | **1.6237** | 2.1274 | 1.3102 | 0.00725 |
| GAP-70 (E1-0 step70: EM filter + EM reward) | EM | EM | 0.4006 | 0.4325 | 1.6296 | 2.1280 | 1.3059 | 0.00744 |
| E1-A-70 (shaped filter + shaped reward) | shaped | shaped | 0.3984 | 0.4317 | 1.5977 | 2.1036 | 1.3166 | 0.00568 |
| GAP-60 (start of the branch) | EM | EM | 0.4007 | 0.4351 | 1.5889 | 2.0922 | 1.3167 | 0.00580 |

## Answers to §14's two questions

**1. E1-D70 vs GAP70** (same EM filter; GAP adds nothing, E1-D adds objective shaping): micro EM 0.4325 -> 0.4333 (+0.08 pp), macro EM 0.4006 -> 0.3976 (-0.30 pp); search rounds 1.6296 -> 1.6237 (-0.36 %), queries 2.1280 -> 2.1274.
**2. E1-D70 vs E1-A70** (same shaped reward; only the filter channel differs): micro EM 0.4317 -> 0.4333 (+0.16 pp), macro EM 0.3984 -> 0.3976 (-0.09 pp); search rounds 1.5977 -> 1.6237 (+1.62 %), queries 2.1036 -> 2.1274.

Paired per-prompt statistics are produced by `run_e1d_analysis.sh` after step120 and are reported in the final report (§17/§22).
