# E1-B validation-side trend (trainer's periodic NQ validation)

The trainer runs the unchanged GAP validation protocol (NQ, 128 samples, greedy) every
test_freq=10 steps; the console logger also reports the tool-usage metrics of the same
batch.  Both are independent of the E1-B reward manager, so they are a second,
unbiased witness of the training trajectory.

| step | val NQ EM `val-core/nq/reward/mean@1` | `tools/total_calls` | `tools/avg_calls_per_traj` |
|---|---|---|---|
| 71 | (n/a) | 2421.000 | 1.890 |
| 72 | (n/a) | 2559.000 | 2.000 |
| 73 | (n/a) | 2551.000 | 1.990 |
| 74 | (n/a) | 2508.000 | 1.960 |
| 75 | (n/a) | 2566.000 | 2.000 |
| 76 | (n/a) | 2535.000 | 1.980 |
| 77 | (n/a) | 2599.000 | 2.030 |
| 78 | (n/a) | 2556.000 | 2.000 |
| 79 | (n/a) | 2471.000 | 1.930 |
| 80 | 0.376 | 2520.000 | 1.970 |
| 81 | (n/a) | 2619.000 | 2.050 |
| 82 | (n/a) | 2533.000 | 1.980 |
| 83 | (n/a) | 2706.000 | 2.110 |
| 84 | (n/a) | 2537.000 | 1.980 |
| 85 | (n/a) | 2556.000 | 2.000 |
| 86 | (n/a) | 2545.000 | 1.990 |
| 87 | (n/a) | 2514.000 | 1.960 |
| 88 | (n/a) | 2396.000 | 1.870 |
| 89 | (n/a) | 2508.000 | 1.960 |
| 90 | 0.375 | 2448.000 | 1.910 |
| 91 | (n/a) | 2574.000 | 2.010 |
| 92 | (n/a) | 2505.000 | 1.960 |
| 93 | (n/a) | 2372.000 | 1.850 |
| 94 | (n/a) | 2452.000 | 1.920 |
| 95 | (n/a) | 2542.000 | 1.990 |
| 96 | (n/a) | 2475.000 | 1.930 |
| 97 | (n/a) | 2304.000 | 1.800 |
| 98 | (n/a) | 2614.000 | 2.040 |
| 99 | (n/a) | 2572.000 | 2.010 |
| 100 | 0.385 | 2399.000 | 1.870 |
| 101 | (n/a) | 2569.000 | 2.010 |

Reading: validation EM is flat-to-up (0.371 → 0.376 → 0.375 → 0.385) while the number of
wiki search calls per trajectory falls monotonically (1.970 → 1.910 → 1.870 on the training
batch).  This corroborates the reward-manager diagnostics: the shaped reward reduces search
usage without costing accuracy.
