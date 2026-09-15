# E1-E training-side diagnostics (step 60 -> 70)

| metric | E1-E (shaped) | baseline E1-0 (original EM) | delta |
|---|---|---|---|
| n_rollouts | 8960 | 147200 | -138240 |
| n_groups | 1119 | 18400 | -17281 |
| n_calls | 10 | 115 | -105 |
| em_rate | 0.4734 | 0.4539 | 0.0195 |
| reward_mean | 0.4739 | 0.4539 | 0.0200 |
| efficiency_mean | 0.0095 | None |  |
| efficiency_nonzero_rate | 0.0097 | None |  |
| rounds_mean | 1.5488 | 1.4407 | 0.1080 |
| queries_mean | 2.1261 | 1.9898 | 0.1364 |
| parallel_factor_mean | 1.5048 | 1.5091 | -0.0043 |
| parallel_sample_rate | 0.4922 | 0.4829 | 0.0093 |
| multi_query_round_rate | 0.4922 | 0.4829 | 0.0093 |
| turns_mean | 2.5482 | 2.4406 | 0.1076 |
| tokens_mean | 1349.3299 | 1277.7553 | 71.5746 |
| zero_search_rate | 0.0000 | 0.0002 | -0.0002 |
| filter_retention_baseline | 300 | 340 | -40 |
| filter_retention_shaped | None | None |  |
| filter_newly_added | None | None |  |
| efficiency_active_groups | 20 | 95 | -75 |
| efficiency_supervision_ratio | 0.0625 | None |  |

## per-step (E1-E)
| step | rollouts | groups | EM | reward mean | E mean | E>0 rate | rounds | queries | base kept | shaped kept | newly added |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 61 | 256 | 32 | 0.5586 | 0.5602 | 0.0312 | 0.0312 | 1.7227 | 2.2383 | 0 | 0 | 0 |
| 62 | 1536 | 32 | 0.4557 | 0.4560 | 0.0052 | 0.0052 | 1.6126 | 2.1628 | 0 | 0 | 0 |
| 63 | 256 | 32 | 0.5430 | 0.5436 | 0.0117 | 0.0117 | 2.0586 | 2.7031 | 0 | 0 | 0 |
| 64 | 1536 | 32 | 0.4564 | 0.4569 | 0.0098 | 0.0098 | 1.4551 | 2.0052 | 0 | 0 | 0 |
| 65 | 256 | 32 | 0.5625 | 0.5639 | 0.0273 | 0.0273 | 2.0312 | 3.1523 | 0 | 0 | 0 |
| 66 | 1536 | 32 | 0.5306 | 0.5309 | 0.0065 | 0.0065 | 1.4134 | 1.9792 | 0 | 0 | 0 |
| 67 | 1536 | 32 | 0.4447 | 0.4449 | 0.0039 | 0.0039 | 1.4512 | 2.0156 | 0 | 0 | 0 |
| 68 | 256 | 32 | 0.5586 | 0.5600 | 0.0273 | 0.0273 | 1.7617 | 2.2812 | 0 | 0 | 0 |
| 69 | 1536 | 32 | 0.4199 | 0.4203 | 0.0078 | 0.0091 | 1.5430 | 2.1113 | 0 | 0 | 0 |
| 70 | 256 | 32 | 0.5039 | 0.5057 | 0.0352 | 0.0352 | 1.7812 | 2.3945 | 0 | 0 | 0 |
