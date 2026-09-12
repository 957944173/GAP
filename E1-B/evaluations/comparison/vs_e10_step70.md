# E1-B step120 vs E1-0 step70 (matched original-reward baseline)

baseline `e10_step70_reference` -> test `step120_20260911_235134`

- macro EM: 0.4006 -> 0.3947 (delta -0.0059, -0.59 pp)
- micro EM: 0.4325 -> 0.4325 (delta 0.0000)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3730 | 0.3810 | +0.80 |
| triviaqa | 0.5640 | 0.5620 | -0.20 |
| popqa | 0.3980 | 0.4030 | +0.50 |
| hotpotqa | 0.3940 | 0.3900 | -0.40 |
| 2wikimultihopqa | 0.4420 | 0.4400 | -0.20 |
| musique | 0.1690 | 0.1630 | -0.60 |
| bamboogle | 0.4640 | 0.4240 | -4.00 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.62956 | 1.29497 | -0.33458 | -20.53% |
| search_queries_mean | 2.12802 | 1.87483 | -0.25320 | -11.90% |
| parallel_factor_mean | 1.30589 | 1.44777 | 0.14188 | +10.86% |
| parallel_sample_rate | 0.43648 | 0.50573 | 0.06926 | +15.87% |
| assistant_turns_mean_all_messages | 2.62150 | 2.29298 | -0.32852 | -12.53% |
| response_tokens_mean | 324.10838 | 312.95264 | -11.15575 | -3.44% |
| zero_search_rate | 0.00031 | 0.00039 | 0.00008 | +25.00% |
| no_answer_tag_rate | 0.00744 | 0.00117 | -0.00627 | -84.25% |

