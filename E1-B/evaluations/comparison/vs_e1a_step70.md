# E1-B step120 vs E1-A step70 (E1-B start)

baseline `e1a_step70_reference` -> test `step120_20260911_235134`

- macro EM: 0.3984 -> 0.3947 (delta -0.0037, -0.37 pp)
- micro EM: 0.4317 -> 0.4325 (delta 0.0008)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3780 | 0.3810 | +0.30 |
| triviaqa | 0.5630 | 0.5620 | -0.10 |
| popqa | 0.3970 | 0.4030 | +0.60 |
| hotpotqa | 0.3970 | 0.3900 | -0.70 |
| 2wikimultihopqa | 0.4380 | 0.4400 | +0.20 |
| musique | 0.1680 | 0.1630 | -0.50 |
| bamboogle | 0.4480 | 0.4240 | -2.40 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.59774 | 1.29497 | -0.30277 | -18.95% |
| search_queries_mean | 2.10363 | 1.87483 | -0.22880 | -10.88% |
| parallel_factor_mean | 1.31663 | 1.44777 | 0.13114 | +9.96% |
| parallel_sample_rate | 0.44124 | 0.50573 | 0.06449 | +14.62% |
| assistant_turns_mean_all_messages | 2.59049 | 2.29298 | -0.29752 | -11.48% |
| response_tokens_mean | 323.33561 | 312.95264 | -10.38298 | -3.21% |
| zero_search_rate | 0.00029 | 0.00039 | 0.00010 | +33.33% |
| no_answer_tag_rate | 0.00568 | 0.00117 | -0.00451 | -79.38% |

