# E1-E step120 vs E1-B step120

baseline `e1b_step120_reference` -> test `step120_20260915_053854`

- macro EM: 0.3947 -> 0.4049 (delta 0.0101, 1.01 pp)
- micro EM: 0.4325 -> 0.4395 (delta 0.0069)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3810 | 0.3860 | +0.50 |
| triviaqa | 0.5620 | 0.5660 | +0.40 |
| popqa | 0.4030 | 0.4100 | +0.70 |
| hotpotqa | 0.3900 | 0.4010 | +1.10 |
| 2wikimultihopqa | 0.4400 | 0.4480 | +0.80 |
| musique | 0.1630 | 0.1670 | +0.40 |
| bamboogle | 0.4240 | 0.4560 | +3.20 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.29497 | 1.52028 | 0.22531 | +17.40% |
| search_queries_mean | 1.87483 | 2.01943 | 0.14461 | +7.71% |
| parallel_factor_mean | 1.44777 | 1.32833 | -0.11944 | -8.25% |
| parallel_sample_rate | 0.50573 | 0.43677 | -0.06896 | -13.64% |
| assistant_turns_mean_all_messages | 2.29298 | 2.51761 | 0.22463 | +9.80% |
| response_tokens_mean | 312.95264 | 322.35145 | 9.39881 | +3.00% |
| zero_search_rate | 0.00039 | 0.00039 | 0.00000 | +0.00% |
| no_answer_tag_rate | 0.00117 | 0.00143 | 0.00025 | +21.67% |

