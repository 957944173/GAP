# E1-D step120 vs E1-A step70 (E1-D start)

baseline `e1a_step70_reference_evaluations` -> test `step120_20260913_165834`

- macro EM: 0.3984 -> 0.4059 (delta 0.0074, 0.74 pp)
- micro EM: 0.4317 -> 0.4386 (delta 0.0068)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3780 | 0.3880 | +1.00 |
| triviaqa | 0.5630 | 0.5680 | +0.50 |
| popqa | 0.3970 | 0.4080 | +1.10 |
| hotpotqa | 0.3970 | 0.4030 | +0.60 |
| 2wikimultihopqa | 0.4380 | 0.4410 | +0.30 |
| musique | 0.1680 | 0.1770 | +0.90 |
| bamboogle | 0.4480 | 0.4560 | +0.80 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.59774 | 1.72336 | 0.12562 | +7.86% |
| search_queries_mean | 2.10363 | 2.23146 | 0.12783 | +6.08% |
| parallel_factor_mean | 1.31663 | 1.29483 | -0.02180 | -1.66% |
| parallel_sample_rate | 0.44124 | 0.43669 | -0.00455 | -1.03% |
| assistant_turns_mean_all_messages | 2.59049 | 2.71752 | 0.12703 | +4.90% |
| response_tokens_mean | 323.33561 | 337.07960 | 13.74399 | +4.25% |
| zero_search_rate | 0.00029 | 0.00021 | -0.00008 | -26.67% |
| no_answer_tag_rate | 0.00568 | 0.00455 | -0.00113 | -19.93% |

