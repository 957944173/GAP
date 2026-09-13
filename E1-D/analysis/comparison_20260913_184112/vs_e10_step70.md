# E1-D step120 vs E1-0 step70 (matched original-reward baseline)

baseline `e10_step70_reference_evaluations` -> test `step120_20260913_165834`

- macro EM: 0.4006 -> 0.4059 (delta 0.0053, 0.53 pp)
- micro EM: 0.4325 -> 0.4386 (delta 0.0060)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3730 | 0.3880 | +1.50 |
| triviaqa | 0.5640 | 0.5680 | +0.40 |
| popqa | 0.3980 | 0.4080 | +1.00 |
| hotpotqa | 0.3940 | 0.4030 | +0.90 |
| 2wikimultihopqa | 0.4420 | 0.4410 | -0.10 |
| musique | 0.1690 | 0.1770 | +0.80 |
| bamboogle | 0.4640 | 0.4560 | -0.80 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.62956 | 1.72336 | 0.09381 | +5.76% |
| search_queries_mean | 2.12802 | 2.23146 | 0.10344 | +4.86% |
| parallel_factor_mean | 1.30589 | 1.29483 | -0.01106 | -0.85% |
| parallel_sample_rate | 0.43648 | 0.43669 | 0.00021 | +0.05% |
| assistant_turns_mean_all_messages | 2.62150 | 2.71752 | 0.09602 | +3.66% |
| response_tokens_mean | 324.10838 | 337.07960 | 12.97122 | +4.00% |
| zero_search_rate | 0.00031 | 0.00021 | -0.00010 | -31.25% |
| no_answer_tag_rate | 0.00744 | 0.00455 | -0.00289 | -38.85% |

