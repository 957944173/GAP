# E1-E step120 vs E1-0 step70 (matched original-reward baseline)

baseline `e10_step70_reference` -> test `step120_20260915_053854`

- macro EM: 0.4006 -> 0.4049 (delta 0.0043, 0.43 pp)
- micro EM: 0.4325 -> 0.4395 (delta 0.0070)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3730 | 0.3860 | +1.30 |
| triviaqa | 0.5640 | 0.5660 | +0.20 |
| popqa | 0.3980 | 0.4100 | +1.20 |
| hotpotqa | 0.3940 | 0.4010 | +0.70 |
| 2wikimultihopqa | 0.4420 | 0.4480 | +0.60 |
| musique | 0.1690 | 0.1670 | -0.20 |
| bamboogle | 0.4640 | 0.4560 | -0.80 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.62956 | 1.52028 | -0.10928 | -6.71% |
| search_queries_mean | 2.12802 | 2.01943 | -0.10859 | -5.10% |
| parallel_factor_mean | 1.30589 | 1.32833 | 0.02244 | +1.72% |
| parallel_sample_rate | 0.43648 | 0.43677 | 0.00029 | +0.07% |
| assistant_turns_mean_all_messages | 2.62150 | 2.51761 | -0.10389 | -3.96% |
| response_tokens_mean | 324.10838 | 322.35145 | -1.75694 | -0.54% |
| zero_search_rate | 0.00031 | 0.00039 | 0.00008 | +25.00% |
| no_answer_tag_rate | 0.00744 | 0.00143 | -0.00602 | -80.84% |

