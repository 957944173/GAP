# E1-E step120 vs E1-D step120 (mandated four-way)

baseline `e1d_step120_reference` -> test `step120_20260915_053854`

- macro EM: 0.4059 -> 0.4049 (delta -0.0010, -0.10 pp)
- micro EM: 0.4386 -> 0.4395 (delta 0.0009)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3880 | 0.3860 | -0.20 |
| triviaqa | 0.5680 | 0.5660 | -0.20 |
| popqa | 0.4080 | 0.4100 | +0.20 |
| hotpotqa | 0.4030 | 0.4010 | -0.20 |
| 2wikimultihopqa | 0.4410 | 0.4480 | +0.70 |
| musique | 0.1770 | 0.1670 | -1.00 |
| bamboogle | 0.4560 | 0.4560 | +0.00 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.72336 | 1.52028 | -0.20308 | -11.78% |
| search_queries_mean | 2.23146 | 2.01943 | -0.21203 | -9.50% |
| parallel_factor_mean | 1.29483 | 1.32833 | 0.03350 | +2.59% |
| parallel_sample_rate | 0.43669 | 0.43677 | 0.00008 | +0.02% |
| assistant_turns_mean_all_messages | 2.71752 | 2.51761 | -0.19991 | -7.36% |
| response_tokens_mean | 337.07960 | 322.35145 | -14.72815 | -4.37% |
| zero_search_rate | 0.00021 | 0.00039 | 0.00018 | +81.82% |
| no_answer_tag_rate | 0.00455 | 0.00143 | -0.00312 | -68.67% |

