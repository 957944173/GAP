# E1-E step120 vs GAP baseline (step120)

baseline `gap_step120_reference` -> test `step120_20260915_053854`

- macro EM: 0.4110 -> 0.4049 (delta -0.0061, -0.61 pp)
- micro EM: 0.4442 -> 0.4395 (delta -0.0047)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3890 | 0.3860 | -0.30 |
| triviaqa | 0.5770 | 0.5660 | -1.10 |
| popqa | 0.4130 | 0.4100 | -0.30 |
| hotpotqa | 0.4110 | 0.4010 | -1.00 |
| 2wikimultihopqa | 0.4450 | 0.4480 | +0.30 |
| musique | 0.1780 | 0.1670 | -1.10 |
| bamboogle | 0.4640 | 0.4560 | -0.80 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.85373 | 1.52028 | -0.33345 | -17.99% |
| search_queries_mean | 2.34331 | 2.01943 | -0.32388 | -13.82% |
| parallel_factor_mean | 1.26411 | 1.32833 | 0.06422 | +5.08% |
| parallel_sample_rate | 0.42292 | 0.43677 | 0.01385 | +3.27% |
| assistant_turns_mean_all_messages | 2.84198 | 2.51761 | -0.32437 | -11.41% |
| response_tokens_mean | 343.95206 | 322.35145 | -21.60061 | -6.28% |
| zero_search_rate | 0.00033 | 0.00039 | 0.00006 | +17.65% |
| no_answer_tag_rate | 0.00625 | 0.00143 | -0.00482 | -77.19% |

