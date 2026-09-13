# E1-D step120 vs GAP baseline (step120)

baseline `gap_step120_reference_evaluations` -> test `step120_20260913_165834`

- macro EM: 0.4110 -> 0.4059 (delta -0.0051, -0.51 pp)
- micro EM: 0.4442 -> 0.4386 (delta -0.0056)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3890 | 0.3880 | -0.10 |
| triviaqa | 0.5770 | 0.5680 | -0.90 |
| popqa | 0.4130 | 0.4080 | -0.50 |
| hotpotqa | 0.4110 | 0.4030 | -0.80 |
| 2wikimultihopqa | 0.4450 | 0.4410 | -0.40 |
| musique | 0.1780 | 0.1770 | -0.10 |
| bamboogle | 0.4640 | 0.4560 | -0.80 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.85373 | 1.72336 | -0.13037 | -7.03% |
| search_queries_mean | 2.34331 | 2.23146 | -0.11185 | -4.77% |
| parallel_factor_mean | 1.26411 | 1.29483 | 0.03072 | +2.43% |
| parallel_sample_rate | 0.42292 | 0.43669 | 0.01377 | +3.26% |
| assistant_turns_mean_all_messages | 2.84198 | 2.71752 | -0.12446 | -4.38% |
| response_tokens_mean | 343.95206 | 337.07960 | -6.87246 | -2.00% |
| zero_search_rate | 0.00033 | 0.00021 | -0.00012 | -35.29% |
| no_answer_tag_rate | 0.00625 | 0.00455 | -0.00170 | -27.19% |

