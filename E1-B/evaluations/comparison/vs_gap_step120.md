# E1-B step120 vs GAP baseline (step120)

baseline `gap_step120_reference` -> test `step120_20260911_235134`

- macro EM: 0.4110 -> 0.3947 (delta -0.0163, -1.63 pp)
- micro EM: 0.4442 -> 0.4325 (delta -0.0116)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3890 | 0.3810 | -0.80 |
| triviaqa | 0.5770 | 0.5620 | -1.50 |
| popqa | 0.4130 | 0.4030 | -1.00 |
| hotpotqa | 0.4110 | 0.3900 | -2.10 |
| 2wikimultihopqa | 0.4450 | 0.4400 | -0.50 |
| musique | 0.1780 | 0.1630 | -1.50 |
| bamboogle | 0.4640 | 0.4240 | -4.00 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.85373 | 1.29497 | -0.55876 | -30.14% |
| search_queries_mean | 2.34331 | 1.87483 | -0.46849 | -19.99% |
| parallel_factor_mean | 1.26411 | 1.44777 | 0.18367 | +14.53% |
| parallel_sample_rate | 0.42292 | 0.50573 | 0.08281 | +19.58% |
| assistant_turns_mean_all_messages | 2.84198 | 2.29298 | -0.54901 | -19.32% |
| response_tokens_mean | 343.95206 | 312.95264 | -30.99942 | -9.01% |
| zero_search_rate | 0.00033 | 0.00039 | 0.00006 | +17.65% |
| no_answer_tag_rate | 0.00625 | 0.00117 | -0.00508 | -81.25% |

