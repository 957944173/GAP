# E1-E step120 vs E1-A step70 (E1-E start)

baseline `e1a_step70_reference` -> test `step120_20260915_053854`

- macro EM: 0.3984 -> 0.4049 (delta 0.0064, 0.64 pp)
- micro EM: 0.4317 -> 0.4395 (delta 0.0077)

| benchmark | baseline EM | test EM | delta (pp) |
|---|---|---|---|
| nq | 0.3780 | 0.3860 | +0.80 |
| triviaqa | 0.5630 | 0.5660 | +0.30 |
| popqa | 0.3970 | 0.4100 | +1.30 |
| hotpotqa | 0.3970 | 0.4010 | +0.40 |
| 2wikimultihopqa | 0.4380 | 0.4480 | +1.00 |
| musique | 0.1680 | 0.1670 | -0.10 |
| bamboogle | 0.4480 | 0.4560 | +0.80 |

| behavior metric | baseline | test | delta | delta % |
|---|---|---|---|---|
| retrieval_rounds_mean | 1.59774 | 1.52028 | -0.07746 | -4.85% |
| search_queries_mean | 2.10363 | 2.01943 | -0.08420 | -4.00% |
| parallel_factor_mean | 1.31663 | 1.32833 | 0.01170 | +0.89% |
| parallel_sample_rate | 0.44124 | 0.43677 | -0.00447 | -1.01% |
| assistant_turns_mean_all_messages | 2.59049 | 2.51761 | -0.07288 | -2.81% |
| response_tokens_mean | 323.33561 | 322.35145 | -0.98417 | -0.30% |
| zero_search_rate | 0.00029 | 0.00039 | 0.00010 | +33.33% |
| no_answer_tag_rate | 0.00568 | 0.00143 | -0.00426 | -74.91% |

