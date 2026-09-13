# E1-D step120 per-benchmark behavior, vs GAP step120 and E1-A step70

## nq

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.3890 | 0.3780 | 0.3880 | -0.0010 | +0.0100 |
| retrieval_rounds_mean | 1.4726 | 1.2640 | 1.3399 | -0.1327 | +0.0759 |
| search_queries_mean | 2.1197 | 1.9230 | 2.0078 | -0.1119 | +0.0848 |
| parallel_factor_mean | 1.5746 | 1.6302 | 1.6118 | +0.0372 | -0.0184 |
| parallel_sample_rate | 0.5241 | 0.5252 | 0.5440 | +0.0199 | +0.0188 |
| assistant_turns_mean_all_messages | 2.4680 | 2.2630 | 2.3380 | -0.1300 | +0.0750 |
| response_tokens_mean | 328.9090 | 313.0530 | 324.7440 | -4.1650 | +11.6910 |
| zero_search_rate | 0.0006 | 0.0003 | 0.0003 | -0.0003 | +0.0000 |
| no_answer_tag_rate | 0.0028 | 0.0014 | 0.0025 | -0.0003 | +0.0011 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## triviaqa

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.5770 | 0.5630 | 0.5680 | -0.0090 | +0.0050 |
| retrieval_rounds_mean | 1.4535 | 1.2838 | 1.3645 | -0.0890 | +0.0807 |
| search_queries_mean | 2.0273 | 1.8619 | 1.9474 | -0.0799 | +0.0855 |
| parallel_factor_mean | 1.5011 | 1.5421 | 1.5293 | +0.0282 | -0.0127 |
| parallel_sample_rate | 0.4610 | 0.4656 | 0.4742 | +0.0132 | +0.0086 |
| assistant_turns_mean_all_messages | 2.4480 | 2.2820 | 2.3630 | -0.0850 | +0.0810 |
| response_tokens_mean | 316.0630 | 302.5670 | 312.4130 | -3.6500 | +9.8460 |
| zero_search_rate | 0.0001 | 0.0001 | 0.0001 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0026 | 0.0011 | 0.0015 | -0.0011 | +0.0004 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## popqa

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4130 | 0.3970 | 0.4080 | -0.0050 | +0.0110 |
| retrieval_rounds_mean | 1.7213 | 1.5006 | 1.5846 | -0.1367 | +0.0840 |
| search_queries_mean | 2.1088 | 1.8912 | 2.0049 | -0.1039 | +0.1136 |
| parallel_factor_mean | 1.3033 | 1.3419 | 1.3441 | +0.0408 | +0.0022 |
| parallel_sample_rate | 0.2998 | 0.3140 | 0.3225 | +0.0227 | +0.0085 |
| assistant_turns_mean_all_messages | 2.7090 | 2.4900 | 2.5790 | -0.1300 | +0.0890 |
| response_tokens_mean | 332.0630 | 312.7790 | 323.8520 | -8.2110 | +11.0730 |
| zero_search_rate | 0.0009 | 0.0008 | 0.0007 | -0.0002 | -0.0001 |
| no_answer_tag_rate | 0.0086 | 0.0089 | 0.0060 | -0.0026 | -0.0030 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## hotpotqa

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4110 | 0.3970 | 0.4030 | -0.0080 | +0.0060 |
| retrieval_rounds_mean | 1.8355 | 1.6155 | 1.7488 | -0.0867 | +0.1333 |
| search_queries_mean | 2.3262 | 2.1210 | 2.2446 | -0.0816 | +0.1236 |
| parallel_factor_mean | 1.4133 | 1.4608 | 1.4313 | +0.0180 | -0.0295 |
| parallel_sample_rate | 0.4639 | 0.4791 | 0.4626 | -0.0014 | -0.0165 |
| assistant_turns_mean_all_messages | 2.8300 | 2.6120 | 2.7450 | -0.0850 | +0.1330 |
| response_tokens_mean | 341.8300 | 321.5920 | 334.1160 | -7.7140 | +12.5240 |
| zero_search_rate | 0.0003 | 0.0001 | 0.0000 | -0.0003 | -0.0001 |
| no_answer_tag_rate | 0.0030 | 0.0028 | 0.0030 | +0.0000 | +0.0001 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## 2wikimultihopqa

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4450 | 0.4380 | 0.4410 | -0.0040 | +0.0030 |
| retrieval_rounds_mean | 2.3295 | 1.9514 | 2.1546 | -0.1749 | +0.2032 |
| search_queries_mean | 2.8487 | 2.5052 | 2.6895 | -0.1592 | +0.1843 |
| parallel_factor_mean | 1.3341 | 1.4269 | 1.3722 | +0.0380 | -0.0547 |
| parallel_sample_rate | 0.4979 | 0.5336 | 0.5085 | +0.0106 | -0.0251 |
| assistant_turns_mean_all_messages | 3.3100 | 2.9400 | 3.1440 | -0.1660 | +0.2040 |
| response_tokens_mean | 377.4640 | 349.1890 | 368.8640 | -8.6000 | +19.6750 |
| zero_search_rate | 0.0000 | 0.0001 | 0.0000 | +0.0000 | -0.0001 |
| no_answer_tag_rate | 0.0099 | 0.0091 | 0.0067 | -0.0033 | -0.0025 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## musique

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.1780 | 0.1680 | 0.1770 | -0.0010 | +0.0090 |
| retrieval_rounds_mean | 2.6271 | 2.2124 | 2.4285 | -0.1987 | +0.2161 |
| search_queries_mean | 2.9287 | 2.5761 | 2.7632 | -0.1655 | +0.1871 |
| parallel_factor_mean | 1.1848 | 1.2622 | 1.2154 | +0.0305 | -0.0469 |
| parallel_sample_rate | 0.2928 | 0.3426 | 0.3123 | +0.0195 | -0.0303 |
| assistant_turns_mean_all_messages | 3.5990 | 3.1990 | 3.4130 | -0.1860 | +0.2140 |
| response_tokens_mean | 399.4950 | 368.8080 | 391.8570 | -7.6380 | +23.0490 |
| zero_search_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0066 | 0.0058 | 0.0079 | +0.0012 | +0.0021 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## bamboogle

| metric | GAP step120 | E1-A step70 | E1-D step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4640 | 0.4480 | 0.4560 | -0.0080 | +0.0080 |
| retrieval_rounds_mean | 1.9600 | 1.8560 | 2.0640 | +0.1040 | +0.2080 |
| search_queries_mean | 2.1040 | 2.0400 | 2.2320 | +0.1280 | +0.1920 |
| parallel_factor_mean | 1.1267 | 1.1587 | 1.1238 | -0.0029 | -0.0349 |
| parallel_sample_rate | 0.1440 | 0.1840 | 0.1520 | +0.0080 | -0.0320 |
| assistant_turns_mean_all_messages | 2.9600 | 2.8560 | 3.0640 | +0.1040 | +0.2080 |
| response_tokens_mean | 294.0880 | 287.2480 | 303.2000 | +9.1120 | +15.9520 |
| zero_search_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

