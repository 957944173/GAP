# E1-E step120 per-benchmark behavior, vs GAP step120 and E1-A step70

## nq

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.3890 | 0.3780 | 0.3860 | -0.0030 | +0.0080 |
| retrieval_rounds_mean | 1.4726 | 1.2640 | 1.2061 | -0.2665 | -0.0579 |
| search_queries_mean | 2.1197 | 1.9230 | 1.8784 | -0.2413 | -0.0446 |
| parallel_factor_mean | 1.5746 | 1.6302 | 1.6443 | +0.0698 | +0.0142 |
| parallel_sample_rate | 0.5241 | 0.5252 | 0.5521 | +0.0280 | +0.0269 |
| assistant_turns_mean_all_messages | 2.4680 | 2.2630 | 2.2060 | -0.2620 | -0.0570 |
| response_tokens_mean | 328.9090 | 313.0530 | 315.1160 | -13.7930 | +2.0630 |
| zero_search_rate | 0.0006 | 0.0003 | 0.0011 | +0.0006 | +0.0008 |
| no_answer_tag_rate | 0.0028 | 0.0014 | 0.0011 | -0.0017 | -0.0003 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## triviaqa

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.5770 | 0.5630 | 0.5660 | -0.0110 | +0.0030 |
| retrieval_rounds_mean | 1.4535 | 1.2838 | 1.2394 | -0.2141 | -0.0444 |
| search_queries_mean | 2.0273 | 1.8619 | 1.8190 | -0.2084 | -0.0430 |
| parallel_factor_mean | 1.5011 | 1.5421 | 1.5497 | +0.0486 | +0.0077 |
| parallel_sample_rate | 0.4610 | 0.4656 | 0.4730 | +0.0120 | +0.0074 |
| assistant_turns_mean_all_messages | 2.4480 | 2.2820 | 2.2390 | -0.2090 | -0.0430 |
| response_tokens_mean | 316.0630 | 302.5670 | 300.6720 | -15.3910 | -1.8950 |
| zero_search_rate | 0.0001 | 0.0001 | 0.0000 | -0.0001 | -0.0001 |
| no_answer_tag_rate | 0.0026 | 0.0011 | 0.0005 | -0.0020 | -0.0006 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## popqa

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4130 | 0.3970 | 0.4100 | -0.0030 | +0.0130 |
| retrieval_rounds_mean | 1.7213 | 1.5006 | 1.3985 | -0.3227 | -0.1021 |
| search_queries_mean | 2.1088 | 1.8912 | 1.8024 | -0.3063 | -0.0888 |
| parallel_factor_mean | 1.3033 | 1.3419 | 1.3584 | +0.0551 | +0.0165 |
| parallel_sample_rate | 0.2998 | 0.3140 | 0.3224 | +0.0226 | +0.0084 |
| assistant_turns_mean_all_messages | 2.7090 | 2.4900 | 2.3930 | -0.3160 | -0.0970 |
| response_tokens_mean | 332.0630 | 312.7790 | 312.2430 | -19.8200 | -0.5360 |
| zero_search_rate | 0.0009 | 0.0008 | 0.0012 | +0.0003 | +0.0004 |
| no_answer_tag_rate | 0.0086 | 0.0089 | 0.0031 | -0.0054 | -0.0058 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## hotpotqa

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4110 | 0.3970 | 0.4010 | -0.0100 | +0.0040 |
| retrieval_rounds_mean | 1.8355 | 1.6155 | 1.5809 | -0.2546 | -0.0346 |
| search_queries_mean | 2.3262 | 2.1210 | 2.0682 | -0.2580 | -0.0528 |
| parallel_factor_mean | 1.4133 | 1.4608 | 1.4501 | +0.0368 | -0.0107 |
| parallel_sample_rate | 0.4639 | 0.4791 | 0.4637 | -0.0003 | -0.0154 |
| assistant_turns_mean_all_messages | 2.8300 | 2.6120 | 2.5800 | -0.2500 | -0.0320 |
| response_tokens_mean | 341.8300 | 321.5920 | 323.8760 | -17.9540 | +2.2840 |
| zero_search_rate | 0.0003 | 0.0001 | 0.0000 | -0.0003 | -0.0001 |
| no_answer_tag_rate | 0.0030 | 0.0028 | 0.0007 | -0.0023 | -0.0022 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## 2wikimultihopqa

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4450 | 0.4380 | 0.4480 | +0.0030 | +0.0100 |
| retrieval_rounds_mean | 2.3295 | 1.9514 | 1.8360 | -0.4936 | -0.1155 |
| search_queries_mean | 2.8487 | 2.5052 | 2.3576 | -0.4911 | -0.1476 |
| parallel_factor_mean | 1.3341 | 1.4269 | 1.4187 | +0.0846 | -0.0082 |
| parallel_sample_rate | 0.4979 | 0.5336 | 0.5058 | +0.0079 | -0.0278 |
| assistant_turns_mean_all_messages | 3.3100 | 2.9400 | 2.8330 | -0.4770 | -0.1070 |
| response_tokens_mean | 377.4640 | 349.1890 | 345.9280 | -31.5360 | -3.2610 |
| zero_search_rate | 0.0000 | 0.0001 | 0.0000 | +0.0000 | -0.0001 |
| no_answer_tag_rate | 0.0099 | 0.0091 | 0.0010 | -0.0089 | -0.0081 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## musique

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.1780 | 0.1680 | 0.1670 | -0.0110 | -0.0010 |
| retrieval_rounds_mean | 2.6271 | 2.2124 | 2.1530 | -0.4741 | -0.0593 |
| search_queries_mean | 2.9287 | 2.5761 | 2.4956 | -0.4330 | -0.0805 |
| parallel_factor_mean | 1.1848 | 1.2622 | 1.2473 | +0.0624 | -0.0149 |
| parallel_sample_rate | 0.2928 | 0.3426 | 0.3185 | +0.0257 | -0.0241 |
| assistant_turns_mean_all_messages | 3.5990 | 3.1990 | 3.1480 | -0.4510 | -0.0510 |
| response_tokens_mean | 399.4950 | 368.8080 | 366.5770 | -32.9180 | -2.2310 |
| zero_search_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0066 | 0.0058 | 0.0008 | -0.0058 | -0.0050 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## bamboogle

| metric | GAP step120 | E1-A step70 | E1-E step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4640 | 0.4480 | 0.4560 | -0.0080 | +0.0080 |
| retrieval_rounds_mean | 1.9600 | 1.8560 | 1.8640 | -0.0960 | +0.0080 |
| search_queries_mean | 2.1040 | 2.0400 | 2.0320 | -0.0720 | -0.0080 |
| parallel_factor_mean | 1.1267 | 1.1587 | 1.1587 | +0.0320 | -0.0000 |
| parallel_sample_rate | 0.1440 | 0.1840 | 0.1600 | +0.0160 | -0.0240 |
| assistant_turns_mean_all_messages | 2.9600 | 2.8560 | 2.8640 | -0.0960 | +0.0080 |
| response_tokens_mean | 294.0880 | 287.2480 | 290.8800 | -3.2080 | +3.6320 |
| zero_search_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

