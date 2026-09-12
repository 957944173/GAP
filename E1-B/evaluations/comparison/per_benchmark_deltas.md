# E1-B step120 per-benchmark behavior, vs GAP step120 and E1-A step70

## nq

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.3890 | 0.3780 | 0.3810 | -0.0080 | +0.0030 |
| retrieval_rounds_mean | 1.4726 | 1.2640 | 1.0629 | -0.4097 | -0.2011 |
| search_queries_mean | 2.1197 | 1.9230 | 1.8620 | -0.2576 | -0.0609 |
| parallel_factor_mean | 1.5746 | 1.6302 | 1.7886 | +0.2141 | +0.1585 |
| parallel_sample_rate | 0.5241 | 0.5252 | 0.6288 | +0.1047 | +0.1036 |
| assistant_turns_mean_all_messages | 2.4680 | 2.2630 | 2.0620 | -0.4060 | -0.2010 |
| response_tokens_mean | 328.9090 | 313.0530 | 307.9050 | -21.0040 | -5.1480 |
| zero_search_rate | 0.0006 | 0.0003 | 0.0006 | +0.0000 | +0.0003 |
| no_answer_tag_rate | 0.0028 | 0.0014 | 0.0011 | -0.0017 | -0.0003 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## triviaqa

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.5770 | 0.5630 | 0.5620 | -0.0150 | -0.0010 |
| retrieval_rounds_mean | 1.4535 | 1.2838 | 1.0861 | -0.3674 | -0.1977 |
| search_queries_mean | 2.0273 | 1.8619 | 1.7816 | -0.2457 | -0.0803 |
| parallel_factor_mean | 1.5011 | 1.5421 | 1.6845 | +0.1834 | +0.1425 |
| parallel_sample_rate | 0.4610 | 0.4656 | 0.5721 | +0.1111 | +0.1065 |
| assistant_turns_mean_all_messages | 2.4480 | 2.2820 | 2.0850 | -0.3630 | -0.1970 |
| response_tokens_mean | 316.0630 | 302.5670 | 301.2280 | -14.8350 | -1.3390 |
| zero_search_rate | 0.0001 | 0.0001 | 0.0001 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0026 | 0.0011 | 0.0006 | -0.0019 | -0.0005 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## popqa

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4130 | 0.3970 | 0.4030 | -0.0100 | +0.0060 |
| retrieval_rounds_mean | 1.7213 | 1.5006 | 1.1747 | -0.5465 | -0.3259 |
| search_queries_mean | 2.1088 | 1.8912 | 1.6600 | -0.4487 | -0.2312 |
| parallel_factor_mean | 1.3033 | 1.3419 | 1.4626 | +0.1593 | +0.1207 |
| parallel_sample_rate | 0.2998 | 0.3140 | 0.3990 | +0.0993 | +0.0850 |
| assistant_turns_mean_all_messages | 2.7090 | 2.4900 | 2.1700 | -0.5390 | -0.3200 |
| response_tokens_mean | 332.0630 | 312.7790 | 309.5800 | -22.4830 | -3.1990 |
| zero_search_rate | 0.0009 | 0.0008 | 0.0010 | +0.0001 | +0.0002 |
| no_answer_tag_rate | 0.0086 | 0.0089 | 0.0026 | -0.0060 | -0.0063 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## hotpotqa

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4110 | 0.3970 | 0.3900 | -0.0210 | -0.0070 |
| retrieval_rounds_mean | 1.8355 | 1.6155 | 1.3741 | -0.4614 | -0.2414 |
| search_queries_mean | 2.3262 | 2.1210 | 1.9295 | -0.3967 | -0.1915 |
| parallel_factor_mean | 1.4133 | 1.4608 | 1.5362 | +0.1229 | +0.0754 |
| parallel_sample_rate | 0.4639 | 0.4791 | 0.5242 | +0.0602 | +0.0451 |
| assistant_turns_mean_all_messages | 2.8300 | 2.6120 | 2.3730 | -0.4570 | -0.2390 |
| response_tokens_mean | 341.8300 | 321.5920 | 311.5410 | -30.2890 | -10.0510 |
| zero_search_rate | 0.0003 | 0.0001 | 0.0000 | -0.0003 | -0.0001 |
| no_answer_tag_rate | 0.0030 | 0.0028 | 0.0003 | -0.0027 | -0.0026 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## 2wikimultihopqa

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4450 | 0.4380 | 0.4400 | -0.0050 | +0.0020 |
| retrieval_rounds_mean | 2.3295 | 1.9514 | 1.5325 | -0.7970 | -0.4189 |
| search_queries_mean | 2.8487 | 2.5052 | 2.0985 | -0.7502 | -0.4066 |
| parallel_factor_mean | 1.3341 | 1.4269 | 1.5137 | +0.1796 | +0.0868 |
| parallel_sample_rate | 0.4979 | 0.5336 | 0.5426 | +0.0447 | +0.0090 |
| assistant_turns_mean_all_messages | 3.3100 | 2.9400 | 2.5320 | -0.7780 | -0.4080 |
| response_tokens_mean | 377.4640 | 349.1890 | 320.9990 | -56.4650 | -28.1900 |
| zero_search_rate | 0.0000 | 0.0001 | 0.0002 | +0.0002 | +0.0002 |
| no_answer_tag_rate | 0.0099 | 0.0091 | 0.0007 | -0.0092 | -0.0084 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## musique

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.1780 | 0.1680 | 0.1630 | -0.0150 | -0.0050 |
| retrieval_rounds_mean | 2.6271 | 2.2124 | 1.8067 | -0.8204 | -0.4056 |
| search_queries_mean | 2.9287 | 2.5761 | 2.2207 | -0.7080 | -0.3555 |
| parallel_factor_mean | 1.1848 | 1.2622 | 1.3457 | +0.1609 | +0.0835 |
| parallel_sample_rate | 0.2928 | 0.3426 | 0.3845 | +0.0917 | +0.0419 |
| assistant_turns_mean_all_messages | 3.5990 | 3.1990 | 2.8040 | -0.7950 | -0.3950 |
| response_tokens_mean | 399.4950 | 368.8080 | 358.5820 | -40.9130 | -10.2260 |
| zero_search_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0066 | 0.0058 | 0.0008 | -0.0058 | -0.0050 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## bamboogle

| metric | GAP step120 | E1-A step70 | E1-B step120 | d(E1B-GAP120) | d(E1B-E1A70) |
|---|---|---|---|---|---|
| em | 0.4640 | 0.4480 | 0.4240 | -0.0400 | -0.0240 |
| retrieval_rounds_mean | 1.9600 | 1.8560 | 1.6800 | -0.2800 | -0.1760 |
| search_queries_mean | 2.1040 | 2.0400 | 1.9120 | -0.1920 | -0.1280 |
| parallel_factor_mean | 1.1267 | 1.1587 | 1.2200 | +0.0933 | +0.0613 |
| parallel_sample_rate | 0.1440 | 0.1840 | 0.2240 | +0.0800 | +0.0400 |
| assistant_turns_mean_all_messages | 2.9600 | 2.8560 | 2.6800 | -0.2800 | -0.1760 |
| response_tokens_mean | 294.0880 | 287.2480 | 285.0880 | -9.0000 | -2.1600 |
| zero_search_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| no_answer_tag_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

