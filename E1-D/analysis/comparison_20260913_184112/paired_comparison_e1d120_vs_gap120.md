# Paired per-prompt comparison (E1-D vs matched baseline)

- paired prompts: **49,947**

| metric | E1-D | baseline | delta |
|---|---|---|---|
| search rounds | 1.7270 | 1.8568 | -0.1298 (-6.99%) |
| search queries | 2.2382 | 2.3473 | -0.1090 |
| response chars | 5863.9 | 6091.2 | -227.3 |
| EM | 0.4392 | 0.4450 | -0.0057 |

- rounds **decreased** on 9,781 prompts (19.58%), **increased** on 5,718 (11.45%), unchanged on 34,448
- sign test on round changes: p = 1.27e-233
- EM up on 2,949 / down on 3,235 prompts (sign test p = 0.000276)

## per benchmark

| benchmark | n | rounds E1-D | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 2.155 | 2.330 | -0.175 | 23.5% | 13.5% | -0.0046 |
| bamboogle | 125 | 2.064 | 1.960 | +0.104 | 9.6% | 15.2% | -0.0080 |
| hotpotqa | 7,404 | 1.749 | 1.835 | -0.087 | 16.6% | 10.2% | -0.0076 |
| musique | 2,411 | 2.428 | 2.627 | -0.199 | 28.1% | 16.5% | -0.0012 |
| nq | 3,610 | 1.340 | 1.473 | -0.133 | 17.5% | 8.9% | -0.0019 |
| popqa | 12,509 | 1.585 | 1.720 | -0.135 | 20.5% | 12.3% | -0.0049 |
| triviaqa | 11,312 | 1.364 | 1.454 | -0.089 | 15.2% | 8.7% | -0.0088 |
